"""Detect partition planes and estimate enclosed hole volumes from a binary PCD.

Run from this directory with:

    python main.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from geometry import (
    build_outer_mask,
    find_hole_components,
    orthogonal_basis,
)
from pcd_io import BinaryPcd, write_xyz_pcd
from plane_detection import (
    AXIS_VECTORS,
    PartitionCandidate,
    PlaneGroup,
    extract_planes_for_axis,
    group_plane_faces,
    group_to_dict,
    plane_to_dict,
    select_partition_candidates,
)


def _as_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    paths = config.setdefault("paths", {})
    for key in ("input_pcd", "output_dir"):
        if key not in paths:
            raise ValueError(f"配置缺少 paths.{key}")
        paths[key] = str(_as_path(paths[key], config_path.parent))
    return config, config_path.parent


def _build_sample(reader: BinaryPcd, config: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray | None]:
    plane_cfg = config.get("plane_detection", {})
    max_points = int(plane_cfg.get("sample_max_points", 400_000))
    seed = int(plane_cfg.get("seed", 7))
    use_normals = bool(plane_cfg.get("use_input_normals", True))
    fields = ["x", "y", "z"]
    normal_fields = ("normal_x", "normal_y", "normal_z")
    if use_normals and reader.has_fields(normal_fields):
        fields.extend(normal_fields)
    sampled = reader.sample(max_points=max_points, fields=fields, seed=seed)
    xyz = np.column_stack((sampled["x"], sampled["y"], sampled["z"])).astype(
        np.float64, copy=False
    )
    finite = np.isfinite(xyz).all(axis=1)
    normals = None
    if all(field in sampled for field in normal_fields):
        normals = np.column_stack([sampled[field] for field in normal_fields]).astype(
            np.float64, copy=False
        )
        finite &= np.isfinite(normals).all(axis=1)
    xyz = xyz[finite]
    if normals is not None:
        normals = normals[finite]
    if len(xyz) < 20:
        raise ValueError("检测样本有效点太少")
    return xyz, normals


def _axis_names(config: Mapping[str, object]) -> list[str]:
    value = str(config.get("coordinate", {}).get("target_normal_axis", "auto")).lower()
    if value == "auto":
        return ["x", "y"]
    if value not in AXIS_VECTORS or value == "z":
        raise ValueError("coordinate.target_normal_axis 只能是 auto、x 或 y")
    return [value]


def _detect_all_axes(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    config: Mapping[str, object],
) -> tuple[dict[str, dict], list[PartitionCandidate]]:
    plane_cfg = config.get("plane_detection", {})
    grouping_cfg = config.get("grouping", {})
    axis_results: dict[str, dict] = {}
    all_candidates: list[PartitionCandidate] = []
    for axis_name in _axis_names(config):
        print(f"\n[平面检测] 目标法向轴 {axis_name}")
        planes = extract_planes_for_axis(xyz, normals, axis_name, plane_cfg)
        groups = group_plane_faces(
            planes,
            face_merge_distance=float(grouping_cfg.get("face_merge_distance", 0.08)),
            min_bbox_overlap=float(grouping_cfg.get("min_face_bbox_overlap", 0.25)),
        )
        candidates = select_partition_candidates(groups, grouping_cfg)
        axis_results[axis_name] = {
            "planes": planes,
            "groups": groups,
            "candidates": candidates,
        }
        all_candidates.extend(candidates)
        print(f"  提取平面 {len(planes)} 个，平面组 {len(groups)} 个，候选隔板 {len(candidates)} 个")
        for group in groups:
            print(
                f"  组 {group.group_id}: position={group.position:.4f}, "
                f"members={[member.plane_id for member in group.members]}, "
                f"extent={group.projected_extent.tolist()}"
            )
        for candidate in candidates:
            print(
                f"  候选 {candidate.group_ids}: score={candidate.score:.3f}, "
                f"thickness={candidate.thickness}, reason={candidate.reason}"
            )
    if not all_candidates:
        raise RuntimeError("没有找到内部隔板候选。请检查轴方向、法向阈值和 RANSAC 参数")
    all_candidates.sort(key=lambda item: item.score, reverse=True)
    return axis_results, all_candidates


def _grid_centers(low: float, high: float, grid_size: float) -> np.ndarray:
    if high <= low:
        high = low + grid_size
    count = max(int(math.ceil((high - low) / grid_size)), 1)
    return low + (np.arange(count, dtype=np.float64) + 0.5) * grid_size


def _accumulate_face_occupancy(
    reader: BinaryPcd,
    face_models: Sequence[np.ndarray],
    horizontal: np.ndarray,
    vertical: np.ndarray,
    u_centers: np.ndarray,
    v_centers: np.ndarray,
    face_band: float,
    chunk_points: int,
) -> np.ndarray:
    counts = np.zeros((len(u_centers), len(v_centers)), dtype=np.uint32)
    u_low = float(u_centers[0] - 0.5 * (u_centers[1] - u_centers[0] if len(u_centers) > 1 else 1.0))
    u_high = float(u_centers[-1] + 0.5 * (u_centers[1] - u_centers[0] if len(u_centers) > 1 else 1.0))
    v_low = float(v_centers[0] - 0.5 * (v_centers[1] - v_centers[0] if len(v_centers) > 1 else 1.0))
    v_high = float(v_centers[-1] + 0.5 * (v_centers[1] - v_centers[0] if len(v_centers) > 1 else 1.0))
    step_u = float(u_centers[1] - u_centers[0]) if len(u_centers) > 1 else 1.0
    step_v = float(v_centers[1] - v_centers[0]) if len(v_centers) > 1 else 1.0

    total_face_points = 0
    for chunk in reader.iter_chunks(("x", "y", "z"), chunk_points):
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
            np.float64, copy=False
        )
        finite = np.isfinite(xyz).all(axis=1)
        if not np.any(finite):
            continue
        xyz = xyz[finite]
        u = xyz @ horizontal
        v = xyz @ vertical
        box = (u >= u_low) & (u < u_high) & (v >= v_low) & (v < v_high)
        if not np.any(box):
            continue
        xyz = xyz[box]
        u = u[box]
        v = v[box]
        face_mask = np.zeros(len(xyz), dtype=bool)
        for model in face_models:
            face_mask |= np.abs(xyz @ model[:3] + model[3]) <= face_band
        if not np.any(face_mask):
            continue
        u = u[face_mask]
        v = v[face_mask]
        iu = np.floor((u - u_low) / step_u).astype(np.int64)
        iv = np.floor((v - v_low) / step_v).astype(np.int64)
        valid = (iu >= 0) & (iu < len(u_centers)) & (iv >= 0) & (iv < len(v_centers))
        if not np.any(valid):
            continue
        flat = iu[valid] * len(v_centers) + iv[valid]
        counts += np.bincount(flat, minlength=counts.size).reshape(counts.shape).astype(
            np.uint32,
            copy=False,
        )
        total_face_points += int(np.count_nonzero(valid))

    print(f"  隔板面带内点数 {total_face_points}")
    return counts


def _candidate_face_positions(candidate: PartitionCandidate, default_thickness: float) -> tuple[float, float, str]:
    if candidate.thickness is not None and candidate.thickness > 0:
        positions = [group.position for group in candidate.face_groups]
        return min(positions), max(positions), "paired_plane_faces"
    if default_thickness <= 0:
        raise ValueError("单面隔板候选没有厚度，请配置 hole.single_face_default_thickness")
    position = candidate.face_groups[0].position
    return position - default_thickness / 2.0, position + default_thickness / 2.0, "configured_single_face_thickness"


def _refine_plane_from_raw(reader: BinaryPcd, model: np.ndarray, axis: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    """Refine a sampled RANSAC plane with all nearby raw XYZ points using TLS."""
    plane_cfg = config.get("plane_detection", {})
    band = float(plane_cfg.get("plane_refine_band", 0.08))
    max_points = int(plane_cfg.get("plane_refine_max_points", 250_000))
    chunks = []
    for chunk in reader.iter_chunks(("x", "y", "z"), int(config.get("io", {}).get("chunk_points", 1_000_000))):
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(np.float64, copy=False)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        keep = np.abs(xyz @ model[:3] + model[3]) <= band
        if np.any(keep):
            chunks.append(xyz[keep])
    if not chunks:
        return model
    points = np.vstack(chunks)
    if len(points) > max_points:
        rng = np.random.default_rng(7)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if np.dot(normal, axis) < 0:
        normal = -normal
    return np.r_[normal, -float(np.dot(normal, center))]


def _cells_to_points(
    cells: np.ndarray,
    u_centers: np.ndarray,
    v_centers: np.ndarray,
    normal: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    s_values: np.ndarray,
) -> np.ndarray:
    if len(cells) == 0 or len(s_values) == 0:
        return np.empty((0, 3), dtype=np.float32)
    uv = np.column_stack((u_centers[cells[:, 0]], v_centers[cells[:, 1]]))
    points = []
    base = uv[:, 0, None] * horizontal[None, :] + uv[:, 1, None] * vertical[None, :]
    for s in s_values:
        points.append(base + float(s) * normal[None, :])
    return np.vstack(points).astype(np.float32, copy=False)


def _extract_partition_review_crop(
    reader: BinaryPcd,
    candidate: PartitionCandidate,
    s_low: float,
    s_high: float,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    normal: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    """Extract raw points around the selected partition for visual review."""

    review_cfg = config.get("review_crop", {})
    horizontal_margin = float(review_cfg.get("horizontal_margin", 0.25))
    vertical_margin = float(review_cfg.get("vertical_margin", 0.25))
    normal_margin = float(review_cfg.get("normal_margin", 0.15))
    chunk_points = int(config.get("io", {}).get("chunk_points", 1_000_000))

    boundary = candidate.boundary_points
    u = boundary @ horizontal
    v = boundary @ vertical
    u_low = float(u.min() - horizontal_margin)
    u_high = float(u.max() + horizontal_margin)
    v_low = float(v.min() - vertical_margin)
    v_high = float(v.max() + vertical_margin)
    s_min = min(s_low, s_high) - normal_margin
    s_max = max(s_low, s_high) + normal_margin

    selected_chunks = []
    for chunk in reader.iter_chunks(("x", "y", "z"), chunk_points):
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
            np.float64, copy=False
        )
        finite = np.isfinite(xyz).all(axis=1)
        if not np.any(finite):
            continue
        xyz = xyz[finite]
        point_u = xyz @ horizontal
        point_v = xyz @ vertical
        point_s = xyz @ normal
        keep = (
            (point_u >= u_low)
            & (point_u <= u_high)
            & (point_v >= v_low)
            & (point_v <= v_high)
            & (point_s >= s_min)
            & (point_s <= s_max)
        )
        if np.any(keep):
            selected_chunks.append(xyz[keep].astype(np.float32, copy=False))

    if not selected_chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(selected_chunks)


def _candidate_to_dict(candidate: PartitionCandidate) -> dict:
    return {
        "axis_name": candidate.axis_name,
        "group_ids": list(candidate.group_ids),
        "score": float(candidate.score),
        "position": float(candidate.position),
        "thickness": None if candidate.thickness is None else float(candidate.thickness),
        "reason": candidate.reason,
    }


def _code_version() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or "nogit"
    except Exception:
        return "nogit"


def _finalize_run_outputs(run_dir: Path, candidate: PartitionCandidate, report: dict, config: Mapping[str, object]) -> None:
    selected_dir = Path(report["outputs"]["candidate_dir"])
    if selected_dir is None:
        raise RuntimeError("没有找到候选输出目录")

    face_groups = candidate.face_groups
    for index, group in enumerate(face_groups[:2], start=1):
        write_xyz_pcd(run_dir / f"partition_face_{index:02d}.pcd", group.points)
    if len(face_groups) == 1:
        estimated = face_groups[0].points + candidate.axis[None, :] * float(report["thickness_m"])
        write_xyz_pcd(run_dir / "partition_face_02_estimated.pcd", estimated)
    write_xyz_pcd(run_dir / "partition_plane_group.pcd", candidate.boundary_points)

    raw_crop = selected_dir / "partition_crop_raw.pcd"
    if raw_crop.exists():
        shutil.copy2(raw_crop, run_dir / "partition_slab_points.pcd")
    for hole in report.get("holes", []):
        if hole.get("excluded_by_area", False):
            continue
        source = Path(hole["voxel_output"])
        target = run_dir / f"hole_{int(hole['hole_index']):02d}_pseudo_points.pcd"
        if source.exists():
            shutil.copy2(source, target)
        cuboid_source = Path(hole["cuboid_output"])
        cuboid_target = run_dir / f"hole_{int(hole['hole_index']):02d}_cuboid_vertices.pcd"
        if cuboid_source.exists():
            shutil.copy2(cuboid_source, cuboid_target)

    for path in run_dir.rglob("*.pcd"):
        if path.parent == run_dir and path.name in {
            "partition_face_01.pcd",
            "partition_face_02.pcd",
            "partition_face_02_estimated.pcd",
            "partition_plane_group.pcd",
            "partition_slab_points.pcd",
        }:
            continue
        if path.parent == run_dir and path.name.startswith("hole_") and path.name.endswith("_pseudo_points.pcd"):
            continue
        if path.parent == run_dir and path.name.startswith("hole_") and path.name.endswith("_cuboid_vertices.pcd"):
            continue
        path.unlink()
    for path in sorted(run_dir.glob("candidate_*")):
        if path.is_dir():
            shutil.rmtree(path)
    for path in run_dir.glob("*.json"):
        path.unlink()
    log_lines = [
        f"code_version: {_code_version()}",
        f"generated_at: {datetime.now().astimezone().isoformat()}",
        f"input_pcd: {config['paths']['input_pcd']}",
        f"selected_axis: {candidate.axis_name}",
        f"selected_groups: {candidate.group_ids}",
        f"selected_position_m: {candidate.position:.6f}",
        f"thickness_m: {report['thickness_m']:.6f}",
        f"thickness_source: {report['thickness_source']}",
        f"hole_count: {report['hole_count']}",
        f"kept_hole_count: {sum(not item.get('excluded_by_area', False) for item in report['holes'])}",
        f"excluded_hole_count: {sum(item.get('excluded_by_area', False) for item in report['holes'])}",
        f"exclude_area_m2: {config.get('hole', {}).get('exclude_area_m2', 1.0)}",
        f"total_volume_m3: {report['total_volume_m3']:.6f}",
        f"rough_total_volume_m3: {report.get('rough_total_volume_m3', report['total_volume_m3']):.6f}",
        "outputs: partition_face_01.pcd, partition_face_02.pcd or partition_face_02_estimated.pcd, partition_plane_group.pcd, partition_slab_points.pcd",
    ]
    for hole in report.get("holes", []):
        if hole.get("excluded_by_area", False):
            continue
        source = run_dir / f"hole_{int(hole['hole_index']):02d}_pseudo_points.pcd"
        if source.exists():
            hole_cloud = BinaryPcd(source)
            bbox_min, bbox_max = hole_cloud.bounds()
            size = bbox_max - bbox_min
            log_lines.append(
                f"hole_{int(hole['hole_index']):02d}_bbox_min: "
                f"[{bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f}]"
            )
            log_lines.append(
                f"hole_{int(hole['hole_index']):02d}_bbox_max: "
                f"[{bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f}]"
            )
            log_lines.append(
                f"hole_{int(hole['hole_index']):02d}_size_xyz_m: "
                f"[{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]"
            )
            log_lines.append(
                f"hole_{int(hole['hole_index']):02d}_cuboid_lwh_m: "
                f"[{hole['cuboid_length_m']:.6f}, {hole['cuboid_thickness_m']:.6f}, {hole['cuboid_height_m']:.6f}]"
            )
            log_lines.append(
                f"hole_{int(hole['hole_index']):02d}_cuboid_volume_m3: "
                f"{hole['cuboid_volume_m3']:.6f}"
            )
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def analyse_candidate(
    reader: BinaryPcd,
    candidate: PartitionCandidate,
    config: Mapping[str, object],
    output_dir: Path,
    candidate_index: int,
) -> dict:
    coord_cfg = config.get("coordinate", {})
    hole_cfg = config.get("hole", {})
    grid_size = float(hole_cfg.get("grid_size", 0.04))
    face_band = float(hole_cfg.get("face_band", 0.06))
    boundary_margin = float(hole_cfg.get("boundary_margin", 0.1))
    vertical_axis_name = str(coord_cfg.get("vertical_axis", "z")).lower()
    vertical = AXIS_VECTORS[vertical_axis_name]
    normal, horizontal, vertical_in_plane = orthogonal_basis(candidate.axis, vertical)

    refined_models = [
        _refine_plane_from_raw(reader, group.members[0].model, candidate.axis, config)
        for group in candidate.face_groups
    ]
    refined_positions = [
        -float(model[3]) / max(float(np.dot(model[:3], candidate.axis)), 1e-9)
        for model in refined_models
    ]
    if len(refined_positions) >= 2:
        s_low = min(refined_positions)
        s_high = max(refined_positions)
        thickness_source = "refined_raw_plane_pair"
    else:
        s_low, s_high, thickness_source = _candidate_face_positions(
            candidate, float(hole_cfg.get("single_face_default_thickness", 0.25))
        )
    thickness = float(s_high - s_low)
    if thickness <= 0:
        raise ValueError("重拟合后的隔板面顺序或厚度无效")

    boundary_points = candidate.boundary_points
    boundary_u = boundary_points @ horizontal
    boundary_v = boundary_points @ vertical_in_plane
    u_low = float(np.min(boundary_u) - boundary_margin)
    u_high = float(np.max(boundary_u) + boundary_margin)
    v_low = float(np.min(boundary_v) - boundary_margin)
    v_high = float(np.max(boundary_v) + boundary_margin)
    u_centers = _grid_centers(u_low, u_high, grid_size)
    v_centers = _grid_centers(v_low, v_high, grid_size)

    outer_mask, loops, outer_method = build_outer_mask(
        np.column_stack((boundary_u, boundary_v)),
        u_centers,
        v_centers,
        max_circumradius=float(hole_cfg.get("alpha_max_circumradius", 0.8)),
        max_edge_length=float(hole_cfg.get("alpha_max_edge_length", 1.0)),
        max_input_points=int(hole_cfg.get("alpha_max_input_points", 8_000)),
        min_alpha_to_hull_ratio=float(hole_cfg.get("min_alpha_to_hull_ratio", 0.25)),
        mode=str(hole_cfg.get("outer_mode", "auto")),
        seed=int(config.get("plane_detection", {}).get("seed", 7)),
    )
    face_models = refined_models
    occupied_counts = _accumulate_face_occupancy(
        reader,
        face_models=face_models,
        horizontal=horizontal,
        vertical=vertical_in_plane,
        u_centers=u_centers,
        v_centers=v_centers,
        face_band=face_band,
        chunk_points=int(config.get("io", {}).get("chunk_points", 1_000_000)),
    )
    occupied_mask = occupied_counts >= int(hole_cfg.get("occupied_min_points", 1))
    components, hole_mask, occupied_for_labels = find_hole_components(
        outer_mask=outer_mask,
        occupied_mask=occupied_mask,
        grid_size=grid_size,
        min_cells=int(hole_cfg.get("min_hole_cells", 20)),
        max_cells=int(hole_cfg.get("max_hole_cells", 200_000)),
        min_boundary_support_ratio=float(hole_cfg.get("min_boundary_support_ratio", 0.15)),
        min_boundary_support_cells=int(hole_cfg.get("min_boundary_support_cells", 8)),
        closing_radius_cells=int(hole_cfg.get("closing_radius_cells", 1)),
        debug_components=bool(config.get("debug", {}).get("print_components", False)),
        reject_touching_outer_boundary=bool(
            hole_cfg.get("reject_touching_outer_boundary", True)
        ),
    )

    volume_voxel_size = float(hole_cfg.get("volume_voxel_size", grid_size))
    layers = max(1, int(math.ceil(thickness / max(volume_voxel_size, 1e-6))))
    s_values = s_low + (np.arange(layers, dtype=np.float64) + 0.5) * thickness / layers

    candidate_dir = output_dir / f"candidate_{candidate_index:02d}_{candidate.axis_name}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for old_hole in candidate_dir.glob("hole_*_volume_voxels.pcd"):
        old_hole.unlink()
    face_points = candidate.boundary_points.astype(np.float32, copy=False)
    write_xyz_pcd(candidate_dir / "partition_face_points.pcd", face_points)

    review_crop = _extract_partition_review_crop(
        reader,
        candidate,
        s_low=s_low,
        s_high=s_high,
        horizontal=horizontal,
        vertical=vertical_in_plane,
        normal=normal,
        config=config,
    )
    review_crop_path = candidate_dir / "partition_crop_raw.pcd"
    write_xyz_pcd(review_crop_path, review_crop)

    combined_hole_mask = np.zeros_like(hole_mask, dtype=bool)
    hole_reports = []
    total_volume = 0.0
    rough_total_volume = 0.0
    exclude_area_m2 = float(hole_cfg.get("exclude_area_m2", 1.0))
    exclude_rule = str(hole_cfg.get("exclude_area_rule", "less_equal")).lower()
    for hole_index, component in enumerate(components, start=1):
        combined_hole_mask[component.cells[:, 0], component.cells[:, 1]] = True
        hole_points = _cells_to_points(
            component.cells,
            u_centers,
            v_centers,
            normal,
            horizontal,
            vertical_in_plane,
            s_values,
        )
        hole_path = candidate_dir / f"hole_{hole_index:02d}_volume_voxels.pcd"
        write_xyz_pcd(hole_path, hole_points)
        u_min = float(u_centers[component.cells[:, 0]].min() - grid_size / 2.0)
        u_max = float(u_centers[component.cells[:, 0]].max() + grid_size / 2.0)
        v_min = float(v_centers[component.cells[:, 1]].min() - grid_size / 2.0)
        v_max = float(v_centers[component.cells[:, 1]].max() + grid_size / 2.0)
        cuboid_length = u_max - u_min
        cuboid_height = v_max - v_min
        cuboid_volume = cuboid_length * cuboid_height * thickness
        volume = cuboid_volume
        rough_total_volume += volume
        excluded = (
            component.area <= exclude_area_m2
            if exclude_rule in {"less_equal", "le", "below"}
            else component.area >= exclude_area_m2
        )
        if not excluded:
            total_volume += volume
        cuboid_vertices = []
        for s in (s_low, s_high):
            for vv in (v_min, v_max):
                for uu in (u_min, u_max):
                    cuboid_vertices.append(
                        horizontal * uu + vertical_in_plane * vv + normal * s
                    )
        cuboid_path = candidate_dir / f"hole_{hole_index:02d}_cuboid_vertices.pcd"
        write_xyz_pcd(cuboid_path, np.asarray(cuboid_vertices, dtype=np.float32))
        hole_reports.append(
            {
                "hole_index": hole_index,
                "grid_cells": int(len(component.cells)),
                "area_m2": float(component.area),
                "thickness_m": thickness,
                "volume_m3": float(volume),
                "cuboid_length_m": float(cuboid_length),
                "cuboid_height_m": float(cuboid_height),
                "cuboid_thickness_m": float(thickness),
                "cuboid_volume_m3": float(cuboid_volume),
                "excluded_by_area": bool(excluded),
                "exclude_area_m2": exclude_area_m2,
                "boundary_support_ratio": component.boundary_support_ratio,
                "touches_outer_boundary": component.touches_outer_boundary,
                "uv_bbox": [
                    [
                        float(u_centers[component.cells[:, 0]].min()),
                        float(v_centers[component.cells[:, 1]].min()),
                    ],
                    [
                        float(u_centers[component.cells[:, 0]].max()),
                        float(v_centers[component.cells[:, 1]].max()),
                    ],
                ],
                "voxel_output": str(hole_path),
                "cuboid_output": str(cuboid_path),
            }
        )

    ideal_surface_mask = outer_mask & ~combined_hole_mask
    ideal_surface_cells = np.argwhere(ideal_surface_mask)
    ideal_surface_points = _cells_to_points(
        ideal_surface_cells,
        u_centers,
        v_centers,
        normal,
        horizontal,
        vertical_in_plane,
        np.asarray([s_low, s_high], dtype=np.float64),
    )
    ideal_path = candidate_dir / "partition_ideal_faces.pcd"
    write_xyz_pcd(ideal_path, ideal_surface_points)

    if bool(config.get("debug", {}).get("write_grid_png", False)):
        try:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            panels = [
                (outer_mask, "outer mask"),
                (occupied_for_labels, "occupied"),
                (hole_mask, "accepted holes"),
            ]
            for axis_plot, (mask, title) in zip(axes, panels):
                axis_plot.imshow(
                    mask.T,
                    origin="lower",
                    extent=[u_centers[0], u_centers[-1], v_centers[0], v_centers[-1]],
                    aspect="equal",
                    interpolation="nearest",
                )
                axis_plot.set_title(title)
                axis_plot.set_xlabel("in-plane horizontal (m)")
                axis_plot.set_ylabel("vertical (m)")
            figure.savefig(candidate_dir / "grid_diagnostic.png", dpi=180)
            plt.close(figure)
        except ImportError:
            print("  未安装 matplotlib，跳过 grid_diagnostic.png")

    grid_report = {
        "grid_size_m": grid_size,
        "u_cells": int(len(u_centers)),
        "v_cells": int(len(v_centers)),
        "outer_method": outer_method,
        "outer_cells": int(np.count_nonzero(outer_mask)),
        "occupied_cells": int(np.count_nonzero(occupied_for_labels)),
        "hole_cells": int(np.count_nonzero(hole_mask)),
        "face_band_m": face_band,
        "u_range": [float(u_low), float(u_high)],
        "v_range": [float(v_low), float(v_high)],
    }
    selection_cfg = config.get("selection", {})
    selection_score = float(candidate.score)
    if candidate.thickness is not None:
        selection_score += float(selection_cfg.get("paired_face_bonus", 100.0))
    selection_score += float(selection_cfg.get("hole_area_weight", 1000.0)) * sum(
        item["area_m2"] for item in hole_reports
    )
    selection_score += float(selection_cfg.get("hole_count_weight", 25.0)) * len(hole_reports)
    position_hint = selection_cfg.get("position_hint", None)
    if position_hint is not None:
        hint_scale = max(float(selection_cfg.get("position_hint_scale", 2.0)), 1e-6)
        hint_bonus = float(selection_cfg.get("position_hint_bonus", 500.0))
        selection_score += hint_bonus * math.exp(
            -abs(candidate.position - float(position_hint)) / hint_scale
        )
    result = {
        "candidate": _candidate_to_dict(candidate),
        "normal": normal.tolist(),
        "horizontal_axis": horizontal.tolist(),
        "vertical_axis": vertical_in_plane.tolist(),
        "face_positions_m": [float(s_low), float(s_high)],
        "thickness_m": thickness,
        "thickness_source": thickness_source,
        "grid": grid_report,
        "holes": hole_reports,
        "hole_count": len(hole_reports),
        "total_volume_m3": float(total_volume),
        "rough_total_volume_m3": float(rough_total_volume),
        "selection_score": float(selection_score),
        "outputs": {
            "candidate_dir": str(candidate_dir),
            "partition_face_points": str(candidate_dir / "partition_face_points.pcd"),
            "partition_crop_raw": str(review_crop_path),
            "partition_crop_raw_points": int(len(review_crop)),
            "partition_ideal_faces": str(ideal_path),
        },
    }
    with (candidate_dir / "candidate_report.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        f"[候选 {candidate_index}] axis={candidate.axis_name}, groups={candidate.group_ids}, "
        f"holes={len(hole_reports)}, total_volume={total_volume:.6f} m3, "
        f"selection_score={selection_score:.3f}"
    )
    return result


def build_report(
    reader: BinaryPcd,
    bounds: tuple[np.ndarray, np.ndarray],
    sample: np.ndarray,
    axis_results: Mapping[str, dict],
    selected: Sequence[dict],
    output_dir: Path,
) -> dict:
    axis_report = {}
    for axis_name, result in axis_results.items():
        axis_report[axis_name] = {
            "planes": [plane_to_dict(plane) for plane in result["planes"]],
            "groups": [group_to_dict(group) for group in result["groups"]],
            "candidates": [_candidate_to_dict(candidate) for candidate in result["candidates"]],
        }
    return {
        "input_pcd": str(reader.path),
        "point_count": int(reader.point_count),
        "field_names": list(reader.field_names),
        "bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "detection_sample_points": int(len(sample)),
        "axis_results": axis_report,
        "selected_candidates": selected,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据内部隔板平面和空白区域估计隔板孔洞体积")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置路径")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
        help="指定处理排序后的候选序号。省略时按 YAML 的 selection.mode 自动选择，-1 处理全部候选。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    paths = config["paths"]
    input_path = Path(paths["input_pcd"])
    output_root = Path(paths["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_v{_code_version()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"输入点云不存在: {input_path}")
    print(f"读取 PCD 头: {input_path}")
    reader = BinaryPcd(input_path)
    print(f"点数={reader.point_count}, 字段={reader.field_names}")
    bounds = reader.bounds(int(config.get("io", {}).get("chunk_points", 1_000_000)))
    print(f"全云范围 min={bounds[0]}, max={bounds[1]}")
    sample, normals = _build_sample(reader, config)
    print(f"检测样本={len(sample)}, 法向字段={'有' if normals is not None else '无'}")

    axis_results, candidates = _detect_all_axes(sample, normals, config)
    report = build_report(reader, bounds, sample, axis_results, [], run_dir)
    with (run_dir / "plane_detection_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    if args.candidate_index == -1:
        selected_candidates = candidates
        selection_mode = "all"
    elif args.candidate_index is not None:
        if args.candidate_index < 0 or args.candidate_index >= len(candidates):
            raise IndexError(f"candidate-index 超出范围，候选数={len(candidates)}")
        selected_candidates = [candidates[args.candidate_index]]
        selection_mode = "explicit"
    else:
        selection_cfg = config.get("selection", {})
        selection_mode = str(selection_cfg.get("mode", "auto_hole_evidence")).lower()
        if selection_mode == "plane_score":
            selected_candidates = [candidates[0]]
        else:
            preferred_axis = str(selection_cfg.get("preferred_axis", "auto")).lower()
            if preferred_axis in AXIS_VECTORS and preferred_axis != "z":
                preferred = [candidate for candidate in candidates if candidate.axis_name == preferred_axis]
            else:
                preferred = list(candidates)
            candidates_for_evaluation = preferred if preferred else list(candidates)
            if bool(selection_cfg.get("require_paired_faces", True)):
                paired = [candidate for candidate in candidates_for_evaluation if candidate.thickness is not None]
            else:
                paired = list(candidates_for_evaluation)
            if paired:
                candidates_for_evaluation = paired
            max_evaluate = min(
                len(candidates_for_evaluation),
                max(int(selection_cfg.get("evaluate_max_candidates", 8)), 1),
            )
            selected_candidates = candidates_for_evaluation[:max_evaluate]

    selected_reports = []
    for index, candidate in enumerate(selected_candidates):
        selected_reports.append(analyse_candidate(reader, candidate, config, run_dir, index))

    if selection_mode == "auto_hole_evidence" and selected_reports:
        best_index = int(np.argmax([item["selection_score"] for item in selected_reports]))
        selected_reports[best_index]["auto_selected"] = True
        report["auto_selected_candidate_index"] = best_index
        report["auto_selected_total_volume_m3"] = selected_reports[best_index]["total_volume_m3"]
        print(
            f"\n自动选择候选 {best_index}: "
            f"体积={selected_reports[best_index]['total_volume_m3']:.6f} m3"
        )
    report["selection_mode"] = selection_mode
    report["selected_candidates"] = selected_reports
    with (run_dir / "hole_volume_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    best_index = int(report.get("auto_selected_candidate_index", 0))
    _finalize_run_outputs(run_dir, selected_candidates[best_index], selected_reports[best_index], config)
    print(f"\n结果目录: {run_dir}")
    print(f"日志文件: {run_dir / 'run.log'}")
    print((run_dir / "run.log").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise

