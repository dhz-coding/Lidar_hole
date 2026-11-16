"""Plane extraction and partition-face grouping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from geometry import normalize


AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


@dataclass
class PlaneCandidate:
    plane_id: int
    axis_name: str
    model: np.ndarray
    inlier_points: np.ndarray
    count: int
    centroid: np.ndarray
    projected_extent: np.ndarray
    residual_rms: float
    source_indices: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        return self.model[:3]

    @property
    def position(self) -> float:
        return float(np.dot(AXIS_VECTORS[self.axis_name], self.centroid))

    @property
    def projected_area(self) -> float:
        return float(self.projected_extent[0] * self.projected_extent[1])


@dataclass
class PlaneGroup:
    group_id: int
    axis_name: str
    members: list[PlaneCandidate] = field(default_factory=list)

    @property
    def axis(self) -> np.ndarray:
        return AXIS_VECTORS[self.axis_name]

    @property
    def position(self) -> float:
        weights = np.asarray([member.count for member in self.members], dtype=float)
        positions = np.asarray([member.position for member in self.members], dtype=float)
        return float(np.average(positions, weights=weights))

    @property
    def count(self) -> int:
        return int(sum(member.count for member in self.members))

    @property
    def points(self) -> np.ndarray:
        if len(self.members) == 1:
            return self.members[0].inlier_points
        return np.vstack([member.inlier_points for member in self.members])

    @property
    def centroid(self) -> np.ndarray:
        weights = np.asarray([member.count for member in self.members], dtype=float)
        centers = np.asarray([member.centroid for member in self.members], dtype=float)
        return np.average(centers, axis=0, weights=weights)

    @property
    def projected_extent(self) -> np.ndarray:
        points = self.points
        axis = self.axis
        in_plane_axes = [index for index in range(3) if abs(axis[index]) < 0.5]
        values = points[:, in_plane_axes]
        return np.ptp(values, axis=0)

    @property
    def projected_area(self) -> float:
        return float(np.prod(self.projected_extent))

    @property
    def projected_bbox(self) -> np.ndarray:
        points = self.points
        axis = self.axis
        in_plane_axes = [index for index in range(3) if abs(axis[index]) < 0.5]
        values = points[:, in_plane_axes]
        return np.array([values.min(axis=0), values.max(axis=0)], dtype=float)


@dataclass
class PartitionCandidate:
    axis_name: str
    group_ids: tuple[int, ...]
    face_groups: tuple[PlaneGroup, ...]
    score: float
    position: float
    thickness: float | None
    reason: str

    @property
    def axis(self) -> np.ndarray:
        return AXIS_VECTORS[self.axis_name]

    @property
    def face_models(self) -> list[np.ndarray]:
        models = []
        for group in self.face_groups:
            models.extend(member.model for member in group.members)
        return models

    @property
    def boundary_points(self) -> np.ndarray:
        return np.vstack([group.points for group in self.face_groups])


def normalize_plane_model(model: Sequence[float], preferred_axis: np.ndarray) -> np.ndarray:
    plane = np.asarray(model, dtype=np.float64)
    if plane.shape != (4,):
        raise ValueError(f"平面模型应为 4 个参数，实际为 {plane.shape}")
    norm = np.linalg.norm(plane[:3])
    if norm <= 1e-12:
        raise ValueError("平面法向量长度为零")
    plane = plane / norm
    if np.dot(plane[:3], preferred_axis) < 0:
        plane = -plane
    return plane


def _downsample_rows(array: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(array) <= max_rows:
        return array, np.arange(len(array), dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(array), size=max_rows, replace=False))
    return array[indices], indices


def _build_cloud(points: np.ndarray):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "平面检测需要 Open3D。请在 Lidar 环境运行，或安装 requirements.txt。"
        ) from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def _candidate_points_for_axis(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    axis: np.ndarray,
    normal_angle_deg: float,
    use_input_normals: bool,
    min_candidate_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_input_normals or normals is None or len(normals) != len(xyz):
        indices = np.arange(len(xyz), dtype=np.int64)
        return xyz, indices
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = normal_norm > 1e-6
    normalized = np.zeros_like(normals, dtype=np.float64)
    normalized[valid] = normals[valid] / normal_norm[valid, None]
    threshold = np.cos(np.deg2rad(normal_angle_deg))
    keep = valid & (np.abs(normalized @ axis) >= threshold)
    if int(np.count_nonzero(keep)) < min_candidate_points:
        indices = np.arange(len(xyz), dtype=np.int64)
        return xyz, indices
    indices = np.flatnonzero(keep).astype(np.int64)
    return xyz[indices], indices


def extract_planes_for_axis(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    axis_name: str,
    config: Mapping[str, object],
) -> list[PlaneCandidate]:
    """Extract multiple RANSAC plane patches whose normals match one axis."""

    axis = AXIS_VECTORS[axis_name]
    max_input_points = int(config.get("max_input_points", 400_000))
    normal_angle_deg = float(config.get("normal_angle_deg", 20.0))
    use_input_normals = bool(config.get("use_input_normals", True))
    min_candidate_points = int(config.get("min_candidate_points", 2_000))
    ransac_distance = float(config.get("ransac_distance_threshold", 0.04))
    ransac_n = int(config.get("ransac_n", 3))
    num_iterations = int(config.get("num_iterations", 3_000))
    max_planes = int(config.get("max_planes", 24))
    min_plane_points = int(config.get("min_plane_points", 300))
    max_face_points = int(config.get("max_face_points", 30_000))
    seed = int(config.get("seed", 7))

    try:
        import open3d as o3d

        o3d.utility.random.seed(seed)
    except ImportError as exc:
        raise RuntimeError(
            "平面检测需要 Open3D。请在 Lidar 环境运行，或安装 requirements.txt。"
        ) from exc

    axis_points, source_indices = _candidate_points_for_axis(
        xyz,
        normals,
        axis,
        normal_angle_deg=normal_angle_deg,
        use_input_normals=use_input_normals,
        min_candidate_points=min_candidate_points,
    )
    axis_points, sample_indices = _downsample_rows(axis_points, max_input_points, seed)
    source_indices = source_indices[sample_indices]
    remaining = axis_points.copy()
    remaining_indices = source_indices.copy()
    planes: list[PlaneCandidate] = []

    for plane_index in range(max_planes):
        if len(remaining) < max(min_plane_points, ransac_n):
            break
        cloud = _build_cloud(remaining)
        model, inliers = cloud.segment_plane(
            distance_threshold=ransac_distance,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_plane_points:
            break
        inlier_indices = np.asarray(inliers, dtype=np.int64)
        plane_points = remaining[inlier_indices]
        model = normalize_plane_model(model, axis)
        distances = plane_points @ model[:3] + model[3]
        centroid = np.mean(plane_points, axis=0)
        in_plane_axes = [index for index in range(3) if abs(axis[index]) < 0.5]
        projected_extent = np.ptp(plane_points[:, in_plane_axes], axis=0)
        sample_points, sample_rows = _downsample_rows(plane_points, max_face_points, seed + plane_index)
        planes.append(
            PlaneCandidate(
                plane_id=plane_index + 1,
                axis_name=axis_name,
                model=model,
                inlier_points=sample_points,
                count=int(len(plane_points)),
                centroid=centroid,
                projected_extent=projected_extent,
                residual_rms=float(np.sqrt(np.mean(distances**2))),
                source_indices=remaining_indices[inlier_indices][sample_rows],
            )
        )
        keep = np.ones(len(remaining), dtype=bool)
        keep[inlier_indices] = False
        remaining = remaining[keep]
        remaining_indices = remaining_indices[keep]

    return planes


def _bbox_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    low = np.maximum(a[0], b[0])
    high = np.minimum(a[1], b[1])
    overlap = np.maximum(high - low, 0.0)
    overlap_area = float(np.prod(overlap))
    area_a = float(np.prod(np.maximum(a[1] - a[0], 0.0)))
    area_b = float(np.prod(np.maximum(b[1] - b[0], 0.0)))
    denominator = max(min(area_a, area_b), 1e-12)
    return overlap_area / denominator


def group_plane_faces(
    planes: Sequence[PlaneCandidate],
    face_merge_distance: float,
    min_bbox_overlap: float,
) -> list[PlaneGroup]:
    """Merge fragments that belong to one physical planar face."""

    if not planes:
        return []
    sorted_planes = sorted(planes, key=lambda item: item.position)
    groups: list[PlaneGroup] = []
    for plane in sorted_planes:
        assigned = None
        for group in groups:
            representative = group.members[0]
            position_close = abs(plane.position - group.position) <= face_merge_distance
            overlap = _bbox_overlap_ratio(plane.inlier_points[:, [0, 2]], representative.inlier_points[:, [0, 2]])
            if plane.axis_name == "x":
                overlap = _bbox_overlap_ratio(plane.inlier_points[:, [1, 2]], representative.inlier_points[:, [1, 2]])
            if position_close and overlap >= min_bbox_overlap:
                assigned = group
                break
        if assigned is None:
            assigned = PlaneGroup(group_id=len(groups) + 1, axis_name=plane.axis_name)
            groups.append(assigned)
        assigned.members.append(plane)
    groups.sort(key=lambda item: item.position)
    for index, group in enumerate(groups, start=1):
        group.group_id = index
    return groups


def _pair_overlap(a: PlaneGroup, b: PlaneGroup) -> float:
    return _bbox_overlap_ratio(a.projected_bbox, b.projected_bbox)


def select_partition_candidates(
    groups: Sequence[PlaneGroup],
    config: Mapping[str, object],
) -> list[PartitionCandidate]:
    """Select interior groups and pair close front/back faces."""

    if not groups:
        return []
    groups = sorted(groups, key=lambda item: item.position)
    end_mode = str(config.get("end_group_mode", "auto")).lower()
    requested_end_groups = config.get("remove_end_groups", None)
    if requested_end_groups is None:
        end_groups = 1 if len(groups) >= 3 else 0
    else:
        end_groups = max(int(requested_end_groups), 0)
        end_groups = min(end_groups, max((len(groups) - 1) // 2, 0))

    if end_mode == "none":
        end_groups = 0
    interior = list(groups[end_groups : len(groups) - end_groups if end_groups else len(groups)])
    if not interior:
        interior = list(groups)

    min_pair_separation = float(config.get("pair_min_separation", 0.03))
    max_pair_separation = float(config.get("pair_max_separation", 0.8))
    min_pair_overlap = float(config.get("pair_min_overlap", 0.35))
    max_candidates = int(config.get("max_partitions", 8))

    candidates: list[PartitionCandidate] = []
    used_groups: set[int] = set()
    for i in range(len(interior)):
        for j in range(i + 1, len(interior)):
            first = interior[i]
            second = interior[j]
            separation = abs(second.position - first.position)
            if not (min_pair_separation <= separation <= max_pair_separation):
                continue
            overlap = _pair_overlap(first, second)
            if overlap < min_pair_overlap:
                continue
            position = 0.5 * (first.position + second.position)
            interior_fraction = 1.0
            if len(groups) > 1:
                span = max(groups[-1].position - groups[0].position, 1e-12)
                edge_distance = min(position - groups[0].position, groups[-1].position - position)
                interior_fraction = float(np.clip(edge_distance / span, 0.0, 1.0))
            extent_score = min(first.projected_area, second.projected_area)
            score = 100.0 * interior_fraction + 10.0 * overlap + np.log1p(extent_score)
            candidates.append(
                PartitionCandidate(
                    axis_name=first.axis_name,
                    group_ids=(first.group_id, second.group_id),
                    face_groups=(first, second),
                    score=float(score),
                    position=float(position),
                    thickness=float(separation),
                    reason="interior_parallel_face_pair",
                )
            )
            used_groups.update((first.group_id, second.group_id))

    for group in interior:
        if group.group_id in used_groups:
            continue
        if len(groups) > 1:
            span = max(groups[-1].position - groups[0].position, 1e-12)
            edge_distance = min(group.position - groups[0].position, groups[-1].position - group.position)
            interior_fraction = float(np.clip(edge_distance / span, 0.0, 1.0))
        else:
            interior_fraction = 0.0
        score = 100.0 * interior_fraction + np.log1p(max(group.projected_area, 0.0))
        candidates.append(
            PartitionCandidate(
                axis_name=group.axis_name,
                group_ids=(group.group_id,),
                face_groups=(group,),
                score=float(score),
                position=float(group.position),
                thickness=None,
                reason="interior_single_face",
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def plane_to_dict(plane: PlaneCandidate) -> dict:
    return {
        "plane_id": int(plane.plane_id),
        "axis_name": plane.axis_name,
        "model": plane.model.tolist(),
        "count": int(plane.count),
        "centroid": plane.centroid.tolist(),
        "position": float(plane.position),
        "projected_extent": plane.projected_extent.tolist(),
        "projected_area": float(plane.projected_area),
        "residual_rms": float(plane.residual_rms),
    }


def group_to_dict(group: PlaneGroup) -> dict:
    return {
        "group_id": int(group.group_id),
        "axis_name": group.axis_name,
        "position": float(group.position),
        "count": int(group.count),
        "projected_extent": group.projected_extent.tolist(),
        "projected_area": float(np.prod(group.projected_extent)),
        "member_plane_ids": [int(member.plane_id) for member in group.members],
    }

