"""2-D geometry and grid utilities used by the partition detector."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, label
from scipy.spatial import ConvexHull, Delaunay, QhullError


@dataclass
class HoleComponent:
    label: int
    cells: np.ndarray
    area: float
    boundary_support_ratio: float
    touches_outer_boundary: bool


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        if fallback is None:
            raise ValueError("无法归一化零向量")
        return normalize(np.asarray(fallback, dtype=np.float64))
    return vector / norm


def orthogonal_basis(normal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (normal, in-plane horizontal, vertical) as an orthonormal basis."""

    normal = normalize(normal)
    vertical = normalize(vertical)
    horizontal = np.cross(vertical, normal)
    if np.linalg.norm(horizontal) <= 1e-12:
        raise ValueError("隔板法向与竖直方向不能平行")
    horizontal = normalize(horizontal)
    vertical_in_plane = normalize(np.cross(normal, horizontal))
    if np.dot(vertical_in_plane, vertical) < 0:
        vertical_in_plane = -vertical_in_plane
    return normal, horizontal, vertical_in_plane


def polygon_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull_polygon(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("轮廓至少需要 3 个二维点")
    unique = np.unique(points, axis=0)
    hull = ConvexHull(unique)
    return unique[hull.vertices]


def _circumradius(tri_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = tri_points[:, 0]
    p1 = tri_points[:, 1]
    p2 = tri_points[:, 2]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    area2 = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    area = np.maximum(area2 * 0.5, 1e-12)
    radius = a * b * c / (4.0 * area)
    max_edge = np.maximum(np.maximum(a, b), c)
    return radius, max_edge


def _boundary_loops(points: np.ndarray, triangles: np.ndarray) -> list[np.ndarray]:
    """Build edge loops from triangles kept by an alpha shape."""

    if len(triangles) == 0:
        return []
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if len(boundary) == 0:
        return []

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary.tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)

    remaining = {tuple(edge) for edge in boundary.tolist()}
    loops: list[np.ndarray] = []
    while remaining:
        first = next(iter(remaining))
        start, current = first
        loop = [start, current]
        remaining.remove(first)
        previous = start
        while current != start:
            neighbors = adjacency[current]
            candidates = [item for item in neighbors if item != previous]
            next_vertex = None
            for item in candidates:
                edge = tuple(sorted((current, item)))
                if edge in remaining or item == start:
                    next_vertex = item
                    break
            if next_vertex is None:
                break
            edge = tuple(sorted((current, next_vertex)))
            remaining.discard(edge)
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
            if len(loop) > len(boundary) + 2:
                break
        if len(loop) >= 3 and current == start:
            loops.append(points[np.asarray(loop, dtype=np.int64)])
    return loops


def alpha_shape_loops(
    points: np.ndarray,
    max_circumradius: float,
    max_edge_length: float,
    max_input_points: int = 8_000,
    seed: int = 7,
) -> list[np.ndarray]:
    """Return boundary loops of a 2-D alpha shape.

    A fallback to a convex hull is handled by the caller because a hull is
    sometimes the safer outer boundary when sampling is sparse.
    """

    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    points = np.unique(points, axis=0)
    if len(points) < 4:
        return []
    if len(points) > max_input_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_input_points, replace=False)]
    try:
        triangulation = Delaunay(points)
    except QhullError:
        return []

    triangles = triangulation.simplices
    tri_points = points[triangles]
    radii, max_edges = _circumradius(tri_points)
    keep = (radii <= max_circumradius) & (max_edges <= max_edge_length)
    return _boundary_loops(points, triangles[keep])


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test."""

    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        crosses = (ay > y) != (by > y)
        denominator = by - ay
        safe_denominator = denominator if abs(denominator) > 1e-15 else 1e-15
        x_intersection = (bx - ax) * (y - ay) / safe_denominator + ax
        inside ^= crosses & (x < x_intersection)
    return inside


def loops_to_mask(
    loops: Sequence[np.ndarray],
    u_centers: np.ndarray,
    v_centers: np.ndarray,
    subtract_nested: bool = False,
) -> np.ndarray:
    """Fill the largest alpha loop and subtract nested loops."""

    if not loops:
        return np.zeros((len(u_centers), len(v_centers)), dtype=bool)
    mesh_u, mesh_v = np.meshgrid(u_centers, v_centers, indexing="ij")
    grid_points = np.column_stack((mesh_u.ravel(), mesh_v.ravel()))
    areas = np.asarray([polygon_area(loop) for loop in loops])
    outer_index = int(np.argmax(areas))
    outer = loops[outer_index]
    mask = points_in_polygon(grid_points, outer)
    if subtract_nested:
        for index, loop in enumerate(loops):
            if index == outer_index:
                continue
            centroid = np.mean(loop, axis=0)
            if points_in_polygon(centroid[None, :], outer)[0]:
                mask &= ~points_in_polygon(grid_points, loop)
    return mask.reshape(len(u_centers), len(v_centers))


def build_outer_mask(
    boundary_points: np.ndarray,
    u_centers: np.ndarray,
    v_centers: np.ndarray,
    max_circumradius: float,
    max_edge_length: float,
    max_input_points: int,
    min_alpha_to_hull_ratio: float = 0.25,
    mode: str = "auto",
    seed: int = 7,
) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Build an outer panel mask, using alpha shape then convex hull fallback."""

    if str(mode).lower() == "bbox":
        mask = np.ones((len(u_centers), len(v_centers)), dtype=bool)
        low_u, high_u = float(u_centers[0]), float(u_centers[-1])
        low_v, high_v = float(v_centers[0]), float(v_centers[-1])
        bbox = np.array([[low_u, low_v], [high_u, low_v], [high_u, high_v], [low_u, high_v]])
        return mask, [bbox], "bbox"

    loops = alpha_shape_loops(
        boundary_points,
        max_circumradius=max_circumradius,
        max_edge_length=max_edge_length,
        max_input_points=max_input_points,
        seed=seed,
    )
    if loops:
        mask = loops_to_mask(loops, u_centers, v_centers)
        hull = convex_hull_polygon(boundary_points)
        hull_mask = loops_to_mask([hull], u_centers, v_centers)
        hull_cells = max(int(hull_mask.sum()), 1)
        if int(mask.sum()) > 0 and int(mask.sum()) / hull_cells >= min_alpha_to_hull_ratio:
            return mask, loops, "alpha_shape"

    hull = convex_hull_polygon(boundary_points)
    mask = loops_to_mask([hull], u_centers, v_centers)
    return mask, [hull], "convex_hull_fallback"


def find_hole_components(
    outer_mask: np.ndarray,
    occupied_mask: np.ndarray,
    grid_size: float,
    min_cells: int,
    max_cells: int,
    min_boundary_support_ratio: float,
    min_boundary_support_cells: int,
    closing_radius_cells: int = 1,
    debug_components: bool = False,
    reject_touching_outer_boundary: bool = True,
) -> tuple[list[HoleComponent], np.ndarray, np.ndarray]:
    """Find enclosed empty components inside the inferred panel outline."""

    occupied = np.asarray(occupied_mask, dtype=bool) & np.asarray(outer_mask, dtype=bool)
    if closing_radius_cells > 0:
        width = closing_radius_cells * 2 + 1
        structure = np.ones((width, width), dtype=bool)
        occupied_for_labels = binary_closing(occupied, structure=structure)
    else:
        occupied_for_labels = occupied

    empty = outer_mask & ~occupied_for_labels
    labels, count = label(empty, structure=np.ones((3, 3), dtype=np.uint8))
    components: list[HoleComponent] = []
    hole_mask = np.zeros_like(empty, dtype=bool)
    outer_boundary = outer_mask & ~binary_erosion(outer_mask, structure=np.ones((3, 3), dtype=bool))

    for component_label in range(1, count + 1):
        cells = np.argwhere(labels == component_label)
        size = len(cells)
        if size < min_cells or size > max_cells:
            continue
        component_mask = labels == component_label
        boundary = binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool)) & ~component_mask
        support = int(np.count_nonzero(boundary & occupied))
        boundary_size = max(int(np.count_nonzero(boundary)), 1)
        support_ratio = support / boundary_size
        touches_outer = bool(np.any(boundary & ~outer_mask) or np.any(component_mask & outer_boundary))
        if debug_components:
            print(
                f"    空白连通域 label={component_label}, cells={size}, "
                f"support={support}/{boundary_size} ({support_ratio:.3f}), "
                f"touches_outer={touches_outer}"
            )
        if support < min_boundary_support_cells:
            continue
        if support_ratio < min_boundary_support_ratio:
            continue
        if reject_touching_outer_boundary and touches_outer:
            continue

        hole_mask[component_mask] = True
        components.append(
            HoleComponent(
                label=int(component_label),
                cells=cells,
                area=float(size * grid_size * grid_size),
                boundary_support_ratio=float(support_ratio),
                touches_outer_boundary=touches_outer,
            )
        )

    return components, hole_mask, occupied_for_labels

