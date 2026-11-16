"""Small, chunked reader/writer for binary PCD files.

The source scan contains extra scalar fields that some Open3D versions do not
parse reliably. This module reads only the fields needed by the detector and
keeps the original file memory-mapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np


_TYPE_MAP = {
    ("F", 4): np.dtype("<f4"),
    ("F", 8): np.dtype("<f8"),
    ("U", 1): np.dtype("u1"),
    ("U", 2): np.dtype("<u2"),
    ("U", 4): np.dtype("<u4"),
    ("U", 8): np.dtype("<u8"),
    ("I", 1): np.dtype("i1"),
    ("I", 2): np.dtype("<i2"),
    ("I", 4): np.dtype("<i4"),
    ("I", 8): np.dtype("<i8"),
}


@dataclass(frozen=True)
class PcdHeader:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    width: int
    height: int
    points: int
    data_type: str
    data_offset: int
    metadata: Mapping[str, str]

    @property
    def point_step(self) -> int:
        return int(sum(size * count for size, count in zip(self.sizes, self.counts)))


def read_pcd_header(path: str | Path) -> PcdHeader:
    """Read only the ASCII header of a PCD file."""

    path = Path(path)
    metadata: Dict[str, str] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header没有 DATA 行: {path}")
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            text = stripped.decode("ascii")
            key, _, value = text.partition(" ")
            metadata[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break

        fields = tuple(metadata["FIELDS"].split())
        sizes = tuple(int(value) for value in metadata["SIZE"].split())
        types = tuple(metadata["TYPE"].split())
        count_values = metadata.get("COUNT", " ".join(["1"] * len(fields))).split()
        counts = tuple(int(value) for value in count_values)

        if len({len(fields), len(sizes), len(types), len(counts)}) != 1:
            raise ValueError(f"PCD 头字段长度不一致: {path}")

        width = int(metadata.get("WIDTH", "0"))
        height = int(metadata.get("HEIGHT", "1"))
        points = int(metadata.get("POINTS", str(width * height)))
        data_type = metadata["DATA"].lower()
        if data_type != "binary":
            raise ValueError(f"目前只支持 DATA binary，实际为 {data_type}: {path}")

        return PcdHeader(
            fields=fields,
            sizes=sizes,
            types=types,
            counts=counts,
            width=width,
            height=height,
            points=points,
            data_type=data_type,
            data_offset=stream.tell(),
            metadata=metadata,
        )


def _structured_dtype(header: PcdHeader) -> np.dtype:
    fields = []
    for name, size, type_code, count in zip(
        header.fields, header.sizes, header.types, header.counts
    ):
        try:
            scalar_dtype = _TYPE_MAP[(type_code.upper(), size)]
        except KeyError as exc:
            raise ValueError(
                f"不支持的 PCD 字段类型: name={name}, TYPE={type_code}, SIZE={size}"
            ) from exc
        if count == 1:
            fields.append((name, scalar_dtype))
        else:
            fields.append((name, scalar_dtype, (count,)))

    dtype = np.dtype(fields, align=False)
    if dtype.itemsize != header.point_step:
        raise ValueError(
            f"PCD 点步长不一致: dtype={dtype.itemsize}, header={header.point_step}"
        )
    return dtype


class BinaryPcd:
    """Memory-mapped binary PCD access with chunked XYZ iteration."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = read_pcd_header(self.path)
        self.dtype = _structured_dtype(self.header)
        expected_bytes = self.header.points * self.header.point_step
        actual_bytes = self.path.stat().st_size - self.header.data_offset
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"PCD 二进制数据长度不足: expected={expected_bytes}, actual={actual_bytes}"
            )
        self._records = np.memmap(
            self.path,
            mode="r",
            dtype=self.dtype,
            offset=self.header.data_offset,
            shape=(self.header.points,),
        )

    @property
    def point_count(self) -> int:
        return self.header.points

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.header.fields

    def has_fields(self, names: Sequence[str]) -> bool:
        return all(name in self._records.dtype.names for name in names)

    def iter_chunks(
        self,
        fields: Sequence[str] = ("x", "y", "z"),
        chunk_points: int = 1_000_000,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield copies of selected fields so downstream code can mutate them."""

        missing = [name for name in fields if name not in self._records.dtype.names]
        if missing:
            raise ValueError(f"PCD 缺少字段 {missing}: {self.path}")
        chunk_points = max(int(chunk_points), 1)
        for start in range(0, self.point_count, chunk_points):
            stop = min(start + chunk_points, self.point_count)
            yield {
                name: np.asarray(self._records[name][start:stop]).copy()
                for name in fields
            }

    def sample(
        self,
        max_points: int,
        fields: Sequence[str] = ("x", "y", "z"),
        seed: int = 7,
    ) -> dict[str, np.ndarray]:
        """Return a deterministic uniform sample without loading all records."""

        missing = [name for name in fields if name not in self._records.dtype.names]
        if missing:
            raise ValueError(f"PCD 缺少字段 {missing}: {self.path}")

        count = min(max(int(max_points), 1), self.point_count)
        if count == self.point_count:
            indices = np.arange(self.point_count, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(self.point_count, size=count, replace=False))

        return {
            name: np.asarray(self._records[name][indices]).copy()
            for name in fields
        }

    def bounds(self, chunk_points: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
        mins = np.full(3, np.inf, dtype=np.float64)
        maxs = np.full(3, -np.inf, dtype=np.float64)
        for chunk in self.iter_chunks(("x", "y", "z"), chunk_points):
            xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
                np.float64, copy=False
            )
            finite = np.isfinite(xyz).all(axis=1)
            if np.any(finite):
                mins = np.minimum(mins, np.min(xyz[finite], axis=0))
                maxs = np.maximum(maxs, np.max(xyz[finite], axis=0))
        if not np.isfinite(mins).all():
            raise ValueError(f"PCD 不包含有效 XYZ 点: {self.path}")
        return mins, maxs


def write_xyz_pcd(path: str | Path, points: np.ndarray) -> None:
    """Write XYZ points as a compact binary PCD."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"XYZ 点数组应为 N×3，实际为 {points.shape}")
    points = points[np.isfinite(points).all(axis=1)]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)

