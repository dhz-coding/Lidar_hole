"""Create lightweight diagnostic PNGs for candidate transverse planes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcd_io import BinaryPcd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", nargs="+", type=float, required=True)
    parser.add_argument("--axis", choices=["x", "y"], default="y")
    parser.add_argument("--top-view", action="store_true")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument("--max-points", type=int, default=250_000)
    args = parser.parse_args()

    reader = BinaryPcd(args.input)
    selected = [[] for _ in args.positions]
    rng = np.random.default_rng(7)
    for chunk in reader.iter_chunks(("x", "y", "z"), 1_000_000):
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(float)
        for index, position in enumerate(args.positions):
            coordinate = xyz[:, 0] if args.axis == "x" else xyz[:, 1]
            points = xyz[np.abs(coordinate - position) <= args.band]
            if len(points):
                selected[index].append(points)

    fig, axes = plt.subplots(1, len(args.positions), figsize=(6 * len(args.positions), 6), squeeze=False)
    for index, (position, chunks) in enumerate(zip(args.positions, selected)):
        ax = axes[0, index]
        points = np.vstack(chunks) if chunks else np.empty((0, 3))
        if len(points) > args.max_points:
            rows = rng.choice(len(points), size=args.max_points, replace=False)
            points = points[rows]
        if args.top_view:
            ax.scatter(points[:, 1], points[:, 0], s=0.2, alpha=0.25, rasterized=True)
            ax.set_xlabel("y")
            ax.set_ylabel("x")
        elif args.hist:
            ax.hist2d(points[:, 0], points[:, 2], bins=240, cmap="viridis", cmin=1)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        elif args.axis == "y":
            ax.scatter(points[:, 0], points[:, 2], s=0.2, alpha=0.25, rasterized=True)
            ax.set_xlabel("x")
        else:
            ax.scatter(points[:, 1], points[:, 2], s=0.2, alpha=0.25, rasterized=True)
            ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(f"{args.axis}={position:.3f}, points={len(points)}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()

