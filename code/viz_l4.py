#!/usr/bin/env python3
"""[brep env] Figures for the scaled 10-class set: (1) point-cloud example grid
(curves render as 1D arcs, surfaces as 2D patches), (2) row-normalized confusion
heatmap from _BASELINE.json. Usage: python viz_l4.py <l4_dataset_dir>"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    d = sys.argv[1]
    X = np.load(os.path.join(d, "points.npy"), mmap_mode="r")  # 2G-safe
    y = np.load(os.path.join(d, "labels.npy"))
    classes = json.load(open(os.path.join(d, "_DATASET_SUMMARY.json")))["classes"]
    rng = np.random.default_rng(0)
    ncol = 3
    is_hard = "hard" in os.path.basename(os.path.normpath(d))

    fig = plt.figure(figsize=(ncol * 2.6, len(classes) * 2.0))
    for ci, c in enumerate(classes):
        idx = np.where(y == ci)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(ncol, len(idx)), replace=False)
        for j, p in enumerate(pick):
            ax = fig.add_subplot(len(classes), ncol, ci * ncol + j + 1, projection="3d")
            pts = np.asarray(X[p])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c=pts[:, 2],
                       cmap="viridis", depthshade=True)
            ax.view_init(elev=18, azim=35)          # fixed informative angle
            ax.set_box_aspect((1, 1, 1))            # fair shape rendering
            ax.set_axis_off()
            if j == 0:
                ax.text2D(-0.12, 0.5, c, transform=ax.transAxes, fontsize=10,
                          rotation=90, va="center", fontweight="bold")
    title = "GeometryBench L1 (scaled, 10 classes >=10k each): point-cloud examples"
    if is_hard:
        title = "GeometryBench L1 scaled — HARD variant (noise sigma=0.03 + 30% occlusion)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(d, "fig_l4_examples.png"), dpi=110)
    print("wrote fig_l4_examples.png")

    try:
        rep = json.load(open(os.path.join(d, "_BASELINE.json")))
        cm = np.array(rep["confusion"], dtype=float)
        if cm.shape == (len(classes), len(classes)):
            cmn = cm / (cm.sum(1, keepdims=True) + 1e-9)
            fig2, ax = plt.subplots(figsize=(7.5, 6.5))
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=8)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_title(f"Confusion (row-norm) — acc {rep['accuracy']} vs maj "
                         f"{rep['majority']}\nsplit: {rep.get('split', '?')}", fontsize=10)
            fig2.tight_layout()
            fig2.savefig(os.path.join(d, "fig_l4_confusion.png"), dpi=110)
            print("wrote fig_l4_confusion.png")
        else:
            print(f"skip confusion fig: cm {cm.shape} != {len(classes)} classes")
    except Exception as e:
        print(f"confusion fig failed: {e}")


if __name__ == "__main__":
    main()
