#!/usr/bin/env python3
"""
Figures for the L1 dataset (headless / Agg — works on the no-display box):
  fig_examples.png   one example point cloud per surface type (3D scatter)
  fig_confusion.png  baseline confusion matrix (if _BASELINE.json present)

Usage: python viz_l1.py <l1_dataset_dir>
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CANON = ["plane", "cylinder", "cone", "sphere", "torus", "freeform"]


def main():
    d = sys.argv[1]
    X = np.load(os.path.join(d, "points.npy"))
    y = np.load(os.path.join(d, "labels.npy"))

    # --- Fig 1: an example cloud per present class ---
    present = [c for c in range(len(CANON)) if (y == c).sum() > 0]
    n = len(present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols                   # 6 classes -> 2x3 grid
    fig = plt.figure(figsize=(4.2 * ncols, 3.8 * nrows))
    for j, c in enumerate(present):
        # pick the example with the most spread (avoids degenerate tiny patches)
        cand = np.where(y == c)[0]
        spreads = X[cand].std(axis=1).sum(axis=1)
        idx = cand[int(np.argmax(spreads))]
        p = X[idx]
        ax = fig.add_subplot(nrows, ncols, j + 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=9, c=p[:, 2], cmap="viridis")
        ax.set_title(f"{CANON[c]}  (n={int((y == c).sum())})", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("GeometryBench L1 — example point clouds by surface type", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(d, "fig_examples.png"), dpi=130)
    print("wrote fig_examples.png")

    # --- Fig 2: confusion matrix from the baseline ---
    bp = os.path.join(d, "_BASELINE.json")
    if os.path.exists(bp):
        b = json.load(open(bp))
        cm = np.array(b["confusion_matrix"], dtype=float)
        names = b["classes_present"]
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig2, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        for i in range(len(names)):
            for k in range(len(names)):
                ax.text(k, i, int(cm[i, k]), ha="center", va="center",
                        fontsize=7, color="0.2")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"L1 baseline confusion (acc={b['test_accuracy']}, "
                     f"majority={b['majority_baseline']})", fontsize=9)
        fig2.colorbar(im, fraction=0.046)
        fig2.tight_layout()
        fig2.savefig(os.path.join(d, "fig_confusion.png"), dpi=120)
        print("wrote fig_confusion.png")
    else:
        print("no _BASELINE.json — skipping confusion matrix")


if __name__ == "__main__":
    main()
