#!/usr/bin/env python3
"""
[brep env] Reconstruction gallery: GT mesh vs cadrille reconstruction for a few
best- and worst-Chamfer examples (honest spread). Headless (Agg).
Usage: python viz_recon.py <abc_out_dir> <gt_stl_dir>
"""
import json
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_norm(path):
    m = trimesh.load(path, force="mesh")
    v = np.asarray(m.vertices)
    c = (v.min(0) + v.max(0)) / 2
    v = v - c
    s = (v.max(0) - v.min(0)).max()
    v = v / s if s > 1e-9 else v
    return v, np.asarray(m.faces)


def show(ax, path, title):
    try:
        v, f = load_norm(path)
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                        color="#4c9be8", edgecolor="none", alpha=0.9)
    except Exception:
        ax.text(0.5, 0.5, 0.5, "—", ha="center")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))


def main():
    out_dir, gt_dir = sys.argv[1], sys.argv[2]
    res = json.load(open(os.path.join(out_dir, "mesh", "_RECON_RESULTS.json")))
    pm = [r for r in res["per_model"] if r.get("chamfer") is not None]
    pm.sort(key=lambda r: r["chamfer"])
    picks = pm[:3] + pm[-3:]          # 3 best + 3 worst
    labels = ["best"] * 3 + ["worst"] * 3

    n = len(picks)
    fig = plt.figure(figsize=(2.6 * n, 5.4))
    for j, (r, lab) in enumerate(zip(picks, labels)):
        mid = r["id"]
        axg = fig.add_subplot(2, n, j + 1, projection="3d")
        show(axg, os.path.join(gt_dir, mid + ".stl"), f"GT ({lab})")
        axp = fig.add_subplot(2, n, n + j + 1, projection="3d")
        show(axp, os.path.join(out_dir, "mesh", mid + ".stl"),
             f"cadrille\nCD={r['chamfer']:.3f} F={r['fscore']:.2f}")
    s = res["summary"]
    fig.suptitle(f"GeometryBench — cadrille-rl point-cloud→B-Rep on ABC  "
                 f"(exec {s['exec_success_rate']:.0%}, valid {s['brep_valid_rate']:.0%}, "
                 f"F@0.02 {s['mean_fscore@0.02']:.2f}, median CD {s['median_chamfer']:.3f})",
                 fontsize=10)
    fig.text(0.5, 0.50, "↑ ground truth      ↓ cadrille reconstruction",
             ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(out_dir, "fig_recon_gallery.png")
    fig.savefig(out, dpi=125)
    print("wrote", out)


if __name__ == "__main__":
    main()
