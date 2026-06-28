#!/usr/bin/env python3
"""
[brep env] Assemble the scaled L1 recognition set: SURFACE patches + CURVE
segments, ~>=10k per class. Surfaces come from extract_batch outputs (*.json +
*.npy), curves from extract_curves outputs (*.cjson + *.cnpy.npy). Each instance
= a point cloud (K pts, normalized to unit sphere) + class label.

Classes (10): plane cylinder cone sphere torus freeform_surf | line circle ellipse bspline_curve

Usage: python assemble_l4.py <out_dir> <K> <cap> <surf_dir1> [surf_dir2 ...] -- <curve_dir1> [...]
"""
import glob
import json
import os
import sys

import numpy as np

SURF = ["plane", "cylinder", "cone", "sphere", "torus", "freeform_surf"]
CURVE = ["line", "circle", "ellipse", "bspline_curve"]
CLASSES = SURF + CURVE
# extract_batch type_id -> surface class
S_ID = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus",
        5: "freeform_surf", 6: "freeform_surf", 7: "freeform_surf",
        8: "freeform_surf", 9: "freeform_surf", 10: "freeform_surf"}
# extract_curves type_id -> curve class (drop hyperbola/parabola/other)
C_ID = {0: "line", 1: "circle", 2: "ellipse", 5: "bspline_curve", 6: "bspline_curve"}
RNG = np.random.default_rng(20260628)
MIN_PTS = 16


def resample(p, K):
    n = len(p)
    idx = RNG.choice(n, K, replace=(n < K))
    return p[idx]


def norm(p):
    c = p.mean(0)
    p = p - c
    s = np.linalg.norm(p, axis=1).max()
    return p / s if s > 1e-9 else p


def harvest(dirs, ext_json, ext_npy, id_map, by_face_key, cap, K, want):
    """Collect per-element clouds for the classes in `want` (dict class->list)."""
    for d in dirs:
        for jp in glob.glob(os.path.join(d, "*" + ext_json)):
            if os.path.basename(jp).startswith("_"):
                continue
            mid = os.path.basename(jp)[:-len(ext_json)]
            npy = os.path.join(d, mid + ext_npy)
            if not os.path.exists(npy):
                continue
            try:
                cloud = np.load(npy)            # (M,5): xyz, elem_id, type_id
            except Exception:
                continue
            for elem_id in np.unique(cloud[:, 3]).astype(int):
                rows = cloud[cloud[:, 3] == elem_id]
                tid = int(rows[0, 4])
                cls = id_map.get(tid)
                if cls is None or len(rows) < MIN_PTS:
                    continue
                if len(want[cls]) >= cap:
                    continue
                # keep the source model id with the cloud -> by-model split later
                want[cls].append((norm(resample(rows[:, :3], K)).astype("float32"), mid))


def main():
    out = sys.argv[1]
    K = int(sys.argv[2])
    cap = int(sys.argv[3])
    rest = sys.argv[4:]
    sep = rest.index("--")
    surf_dirs, curve_dirs = rest[:sep], rest[sep + 1:]
    os.makedirs(out, exist_ok=True)

    want = {c: [] for c in CLASSES}
    harvest(surf_dirs, ".json", ".npy", S_ID, 3, cap, K, want)
    harvest(curve_dirs, ".cjson", ".cnpy.npy", C_ID, 3, cap, K, want)

    X, y, gids = [], [], []
    gmap = {}
    for ci, c in enumerate(CLASSES):
        for cloud, mid in want[c]:
            X.append(cloud)
            y.append(ci)
            if mid not in gmap:
                gmap[mid] = len(gmap)
            gids.append(gmap[mid])
    X = np.stack(X).astype("float32")
    y = np.array(y, dtype="int64")
    G = np.array(gids, dtype="int64")
    perm = RNG.permutation(len(y))
    X, y, G = X[perm], y[perm], G[perm]
    np.save(os.path.join(out, "points.npy"), X)
    np.save(os.path.join(out, "labels.npy"), y)
    np.save(os.path.join(out, "groups.npy"), G)   # source-model id per instance
    dist = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    summary = {"n_tasks": int(len(y)), "n_points_each": K, "classes": CLASSES,
               "class_distribution": dist, "cap": cap, "n_models": len(gmap),
               "points_shape": list(X.shape)}
    json.dump(summary, open(os.path.join(out, "_DATASET_SUMMARY.json"), "w"), indent=1)
    print("DATASET " + json.dumps(summary))


if __name__ == "__main__":
    main()
