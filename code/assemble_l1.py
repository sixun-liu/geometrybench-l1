#!/usr/bin/env python3
"""
Assemble the L1 recognition benchmark from per-model extractions.

Each task instance = a point cloud sampled from ONE CAD face + the kernel's
surface-type label. Points are recentred and scaled to a unit sphere so a model
can't cheat on absolute position/scale; every instance is resampled to exactly
K points. Optionally balances classes by capping per-class count.

Outputs (in out_dir):
  points.npy  (N, K, 3) float32   — the clouds, aligned with...
  labels.npy  (N,) int            — class id (index into CANON)
  tasks.jsonl                     — one record/task (model, face, difficulty,
                                    label, params, MCQ options + answer)
  _DATASET_SUMMARY.json           — counts, class distribution, by difficulty

Usage: python assemble_l1.py <extract_dir> <out_dir> [K=256] [max_per_class=0]
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

CANON = ["plane", "cylinder", "cone", "sphere", "torus", "freeform"]
FREEFORM = {"bspline", "bezier", "revolution", "extrusion", "offset", "other"}


def split_of(model_id, test_frac=0.2):
    """Deterministic train/test assignment BY MODEL — all faces of one part go to
    the same split, so a model can't leak from train into test (faces of the same
    part are highly correlated). Stable hash → reproducible."""
    h = int(hashlib.md5(model_id.encode()).hexdigest(), 16)
    return "test" if (h % 1000) < test_frac * 1000 else "train"


def canon(name):
    if name in CANON:
        return name
    if name in FREEFORM:
        return "freeform"
    return None                                   # unknown -> drop


def resample(pts, K, rng):
    n = len(pts)
    if n >= K:
        idx = rng.choice(n, K, replace=False)
    else:
        idx = rng.choice(n, K, replace=True)      # pad small faces by resampling
    return pts[idx]


def normalize(pts):
    c = pts.mean(0)
    p = pts - c
    s = np.linalg.norm(p, axis=1).max()
    return p / s if s > 1e-9 else p


def main():
    ext_dir, out_dir = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    min_pts = 24
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260615)

    inst = []                                     # (cloud[K,3], label, meta)
    for jp in sorted(glob.glob(os.path.join(ext_dir, "*.json"))):
        if os.path.basename(jp).startswith("_"):
            continue
        mid = os.path.splitext(os.path.basename(jp))[0]
        npy = os.path.join(ext_dir, mid + ".npy")
        if not os.path.exists(npy):
            continue
        rec = json.load(open(jp))
        diff = rec.get("difficulty") or "unknown"
        cloud = np.load(npy)                       # (M,5): x,y,z,face_id,type_id
        for fc in rec["faces"]:
            lab = canon(fc["surface_type"])
            if lab is None:
                continue
            pts = cloud[cloud[:, 3] == fc["face_id"]][:, :3]
            if len(pts) < min_pts:
                continue
            cl = normalize(resample(pts, K, rng)).astype("float32")
            inst.append((cl, lab, {
                "model_id": mid, "face_id": fc["face_id"], "difficulty": diff,
                "surface_type": lab, "n_points_orig": int(fc["n_points"]),
                "area": fc["area"], "params": fc["params"],
            }))

    # class balancing (optional)
    if cap > 0:
        by = {}
        for it in inst:
            by.setdefault(it[1], []).append(it)
        for k in by:
            rng.shuffle(by[k])
            by[k] = by[k][:cap]
        inst = [it for k in by for it in by[k]]
    rng.shuffle(inst)

    if not inst:
        print("NO INSTANCES — extraction dir empty or no usable faces?")
        return
    X = np.stack([it[0] for it in inst])
    y = np.array([CANON.index(it[1]) for it in inst], dtype="int64")
    sp = np.array([1 if split_of(it[2]["model_id"]) == "test" else 0
                   for it in inst], dtype="int8")            # by-model split (no leak)
    np.save(os.path.join(out_dir, "points.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    np.save(os.path.join(out_dir, "split.npy"), sp)

    with open(os.path.join(out_dir, "tasks.jsonl"), "w") as f:
        for i, it in enumerate(inst):
            m = it[2]
            f.write(json.dumps({
                "task_id": f"L1-T1.1-{i:06d}", "level": 1, "task": "surface_recognition",
                "question": "Classify the surface type of the CAD face this point "
                            "cloud was sampled from.",
                "options": CANON, "answer": CANON.index(it[1]),
                "answer_label": it[1], "split": ("test" if sp[i] else "train"), **m,
            }) + "\n")

    dist = {c: int((y == CANON.index(c)).sum()) for c in CANON}
    diffs = {}
    for it in inst:
        diffs[it[2]["difficulty"]] = diffs.get(it[2]["difficulty"], 0) + 1
    summary = {
        "n_tasks": len(inst), "n_points_each": K,
        "classes": CANON, "class_distribution": dist,
        "by_difficulty": diffs,
        "split_by_model": {"train": int((sp == 0).sum()), "test": int((sp == 1).sum())},
        "points_shape": list(X.shape),
    }
    json.dump(summary, open(os.path.join(out_dir, "_DATASET_SUMMARY.json"), "w"), indent=1)
    print("DATASET " + json.dumps(summary))


if __name__ == "__main__":
    main()
