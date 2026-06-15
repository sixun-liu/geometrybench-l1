#!/usr/bin/env python3
"""
Derive a HARD variant of the L1 dataset to mimic SCANNED point clouds (the real
reverse-engineering setting): per cloud, simulate self-occlusion (drop a
contiguous chunk along a random view direction, resample), add Gaussian noise,
and re-normalize. Same labels/split — only the input gets harder. Pure
post-processing of the clean dataset; no re-extraction.

Usage: python make_hard.py <clean_dir> <hard_dir> [noise_sigma=0.03] [occlude=0.3]
"""
import json
import os
import shutil
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    occ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(20260616)

    X = np.load(os.path.join(src, "points.npy"))       # (N,K,3) unit-normalized
    N, K, _ = X.shape
    keepn = max(8, int(K * (1 - occ)))
    out = np.empty_like(X)
    for i in range(N):
        p = X[i]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d) + 1e-9                  # random view direction
        kept = p[np.argsort(p @ d)[:keepn]]            # occlude the far side
        q = kept[rng.choice(keepn, K, replace=True)]   # resample to K
        q = q + rng.normal(scale=sigma, size=q.shape)  # sensor noise
        q = q - q.mean(0)                              # re-normalize
        s = np.linalg.norm(q, axis=1).max()
        out[i] = q / s if s > 1e-9 else q
    np.save(os.path.join(dst, "points.npy"), out.astype("float32"))

    for f in ("labels.npy", "split.npy", "tasks.jsonl"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))

    summary = {"variant": "hard", "from": os.path.basename(src), "n_tasks": int(N),
               "noise_sigma": sigma, "occlude_frac": occ,
               "pipeline": "occlude(random view) -> resample -> gaussian noise -> renormalize"}
    json.dump(summary, open(os.path.join(dst, "_HARD_SUMMARY.json"), "w"), indent=1)
    print("HARD " + json.dumps(summary))


if __name__ == "__main__":
    main()
