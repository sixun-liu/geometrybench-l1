#!/usr/bin/env python3
"""
[brep env] HONESTY CHECK — are SYNTHETIC faces/edges distinguishable from REAL
ABC ones *within the same class*? For each class that has both, train a binary
real-vs-synthetic classifier on the SAME geometry features the L1 baseline uses.

  acc ~0.5  -> indistinguishable: synthetic samples are representative (good).
  acc ~1.0  -> a model could 'cheat' by detecting synthetic-ness (disclose).

Only classes with enough of BOTH are tested (cone/sphere/torus are ~all synthetic
by necessity -> not testable, an honest limitation). Usage: python sep_check.py <out_json>
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

W = "/root/autodl-tmp/l4_work"
EXTRACT_OUT = "/root/autodl-tmp/gbench_work/extract_out"
RNG = np.random.default_rng(7)
K, MINPTS, CAP = 128, 16, 2500

# name, json_ext, npy_ext, accepted type_ids, real_dirs, synth_dirs
TESTS = [
    ("plane",         ".json",  ".npy",      [0],               [EXTRACT_OUT], [f"{W}/syn_surf"]),
    ("cylinder",      ".json",  ".npy",      [1],               [EXTRACT_OUT], [f"{W}/syn_surf"]),
    ("freeform_surf", ".json",  ".npy",      [5, 6, 7, 8, 9, 10], [EXTRACT_OUT], [f"{W}/frf_surf"]),
    ("line",          ".cjson", ".cnpy.npy", [0],               [f"{W}/abc_curve"], [f"{W}/syn_curve"]),
    ("circle",        ".cjson", ".cnpy.npy", [1],               [f"{W}/abc_curve"], [f"{W}/syn_curve"]),
    ("bspline_curve", ".cjson", ".cnpy.npy", [5, 6],            [f"{W}/abc_curve"], [f"{W}/syn_curve"]),
]


def resample(p):
    idx = RNG.choice(len(p), K, replace=(len(p) < K))
    return p[idx]


def norm(p):
    p = p - p.mean(0)
    s = np.linalg.norm(p, axis=1).max()
    return p / s if s > 1e-9 else p


def feats(C):
    cov = np.einsum("nij,nik->njk", C, C) / C.shape[1]
    ev = np.linalg.eigvalsh(cov)
    evn = ev / (ev.sum(1, keepdims=True) + 1e-12)
    r = np.linalg.norm(C, axis=2)
    return np.column_stack([evn[:, 0], evn[:, 1], evn[:, 2],
                            evn[:, 0] / (evn[:, 2] + 1e-9),
                            np.sqrt(np.maximum(ev[:, 0], 0)), r.mean(1), r.std(1)])


def gather(dirs, jext, npext, tids, cap):
    out = []
    for d in dirs:
        for jp in sorted(glob.glob(os.path.join(d, "*" + jext))):
            if os.path.basename(jp).startswith("_"):
                continue
            mid = os.path.basename(jp)[:-len(jext)]
            npy = os.path.join(d, mid + npext)
            if not os.path.exists(npy):
                continue
            try:
                cl = np.load(npy)
            except Exception:
                continue
            for eid in np.unique(cl[:, 3]).astype(int):
                rows = cl[cl[:, 3] == eid]
                if int(rows[0, 4]) in tids and len(rows) >= MINPTS:
                    out.append(norm(resample(rows[:, :3])).astype("float32"))
                    if len(out) >= cap:
                        return out
    return out


def main():
    res = {}
    for name, je, ne, tids, rdirs, sdirs in TESTS:
        real = gather(rdirs, je, ne, tids, CAP)
        synth = gather(sdirs, je, ne, tids, CAP)
        if len(real) < 100 or len(synth) < 100:
            res[name] = {"note": f"not testable (real {len(real)}, synth {len(synth)})"}
            print(name, res[name], flush=True)
            continue
        n = min(len(real), len(synth))
        X = np.stack(real[:n] + synth[:n]).astype("float64")
        y = np.array([0] * n + [1] * n)
        F = feats(X)
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.3, random_state=0, stratify=y)
        clf = RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=1,
                                     random_state=0).fit(Xtr, ytr)
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        res[name] = {"n_each": int(n), "real_vs_synth_acc": round(acc, 3), "auc": round(auc, 3)}
        print(name, res[name], flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("SEPCHECK " + json.dumps(res))


if __name__ == "__main__":
    main()
