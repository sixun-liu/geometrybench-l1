#!/usr/bin/env python3
"""
A light baseline for the L1 surface-recognition benchmark — shows the task is
learnable yet non-trivial. No deep net (the 0.5-core box would crawl): we hand
a handful of geometry-aware features per point cloud to a RandomForest.

Features per (256,3) unit-normalized cloud (already centred at its centroid):
  - 3 normalized covariance eigenvalues (planarity / elongation / isotropy)
  - planarity ratio  e_min / e_max
  - plane-fit residual rms (sqrt of smallest eigenvalue)
  - radial-distance mean & std to the centroid
These separate plane vs cylinder vs sphere etc. by construction, so accuracy >>
the majority-class rate demonstrates the benchmark carries real signal; the
confusion matrix shows where it's genuinely hard (e.g. cone vs cylinder).

Usage: python baseline_l1.py <l1_dataset_dir>
"""
import json
import os
import sys

import numpy as np

CANON = ["plane", "cylinder", "cone", "sphere", "torus", "freeform"]


def features(X):                                   # X: (N,256,3) centred
    cov = np.einsum("nij,nik->njk", X, X) / X.shape[1]
    ev = np.linalg.eigvalsh(cov)                    # ascending (N,3)
    s = ev.sum(1, keepdims=True) + 1e-12
    evn = ev / s
    r = np.linalg.norm(X, axis=2)                   # (N,256)
    return np.column_stack([
        evn[:, 0], evn[:, 1], evn[:, 2],
        evn[:, 0] / (evn[:, 2] + 1e-9),
        np.sqrt(np.maximum(ev[:, 0], 0)),
        r.mean(1), r.std(1),
    ])


def main():
    d = sys.argv[1]
    X = np.load(os.path.join(d, "points.npy"))
    y = np.load(os.path.join(d, "labels.npy"))
    print(f"dataset: {X.shape[0]} tasks, points {X.shape}")

    F = features(X)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    spf = os.path.join(d, "split.npy")                    # prefer the by-model split
    if os.path.exists(spf):
        sp = np.load(spf)
        tr, te = sp == 0, sp == 1
        Xtr, Xte, ytr, yte = F[tr], F[te], y[tr], y[te]
        split_kind = "by-model (no leakage)"
    else:
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.2,
                                              random_state=0, stratify=y)
        split_kind = "random fallback"
    print(f"split: {split_kind}  train={len(ytr)} test={len(yte)}")
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    maj = np.bincount(ytr).argmax()
    majacc = float((yte == maj).mean())
    present = sorted(set(yte) | set(pred))
    names = [CANON[i] for i in present]

    print(f"\n=== L1 baseline (RandomForest on geometry features) ===")
    print(f"test accuracy : {acc:.3f}")
    print(f"majority-class: {majacc:.3f}  (class '{CANON[maj]}')")
    print(f"lift over trivial: +{acc - majacc:.3f}")
    print("\nconfusion matrix (rows=true, cols=pred), classes:", names)
    print(confusion_matrix(yte, pred, labels=present))
    print("\nper-class report:")
    print(classification_report(yte, pred, labels=present,
                                target_names=names, zero_division=0))

    rep = {"n_tasks": int(X.shape[0]), "split": split_kind,
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "test_accuracy": round(acc, 4),
           "majority_baseline": round(majacc, 4), "lift": round(acc - majacc, 4),
           "classes_present": names,
           "confusion_matrix": confusion_matrix(yte, pred, labels=present).tolist()}
    json.dump(rep, open(os.path.join(d, "_BASELINE.json"), "w"), indent=1)
    print("\nwrote", os.path.join(d, "_BASELINE.json"))


if __name__ == "__main__":
    main()
