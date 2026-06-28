#!/usr/bin/env python3
"""[brep env] Baseline for the scaled 10-class (surface+curve) recognition set.
Same geometry features as L1 (PCA eigvals separate 1D curves / 2D patches / 3D).
Usage: python baseline_l4.py <l4_dataset_dir>"""
import json
import os
import sys

import numpy as np


def main():
    d = sys.argv[1]
    X = np.load(os.path.join(d, "points.npy"), mmap_mode="r")  # 2G-safe: not resident
    y = np.load(os.path.join(d, "labels.npy"))
    classes = json.load(open(os.path.join(d, "_DATASET_SUMMARY.json")))["classes"]
    print(f"dataset: {len(y)} tasks, {len(classes)} classes, points {X.shape}")

    # chunked feature extraction -> 2G-safe even at ~120k clouds
    feats = []
    for i in range(0, len(X), 4000):
        Xb = np.asarray(X[i:i + 4000], dtype="float64")
        cov = np.einsum("nij,nik->njk", Xb, Xb) / Xb.shape[1]
        ev = np.linalg.eigvalsh(cov)
        s = ev.sum(1, keepdims=True) + 1e-12
        evn = ev / s
        r = np.linalg.norm(Xb, axis=2)
        feats.append(np.column_stack([
            evn[:, 0], evn[:, 1], evn[:, 2], evn[:, 0] / (evn[:, 2] + 1e-9),
            np.sqrt(np.maximum(ev[:, 0], 0)), r.mean(1), r.std(1)]).astype("float32"))
    F = np.concatenate(feats)
    del X, feats

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    gpath = os.path.join(d, "groups.npy")
    if os.path.exists(gpath) and os.environ.get("BASELINE_RANDOM") != "1":
        G = np.load(gpath)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(F, y, G))
        Xtr, Xte, ytr, yte = F[tr], F[te], y[tr], y[te]
        split_kind = f"by-model (GroupShuffleSplit, {len(set(G))} models, no train/test leak)"
    else:
        Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.2, random_state=0, stratify=y)
        split_kind = "random stratified"
    print(f"split: {split_kind}")
    # bounded forest -> ~400MB (2G box shares ~550MB with AutoDL system services):
    # depth/sample caps keep tree-node memory well under the cgroup limit.
    clf = RandomForestClassifier(n_estimators=100, max_depth=16, max_samples=0.4,
                                 n_jobs=1, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    maj = float((yte == np.bincount(ytr).argmax()).mean())
    present = sorted(set(yte) | set(pred))
    names = [classes[i] for i in present]
    print(f"\n=== L1-scaled baseline ({len(classes)} classes) ===")
    print(f"accuracy {acc:.3f}  majority {maj:.3f}  lift +{acc - maj:.3f}")
    print(classification_report(yte, pred, labels=present, target_names=names,
                                zero_division=0))
    rep = {"n_tasks": int(len(y)), "n_classes": len(classes), "classes": classes,
           "split": split_kind, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "accuracy": round(acc, 4), "majority": round(maj, 4),
           "confusion": confusion_matrix(yte, pred, labels=present).tolist()}
    outname = "_BASELINE_random.json" if os.environ.get("BASELINE_RANDOM") == "1" else "_BASELINE.json"
    json.dump(rep, open(os.path.join(d, outname), "w"), indent=1)
    print(f"wrote {outname}")


if __name__ == "__main__":
    main()
