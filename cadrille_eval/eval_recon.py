#!/usr/bin/env python3
"""
[brep env] Reconstruction metrics for the cadrille eval: prediction vs GT.

For each model:
  - Chamfer distance + F-score@tau  : pred mesh vs GT mesh (both sampled to N
    surface points and re-normalized to a unit cube, since cadrille output is at
    a ~±100 scale while GT is unit — align frames before comparing).
  - exec-success                    : did cadrille's code produce a usable mesh?
  - B-Rep validity                  : load the predicted STEP with the kernel
    (BRepCheck) — watertight/closed solid?  (curve/surface count too)

Aggregates -> _RECON_RESULTS.json + stdout.
Needs: trimesh, scipy (both already in `brep` env via sklearn), pythonocc.
Usage: python eval_recon.py <gt_stl_dir> <pred_mesh_dir> [pred_step_dir] [n_points]
"""
import glob
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def norm_unit(p):
    c = (p.min(0) + p.max(0)) / 2.0
    p = p - c
    s = (p.max(0) - p.min(0)).max()
    return p / s if s > 1e-9 else p


def sample(path, n):
    m = trimesh.load(path, force="mesh")
    if m.vertices.shape[0] < 4 or m.area <= 0:
        return None
    pts, _ = trimesh.sample.sample_surface(m, n)
    return norm_unit(np.asarray(pts))


def chamfer_f(pred, gt, tau=0.02):
    tp, tg = cKDTree(pred), cKDTree(gt)
    dp, _ = tg.query(pred)        # pred -> gt
    dg, _ = tp.query(gt)          # gt -> pred
    chamfer = float(dp.mean() + dg.mean())
    prec = float((dp < tau).mean())
    rec = float((dg < tau).mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return chamfer, f


def step_valid(path):
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        r = STEPControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            return False
        r.TransferRoots()
        s = r.OneShape()
        if s.IsNull():
            return False
        return bool(BRepCheck_Analyzer(s).IsValid())
    except Exception:
        return False


def main():
    gt_dir, pred_mesh_dir = sys.argv[1], sys.argv[2]
    pred_step_dir = sys.argv[3] if len(sys.argv) > 3 else ""
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 4096

    gts = sorted(glob.glob(os.path.join(gt_dir, "*.stl")))
    rows, chamfers, fs = [], [], []
    n_exec = n_valid = n_total = 0
    for g in gts:
        mid = os.path.splitext(os.path.basename(g))[0]
        n_total += 1
        # pred mesh may be named <mid>.stl or <mid>+0.stl etc.
        cand = glob.glob(os.path.join(pred_mesh_dir, mid + "*")) if pred_mesh_dir else []
        cand = [c for c in cand if c.lower().endswith((".stl", ".ply", ".obj"))]
        rec = {"id": mid, "exec_ok": False, "chamfer": None, "fscore": None, "valid": None}
        if cand:
            pp = sample(cand[0], n)
            gp = sample(g, n)
            if pp is not None and gp is not None:
                n_exec += 1
                rec["exec_ok"] = True
                cd, f = chamfer_f(pp, gp)
                rec["chamfer"], rec["fscore"] = round(cd, 5), round(f, 4)
                chamfers.append(cd)
                fs.append(f)
        if pred_step_dir:
            sp = glob.glob(os.path.join(pred_step_dir, mid + "*.step"))
            if sp:
                v = step_valid(sp[0])
                rec["valid"] = v
                n_valid += int(v)
        rows.append(rec)

    summary = {
        "n_models": n_total,
        "exec_success_rate": round(n_exec / max(n_total, 1), 4),
        "mean_chamfer": round(float(np.mean(chamfers)), 5) if chamfers else None,
        "median_chamfer": round(float(np.median(chamfers)), 5) if chamfers else None,
        "mean_fscore@0.02": round(float(np.mean(fs)), 4) if fs else None,
        "brep_valid_rate": round(n_valid / max(n_total, 1), 4) if pred_step_dir else None,
        "method": "cadrille-rl", "test_set": os.path.basename(gt_dir.rstrip("/")),
    }
    out = os.path.join(pred_mesh_dir, "_RECON_RESULTS.json")
    json.dump({"summary": summary, "per_model": rows[:500]}, open(out, "w"), indent=1)
    print("RECON " + json.dumps(summary))
    print("wrote", out)


if __name__ == "__main__":
    main()
