#!/usr/bin/env python3
"""
[brep env] Enhanced recon eval: ① per-difficulty (simple vs complex) breakdown
and ② ICP-aligned Chamfer/F-score (fair metric — cadrille is not pose-invariant,
so we report both raw and best-alignment numbers).

Usage: python eval_recon2.py <gt_stl_dir> <pred_mesh_dir> <pred_step_dir> <abc_data_dir> [n]
"""
import glob
import json
import os
import sys

import numpy as np
import trimesh
from trimesh.registration import icp
from scipy.spatial import cKDTree


def norm_unit(p):
    c = (p.min(0) + p.max(0)) / 2
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
    dp, _ = cKDTree(gt).query(pred)
    dg, _ = cKDTree(pred).query(gt)
    cd = float(dp.mean() + dg.mean())
    prec, rec = float((dp < tau).mean()), float((dg < tau).mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return cd, f


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
        return (not s.IsNull()) and bool(BRepCheck_Analyzer(s).IsValid())
    except Exception:
        return False


def diff_map(abc_dir):
    m = {}
    for d in ("simple", "complex"):
        for f in glob.glob(os.path.join(abc_dir, "step_files", d, "*.step")):
            m[os.path.splitext(os.path.basename(f))[0]] = d
    return m


def main():
    gt_dir, pmesh, pstep, abc_dir = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
    dm = diff_map(abc_dir)
    buckets = {"all": [], "simple": [], "complex": []}
    valid = {"all": [0, 0], "simple": [0, 0], "complex": [0, 0]}  # [n_valid, n]

    for g in sorted(glob.glob(os.path.join(gt_dir, "*.stl"))):
        mid = os.path.splitext(os.path.basename(g))[0]
        d = dm.get(mid, "simple")
        cand = glob.glob(os.path.join(pmesh, mid + "*"))
        cand = [c for c in cand if c.lower().endswith((".stl", ".ply", ".obj"))]
        if not cand:
            continue
        pp, gp = sample(cand[0], n), sample(g, n)
        if pp is None or gp is None:
            continue
        cd_raw, f_raw = chamfer_f(pp, gp)
        try:
            _, aligned, _ = icp(pp, gp, max_iterations=30)
            cd_icp, f_icp = chamfer_f(np.asarray(aligned), gp)
        except Exception:
            cd_icp, f_icp = cd_raw, f_raw
        row = {"cd_raw": cd_raw, "f_raw": f_raw, "cd_icp": cd_icp, "f_icp": f_icp}
        buckets["all"].append(row)
        buckets[d].append(row)
        sp = glob.glob(os.path.join(pstep, mid + "*.step"))
        v = step_valid(sp[0]) if sp else False
        for k in ("all", d):
            valid[k][0] += int(v)
            valid[k][1] += 1

    def agg(rows, key):
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None

    out = {}
    for k in ("all", "simple", "complex"):
        rows = buckets[k]
        out[k] = {
            "n": len(rows),
            "median_cd_raw": round(float(np.median([r["cd_raw"] for r in rows])), 4) if rows else None,
            "median_cd_icp": round(float(np.median([r["cd_icp"] for r in rows])), 4) if rows else None,
            "f@0.02_raw": agg(rows, "f_raw"),
            "f@0.02_icp": agg(rows, "f_icp"),
            "valid_rate": round(valid[k][0] / max(valid[k][1], 1), 4),
        }
    json.dump(out, open(os.path.join(pmesh, "_RECON2_RESULTS.json"), "w"), indent=1)
    print("RECON2 " + json.dumps(out))


if __name__ == "__main__":
    main()
