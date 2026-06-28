#!/usr/bin/env python3
"""
[brep env] L1 CURVE recognition extractor — the curve counterpart of
extract_batch.py. For each STEP model, walk its edges, read the curve TYPE from
the kernel (line/circle/ellipse/bspline/...), and sample points ALONG the edge.
Writes <id>.cjson + <id>.cnpy (N,5: x,y,z,edge_id,type_id) per model + _CSUMMARY.

Usage: python extract_curves.py <in_dir_with_.step> <out_dir>
"""
import gc
import glob
import json
import os
import sys

import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_OtherCurve)

try:
    from OCC.Core.TopoDS import topods
    as_edge = topods.Edge
except Exception:
    from OCC.Core.TopoDS import topods_Edge as as_edge

TYPE_NAMES = {GeomAbs_Line: "line", GeomAbs_Circle: "circle", GeomAbs_Ellipse: "ellipse",
              GeomAbs_Hyperbola: "hyperbola", GeomAbs_Parabola: "parabola",
              GeomAbs_BezierCurve: "bezier", GeomAbs_BSplineCurve: "bspline",
              GeomAbs_OtherCurve: "other"}
TYPE_ID = {n: i for i, n in enumerate(
    ["line", "circle", "ellipse", "hyperbola", "parabola", "bezier", "bspline",
     "other", "unknown"])}
N_PER_EDGE = int(os.environ.get("N_PER_EDGE", "128"))
MAX_EDGES = int(os.environ.get("MAX_EDGES", "800"))


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("read failed")
    r.TransferRoots()
    s = r.OneShape()
    if s.IsNull():
        raise RuntimeError("null")
    return s


def sample_edge(c, n):
    a, b = c.FirstParameter(), c.LastParameter()
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return None
    ts = np.linspace(a, b, n)
    pts = np.empty((n, 3), dtype="float32")
    for i, t in enumerate(ts):
        p = c.Value(float(t))
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


def process(path, out_dir, mid):
    shape = read_step(path)
    edges, hist, parts = [], {}, []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    eid = 0
    while exp.More():
        if eid > MAX_EDGES:
            break
        try:
            c = BRepAdaptor_Curve(as_edge(exp.Current()))
            name = TYPE_NAMES.get(c.GetType(), "unknown")
            pts = sample_edge(c, N_PER_EDGE)
            if pts is not None and len(pts):
                blk = np.empty((len(pts), 5), dtype="float32")
                blk[:, :3] = pts
                blk[:, 3] = eid
                blk[:, 4] = TYPE_ID[name]
                parts.append(blk)
                edges.append({"edge_id": eid, "curve_type": name,
                              "type_id": TYPE_ID[name], "n_points": int(len(pts))})
                hist[name] = hist.get(name, 0) + 1
        except Exception:
            pass
        eid += 1
        exp.Next()
    cloud = np.concatenate(parts) if parts else np.empty((0, 5), dtype="float32")
    rec = {"model_id": mid, "n_edges": len(edges),
           "curve_type_histogram": hist, "edges": edges}
    json.dump(rec, open(os.path.join(out_dir, mid + ".cjson"), "w"))
    if len(cloud):
        np.save(os.path.join(out_dir, mid + ".cnpy.npy"), cloud)
    return rec


def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.step"), recursive=True))
    print(f"found {len(files)} step files (N_PER_EDGE={N_PER_EDGE})", flush=True)
    ok = skip = 0
    hist = {}
    for i, f in enumerate(files):
        mid = os.path.splitext(os.path.basename(f))[0]
        jp = os.path.join(out_dir, mid + ".cjson")
        if os.path.exists(jp):
            ok += 1
            continue
        try:
            rec = process(f, out_dir, mid)
            ok += 1
            for k, v in rec["curve_type_histogram"].items():
                hist[k] = hist.get(k, 0) + v
        except Exception as e:
            skip += 1
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{len(files)} ok={ok} skip={skip}", flush=True)
            gc.collect()
    json.dump({"total": len(files), "ok": ok, "skip": skip, "curve_hist": hist},
              open(os.path.join(out_dir, "_CSUMMARY.json"), "w"), indent=1)
    print("CSUMMARY " + json.dumps({"ok": ok, "skip": skip, "curve_hist": hist}))


if __name__ == "__main__":
    main()
