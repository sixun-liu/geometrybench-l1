#!/usr/bin/env python3
"""
Batched L1 extractor (imports OCCT once, loops over many STEP files).

Per face we now sample N_PER_FACE points by AREA-WEIGHTED sampling over the
face's triangulation (barycentric), not just the triangulation vertices — so
flat faces get a dense cloud too instead of 4 corner nodes. Points are stored
in the model's .npy as (x,y,z,face_id,type_id); labels come from the kernel.

Robustness on the 2 GB / 0.5-core box:
  * resume-skip: a model whose <id>.json already exists is skipped (re-runs are
    cheap and survive an OOM kill on some complex model);
  * size pre-filter: STEP files > MAX_MB are skipped (logged) — the complexity
    cap from the proposal, and the main OOM guard;
  * per-model try/except + gc every 25 models.

Env: N_PER_FACE (default 256), MAX_MB (default 3).
Usage: python extract_batch.py <in_dir_with_.step> <out_dir>
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
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
)

try:
    from OCC.Core.TopoDS import topods
    as_face = topods.Face
except Exception:
    from OCC.Core.TopoDS import topods_Face as as_face
try:
    from OCC.Core.BRepGProp import brepgprop
    surface_props = brepgprop.SurfaceProperties
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties as surface_props
try:
    from OCC.Core.BRepBndLib import brepbndlib
    bbox_add = brepbndlib.Add
except Exception:
    from OCC.Core.BRepBndLib import brepbndlib_Add as bbox_add

TYPE_NAMES = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier", GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset", GeomAbs_OtherSurface: "other",
}
TYPE_ID = {n: i for i, n in enumerate(
    ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "bezier",
     "revolution", "extrusion", "offset", "other", "unknown"])}

N_PER_FACE = int(os.environ.get("N_PER_FACE", "128"))
MAX_MB = float(os.environ.get("MAX_MB", "3"))
MAX_FACES = int(os.environ.get("MAX_FACES", "500"))
RNG = np.random.default_rng(20260615)


def count(shape, kind):
    exp, n = TopExp_Explorer(shape, kind), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def face_params(name, surf):
    try:
        if name == "cylinder":
            return {"radius": round(surf.Cylinder().Radius(), 5)}
        if name == "sphere":
            return {"radius": round(surf.Sphere().Radius(), 5)}
        if name == "cone":
            return {"half_angle": round(surf.Cone().SemiAngle(), 5)}
        if name == "torus":
            t = surf.Torus()
            return {"major_r": round(t.MajorRadius(), 5),
                    "minor_r": round(t.MinorRadius(), 5)}
    except Exception:
        pass
    return {}


def sample_face(face, n):
    """Area-weighted barycentric sampling over the face's triangulation."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None or tri.NbNodes() < 3 or tri.NbTriangles() < 1:
        return None
    trsf = loc.Transformation()
    nb = tri.NbNodes()
    nodes = np.empty((nb, 3))
    for i in range(1, nb + 1):
        p = tri.Node(i).Transformed(trsf)
        nodes[i - 1] = (p.X(), p.Y(), p.Z())
    nt = tri.NbTriangles()
    idx = np.empty((nt, 3), dtype=np.int64)
    for i in range(1, nt + 1):
        a, b, c = tri.Triangle(i).Get()
        idx[i - 1] = (a - 1, b - 1, c - 1)
    A, B, C = nodes[idx[:, 0]], nodes[idx[:, 1]], nodes[idx[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    tot = areas.sum()
    if tot <= 1e-12:
        return nodes if nb >= 3 else None
    sel = RNG.choice(nt, size=n, p=areas / tot)
    r1 = np.sqrt(RNG.random(n))
    r2 = RNG.random(n)
    u, v, w = (1 - r1)[:, None], (r1 * (1 - r2))[:, None], (r1 * r2)[:, None]
    return u * A[sel] + v * B[sel] + w * C[sel]


def process(path, out_dir, mid, difficulty):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("read failed")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError("null shape")
    nfaces = count(shape, TopAbs_FACE)
    if nfaces > MAX_FACES:                              # complexity cap (memory guard)
        raise RuntimeError(f"too_many_faces:{nfaces}")

    bb = Bnd_Box()
    bbox_add(shape, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    BRepMesh_IncrementalMesh(shape, max(diag * 0.01, 1e-3), False, 0.5, True)

    faces, hist, parts = [], {}, []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fid = 0
    while exp.More():
        face = as_face(exp.Current())
        surf = BRepAdaptor_Surface(face, True)
        name = TYPE_NAMES.get(surf.GetType(), "unknown")
        props = GProp_GProps()
        surface_props(face, props)
        pts = sample_face(face, N_PER_FACE)
        npts = 0 if pts is None else len(pts)
        faces.append({"face_id": fid, "surface_type": name, "type_id": TYPE_ID[name],
                      "params": face_params(name, surf), "area": round(props.Mass(), 5),
                      "n_points": npts})
        if npts:                                        # numpy block, not tuple list
            block = np.empty((npts, 5), dtype="float32")
            block[:, :3] = pts
            block[:, 3] = fid
            block[:, 4] = TYPE_ID[name]
            parts.append(block)
        hist[name] = hist.get(name, 0) + 1
        fid += 1
        exp.Next()

    cloud = np.concatenate(parts) if parts else np.empty((0, 5), dtype="float32")
    rec = {"model_id": mid, "source": "ABC", "difficulty": difficulty,
           "n_faces": nfaces, "n_edges": count(shape, TopAbs_EDGE),
           "n_vertices": count(shape, TopAbs_VERTEX), "bbox_diag": round(diag, 5),
           "surface_type_histogram": hist, "n_points_total": int(len(cloud)),
           "faces": faces}
    json.dump(rec, open(os.path.join(out_dir, mid + ".json"), "w"), indent=1)
    if len(cloud):
        np.save(os.path.join(out_dir, mid + ".npy"), cloud)
    return rec


def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.step"), recursive=True))
    print(f"found {len(files)} step files (N_PER_FACE={N_PER_FACE} MAX_MB={MAX_MB})",
          flush=True)

    ok = skip = toobig = resumed = 0
    hist, by_diff, faces_by_diff, skips = {}, {}, {}, []

    def add(rec, diff):
        for k, v in rec["surface_type_histogram"].items():
            hist[k] = hist.get(k, 0) + v
        by_diff[diff] = by_diff.get(diff, 0) + 1
        faces_by_diff[diff] = faces_by_diff.get(diff, 0) + rec["n_faces"]

    for i, f in enumerate(files):
        diff = ("complex" if os.sep + "complex" + os.sep in f else
                "simple" if os.sep + "simple" + os.sep in f else "unknown")
        mid = os.path.splitext(os.path.basename(f))[0]
        jp = os.path.join(out_dir, mid + ".json")
        if os.path.exists(jp):                                  # resume-skip
            try:
                add(json.load(open(jp)), diff)
                ok += 1
                resumed += 1
                continue
            except Exception:
                pass
        if os.path.getsize(f) > MAX_MB * 1e6:                   # size guard
            toobig += 1
            skips.append({"id": mid, "diff": diff, "why": "too_big"})
            continue
        try:
            add(process(f, out_dir, mid, diff), diff)
            ok += 1
        except Exception as e:
            skip += 1
            skips.append({"id": mid, "diff": diff, "why": str(e)[:160]})
        if (i + 1) % 25 == 0:
            print(f"...{i + 1}/{len(files)} ok={ok} skip={skip} toobig={toobig}",
                  flush=True)
            gc.collect()

    summary = {"total": len(files), "ok": ok, "skip": skip, "too_big": toobig,
               "resumed": resumed, "surface_type_histogram": hist,
               "models_by_difficulty": by_diff,
               "avg_faces_by_difficulty": {d: round(faces_by_diff[d] / by_diff[d], 1)
                                           for d in by_diff},
               "skips": skips[:60]}
    json.dump(summary, open(os.path.join(out_dir, "_SUMMARY.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps({k: summary[k] for k in
          ("total", "ok", "skip", "too_big", "surface_type_histogram",
           "models_by_difficulty", "avg_faces_by_difficulty")}))


if __name__ == "__main__":
    main()
