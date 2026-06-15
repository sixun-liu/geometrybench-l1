#!/usr/bin/env python3
"""
GeometryBench L1 extractor — process ONE STEP file into recognition-task data.

For each face: surface type + headline parameters (read straight from the
kernel = ground truth), a point sample taken on the trimmed face, and its area.
Writes <out>/<id>.json (per-face records + V/E/F counts) and, if numpy is
present, <out>/<id>.npy holding an (N,5) cloud of [x,y,z,face_id,type_id].

Designed to be called once per model by a driver so OCCT memory is released
between models (key on the 2 GB box). Exit 0 = ok, non-zero = skip-and-log.

Usage: python extract_l1.py <model.step> <out_dir> [model_id]
"""
import json
import os
import sys

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
except Exception:                                    # pragma: no cover
    from OCC.Core.TopoDS import topods_Face as as_face
try:                                                 # API drift 7.6 -> 7.7+
    from OCC.Core.BRepGProp import brepgprop
    surface_props = brepgprop.SurfaceProperties
except Exception:                                    # pragma: no cover
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties as surface_props
try:
    from OCC.Core.BRepBndLib import brepbndlib
    bbox_add = brepbndlib.Add
except Exception:                                    # pragma: no cover
    from OCC.Core.BRepBndLib import brepbndlib_Add as bbox_add

try:
    import numpy as np
except Exception:
    np = None

TYPE_NAMES = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier", GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset", GeomAbs_OtherSurface: "other",
}
# stable integer label per type (the recognition target)
TYPE_ID = {n: i for i, n in enumerate(
    ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "bezier",
     "revolution", "extrusion", "offset", "other", "unknown"])}


def read_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError("null shape")
    return shape


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
            return {"half_angle": round(surf.Cone().SemiAngle(), 5),
                    "ref_radius": round(surf.Cone().RefRadius(), 5)}
        if name == "torus":
            t = surf.Torus()
            return {"major_r": round(t.MajorRadius(), 5),
                    "minor_r": round(t.MinorRadius(), 5)}
    except Exception:
        pass
    return {}


def face_points(face):
    """Triangulation nodes lie on the trimmed face -> use them as the sample."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None:
        return []
    trsf = loc.Transformation()
    pts = []
    try:
        nb = tri.NbNodes()
        get = tri.Node                                # 7.7+: gp_Pnt = tri.Node(i)
    except Exception:                                 # pragma: no cover
        nodes = tri.Nodes()
        nb = nodes.Length()
        get = nodes.Value
    for i in range(1, nb + 1):
        p = get(i).Transformed(trsf)
        pts.append((p.X(), p.Y(), p.Z()))
    return pts


def main():
    if len(sys.argv) < 3:
        print("usage: extract_l1.py <model.step> <out_dir> [id]", file=sys.stderr)
        return 2
    path, out_dir = sys.argv[1], sys.argv[2]
    mid = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)

    shape = read_step(path)

    # deflection relative to bbox so density is consistent across model scales
    bb = Bnd_Box()
    bbox_add(shape, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    defl = max(diag * 0.01, 1e-3)
    BRepMesh_IncrementalMesh(shape, defl, False, 0.5, True)

    faces, hist, cloud = [], {}, []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fid = 0
    while exp.More():
        face = as_face(exp.Current())
        surf = BRepAdaptor_Surface(face, True)
        name = TYPE_NAMES.get(surf.GetType(), "unknown")
        props = GProp_GProps()
        surface_props(face, props)
        pts = face_points(face)
        faces.append({
            "face_id": fid, "surface_type": name, "type_id": TYPE_ID[name],
            "params": face_params(name, surf),
            "area": round(props.Mass(), 5), "n_points": len(pts),
        })
        hist[name] = hist.get(name, 0) + 1
        for (x, y, z) in pts:
            cloud.append((x, y, z, fid, TYPE_ID[name]))
        fid += 1
        exp.Next()

    rec = {
        "model_id": mid, "source": "ABC",
        "n_faces": count(shape, TopAbs_FACE),
        "n_edges": count(shape, TopAbs_EDGE),
        "n_vertices": count(shape, TopAbs_VERTEX),
        "bbox_diag": round(diag, 5),
        "surface_type_histogram": hist,
        "n_points_total": len(cloud),
        "faces": faces,
    }
    with open(os.path.join(out_dir, mid + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    if np is not None and cloud:
        np.save(os.path.join(out_dir, mid + ".npy"),
                np.asarray(cloud, dtype="float32"))

    print(f"OK {mid}: faces={rec['n_faces']} types={hist} pts={len(cloud)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                            # skip-and-log, never crash driver
        print(f"FAIL {sys.argv[1] if len(sys.argv) > 1 else '?'}: {e}", file=sys.stderr)
        sys.exit(1)
