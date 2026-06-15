#!/usr/bin/env python3
"""
GeometryBench L1 — toolchain smoke test.

Builds a synthetic part (box minus a through-cylinder) directly in the CAD
kernel, then runs the exact face-classification logic the ABC extractor will
use. No external data needed: this proves pythonOCC + the surface-typing path
work end to end. Exit 0 = PASS.
"""
import sys

# --- robust imports across pythonocc-core versions -------------------------
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
)

try:                                   # 7.7+: topods.Face ; older: topods_Face
    from OCC.Core.TopoDS import topods
    as_face = topods.Face
except Exception:                      # pragma: no cover
    from OCC.Core.TopoDS import topods_Face as as_face

TYPE_NAMES = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier", GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset", GeomAbs_OtherSurface: "other",
}


def surface_type(face):
    surf = BRepAdaptor_Surface(face, True)
    return TYPE_NAMES.get(surf.GetType(), "unknown"), surf


def face_param(name, surf):
    """Pull the headline parameter for the common analytic surfaces."""
    try:
        if name == "cylinder":
            return {"radius": round(surf.Cylinder().Radius(), 4)}
        if name == "sphere":
            return {"radius": round(surf.Sphere().Radius(), 4)}
        if name == "cone":
            return {"half_angle": round(surf.Cone().SemiAngle(), 4)}
        if name == "torus":
            t = surf.Torus()
            return {"major_r": round(t.MajorRadius(), 4),
                    "minor_r": round(t.MinorRadius(), 4)}
    except Exception:
        pass
    return {}


def count(shape, kind):
    exp, n = TopExp_Explorer(shape, kind), 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def main():
    # box 10x10x10, centered hole r=2 punched all the way through (z axis)
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10, 10, 10).Shape()
    axis = gp_Ax2(gp_Pnt(5, 5, -1), gp_Dir(0, 0, 1))
    cyl = BRepPrimAPI_MakeCylinder(axis, 2.0, 12).Shape()
    solid = BRepAlgoAPI_Cut(box, cyl).Shape()

    hist, faces = {}, []
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    fid = 0
    while exp.More():
        face = as_face(exp.Current())
        name, surf = surface_type(face)
        hist[name] = hist.get(name, 0) + 1
        faces.append((fid, name, face_param(name, surf)))
        fid += 1
        exp.Next()

    nf = count(solid, TopAbs_FACE)
    ne = count(solid, TopAbs_EDGE)
    nv = count(solid, TopAbs_VERTEX)

    print("=== synthetic part: box(10^3) - through-hole(r=2) ===")
    print(f"faces={nf}  edges={ne}  vertices={nv}")
    print(f"surface-type histogram: {hist}")
    for fid, name, p in faces:
        print(f"  face#{fid:<2} {name:<10} {p}")

    # expectation: the side/top/bottom are planes, the hole wall is a cylinder
    ok = hist.get("cylinder", 0) >= 1 and hist.get("plane", 0) >= 4
    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
