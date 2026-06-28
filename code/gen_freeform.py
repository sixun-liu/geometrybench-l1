#!/usr/bin/env python3
"""
[brep env] Synthetic FREEFORM (BSpline) surface generator — fills the one L1 class
synthetic boolean-of-primitives can't make. Each model = 1-3 lofted solids
(BRepOffsetAPI_ThruSections through scaled/rotated n-gon sections, ruled=False ->
smooth BSpline side faces) optionally fused onto a small box. Kernel labels the
side faces as BSplineSurface -> extract_batch type_id 5 -> freeform_surf. 2G-safe.

Usage: python gen_freeform.py <out_step_dir> <n_models> [seed]
"""
import math
import os
import random
import sys

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepCheck import BRepCheck_Analyzer


def loft_solid():
    """A solid whose lateral faces are BSpline (smooth loft through n-gon sections)."""
    gen = BRepOffsetAPI_ThruSections(True, False, 1e-6)  # solid=True, ruled=False
    cx, cy = random.uniform(-3, 3), random.uniform(-3, 3)
    n = random.choice([4, 5, 6, 8])
    base_ang = random.uniform(0, math.pi)
    nz = random.randint(2, 4)
    for k in range(nz):
        z = k * random.uniform(2.0, 4.0)
        r = random.uniform(1.5, 4.0) * (1.0 + 0.35 * random.uniform(-1, 1))
        twist = base_ang + 0.4 * k * random.uniform(-1, 1)
        poly = BRepBuilderAPI_MakePolygon()
        for i in range(n):
            ang = 2 * math.pi * i / n + twist
            jx = random.uniform(-0.3, 0.3)
            poly.Add(gp_Pnt(cx + r * math.cos(ang) + jx,
                            cy + r * math.sin(ang), z))
        poly.Close()
        if poly.IsDone():
            gen.AddWire(poly.Wire())
    gen.Build()
    s = gen.Shape()
    if s.IsNull():
        raise RuntimeError("loft null")
    return s


def gen_one():
    shape = loft_solid()
    for _ in range(random.randint(0, 2)):
        try:
            other = loft_solid()
            op = BRepAlgoAPI_Cut if random.random() < 0.35 else BRepAlgoAPI_Fuse
            res = op(shape, other).Shape()
            if not res.IsNull():
                shape = res
        except Exception:
            pass
    if random.random() < 0.4:  # sometimes anchor on a box for varied trimming
        try:
            box = BRepPrimAPI_MakeBox(gp_Pnt(-3, -3, -1), 6, 6, 3).Shape()
            res = BRepAlgoAPI_Fuse(shape, box).Shape()
            if not res.IsNull():
                shape = res
        except Exception:
            pass
    return shape


def main():
    out, n = sys.argv[1], int(sys.argv[2])
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 54321)
    os.makedirs(out, exist_ok=True)
    ok = 0
    for i in range(n):
        try:
            s = gen_one()
            if s.IsNull() or not BRepCheck_Analyzer(s).IsValid():
                continue
            w = STEPControl_Writer()
            w.Transfer(s, STEPControl_AsIs)
            if w.Write(os.path.join(out, f"frf_{i:06d}.step")) == 1:
                ok += 1
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{n} written={ok}", flush=True)
    print(f"DONE {ok}/{n} freeform STEP -> {out}")


if __name__ == "__main__":
    main()
