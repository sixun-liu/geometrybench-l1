#!/usr/bin/env python3
"""
[brep env] Synthetic CAD generator for L1 RARE classes (sphere/cone/torus + the
analytic curves they bring). Each model = a base block with a few random
primitives fused/cut in, so faces include real (often trimmed/partial) sphere,
cone, torus, cylinder patches — kernel-labeled for free. Small by design → 2G-safe.

Usage: python gen_synthetic.py <out_step_dir> <n_models> [seed]
"""
import os
import random
import sys

from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepCheck import BRepCheck_Analyzer


def ax2():
    return gp_Ax2(gp_Pnt(random.uniform(-3, 3), random.uniform(-3, 3),
                         random.uniform(-3, 3)),
                  gp_Dir(*random.choice([(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                         (1, 1, 1)])))


def rand_prim():
    t = random.choice(["sphere", "cone", "torus", "cylinder", "sphere", "cone", "torus"])
    r = random.uniform(1.0, 3.5)
    if t == "sphere":
        return BRepPrimAPI_MakeSphere(gp_Pnt(random.uniform(-3, 3), random.uniform(-3, 3),
                                             random.uniform(-3, 3)), r).Shape()
    if t == "cylinder":
        return BRepPrimAPI_MakeCylinder(ax2(), r, random.uniform(2, 6)).Shape()
    if t == "cone":
        return BRepPrimAPI_MakeCone(ax2(), r, random.uniform(0, r * 0.6),
                                    random.uniform(2, 6)).Shape()
    return BRepPrimAPI_MakeTorus(ax2(), r, random.uniform(0.3, r * 0.7)).Shape()


def gen_one():
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-4, -4, -4), 8, 8, 8).Shape()
    for _ in range(random.randint(3, 7)):
        try:
            p = rand_prim()
            op = BRepAlgoAPI_Cut if random.random() < 0.5 else BRepAlgoAPI_Fuse
            res = op(shape, p).Shape()
            if not res.IsNull():
                shape = res
        except Exception:
            pass
    return shape


def main():
    out, n = sys.argv[1], int(sys.argv[2])
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 12345)
    os.makedirs(out, exist_ok=True)
    ok = 0
    for i in range(n):
        try:
            s = gen_one()
            if s.IsNull() or not BRepCheck_Analyzer(s).IsValid():
                continue
            w = STEPControl_Writer()
            w.Transfer(s, STEPControl_AsIs)
            if w.Write(os.path.join(out, f"syn_{i:06d}.step")) == 1:
                ok += 1
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"...{i + 1}/{n} written={ok}", flush=True)
    print(f"DONE {ok}/{n} synthetic STEP -> {out}")


if __name__ == "__main__":
    main()
