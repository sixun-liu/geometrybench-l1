#!/usr/bin/env python3
"""
Build the cadrille test set: ABC STEP -> mesh tessellation -> normalize to the
UNIT CUBE [0,1]^3 (center 0.5, longest edge 1) -> STL.

cadrille's stock loader reads .stl from data/<split>/, samples 8192 surface
points, FPS-downsamples to 256, then maps to [-1,1] via (x-0.5)*2 — so it
ASSUMES exactly this unit-cube normalization (see runbook). The same normalized
STL is ALSO the ground-truth surface for Chamfer/F-score in eval_recon.py, so
prediction and GT live in the same frame.

Runs in the `brep` conda env (pythonocc) + trimesh:
    /root/miniconda3/envs/brep/bin/pip install trimesh   # if missing
Usage: python prep_testset.py <in_dir_with_.step> <out_stl_dir> [max_models] [max_mb]
"""
import glob
import os
import sys
import tempfile

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.Bnd import Bnd_Box
try:
    from OCC.Core.BRepBndLib import brepbndlib
    bbox_add = brepbndlib.Add
except Exception:
    from OCC.Core.BRepBndLib import brepbndlib_Add as bbox_add

import trimesh


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("read failed")
    r.TransferRoots()
    s = r.OneShape()
    if s.IsNull():
        raise RuntimeError("null shape")
    return s


def main():
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    max_models = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    max_mb = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.step"), recursive=True))
    ok = skip = 0
    for f in files:
        if max_models and ok >= max_models:
            break
        mid = os.path.splitext(os.path.basename(f))[0]
        out_stl = os.path.join(out_dir, mid + ".stl")
        if os.path.exists(out_stl):
            ok += 1
            continue
        if os.path.getsize(f) > max_mb * 1e6:
            skip += 1
            continue
        try:
            shape = read_step(f)
            bb = Bnd_Box()
            bbox_add(shape, bb)
            xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
            diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
            BRepMesh_IncrementalMesh(shape, max(diag * 0.005, 1e-3), False, 0.5, True)
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_path = tmp.name
            if not StlAPI_Writer().Write(shape, tmp_path):
                raise RuntimeError("stl write failed")
            m = trimesh.load(tmp_path, force="mesh")
            os.unlink(tmp_path)
            if m.vertices.shape[0] < 4 or m.extents.max() <= 0:
                raise RuntimeError("degenerate mesh")
            m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)  # center -> 0
            m.apply_scale(1.0 / m.extents.max())                     # longest -> 1
            m.apply_translation([0.5, 0.5, 0.5])                     # -> unit cube
            m.export(out_stl)
            ok += 1
            if ok % 25 == 0:
                print(f"...{ok} stl written (skip={skip})", flush=True)
        except Exception as e:
            skip += 1
            print(f"SKIP {mid}: {str(e)[:100]}", file=sys.stderr)
    print(f"DONE {ok} normalized STL in {out_dir} (skipped {skip})")


if __name__ == "__main__":
    main()
