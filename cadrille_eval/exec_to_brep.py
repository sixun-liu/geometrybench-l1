#!/usr/bin/env python3
"""
[cadrille env] Execute cadrille's generated CadQuery code -> STEP (B-Rep) + STL.
Each file is exec'd in a SUBPROCESS with a timeout, because generated code can
hang / leak (CadQuery issue #1665). Result solid is the variable `r`.

Usage: python exec_to_brep.py <py_dir> <out_dir>
  writes <out_dir>/step/<id>.step  and  <out_dir>/mesh/<id>.stl
"""
import glob
import json
import os
import subprocess
import sys

PY = sys.executable  # the cadrille env python (has cadquery)


def main():
    py_dir, out_dir = sys.argv[1], sys.argv[2]
    step_dir = os.path.join(out_dir, "step")
    mesh_dir = os.path.join(out_dir, "mesh")
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(py_dir, "*.py")))
    ok = fail = 0
    fails = []
    for p in files:
        mid = os.path.basename(p)[:-3].replace("+0", "")
        step = os.path.join(step_dir, mid + ".step")
        mesh = os.path.join(mesh_dir, mid + ".stl")
        snippet = (
            "import cadquery as cq\n"
            "g={}\n"
            f"exec(open({p!r}).read(), g)\n"
            "r=g['r']; sol = r.val() if hasattr(r,'val') else r\n"
            f"cq.exporters.export(sol, {step!r})\n"
            f"cq.exporters.export(sol, {mesh!r})\n"
        )
        try:
            r = subprocess.run([PY, "-c", snippet], capture_output=True,
                               text=True, timeout=15)
            if r.returncode == 0 and os.path.exists(step) and os.path.getsize(step) > 0:
                ok += 1
            else:
                fail += 1
                fails.append({"id": mid, "why": (r.stderr or "no output").strip()[-120:]})
        except subprocess.TimeoutExpired:
            fail += 1
            fails.append({"id": mid, "why": "exec timeout"})
        except Exception as e:
            fail += 1
            fails.append({"id": mid, "why": str(e)[:120]})
        if (ok + fail) % 50 == 0:
            print(f"...{ok + fail}/{len(files)} ok={ok} fail={fail}", flush=True)

    summary = {"total": len(files), "exec_ok": ok, "exec_fail": fail,
               "exec_success_rate": round(ok / max(len(files), 1), 4), "fails": fails[:40]}
    json.dump(summary, open(os.path.join(out_dir, "_EXEC_SUMMARY.json"), "w"), indent=1)
    print("EXEC " + json.dumps({k: summary[k] for k in
          ("total", "exec_ok", "exec_fail", "exec_success_rate")}))


if __name__ == "__main__":
    main()
