#!/bin/bash
# [4090] Full CAD-Recode eval on the same 300 ABC test STLs (comparison vs cadrille).
set -e
source /etc/network_turbo 2>/dev/null || true
export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1
CRPY=/root/autodl-tmp/cadrecode_env/bin/python
BREPPY=/root/miniconda3/envs/brep/bin/python
EVAL=/root/gbench/cadrille_eval
OUT=/root/autodl-tmp/cadrecode_work
STL=/root/autodl-tmp/gbench_work/abc_test_stl

echo "[$(date +%H:%M:%S)] [1/3] CAD-Recode generate code (300)"
rm -rf "$OUT/py" && mkdir -p "$OUT/py"
$CRPY "$EVAL/cadrecode_run.py" "$STL" "$OUT/py"
echo "[$(date +%H:%M:%S)] generated $(ls "$OUT/py"/*.py 2>/dev/null | wc -l)"

echo "[$(date +%H:%M:%S)] [2/3] exec -> STEP + mesh"
rm -rf "$OUT/abc_out"
$CRPY "$EVAL/exec_to_brep.py" "$OUT/py" "$OUT/abc_out"

echo "[$(date +%H:%M:%S)] [3/3] recon metrics (raw + ICP, per difficulty)"
$BREPPY "$EVAL/eval_recon2.py" "$STL" "$OUT/abc_out/mesh" "$OUT/abc_out/step" \
        /root/autodl-tmp/gbench_work/abc_data 4096

echo "[$(date +%H:%M:%S)] [DONE] $OUT/abc_out"
