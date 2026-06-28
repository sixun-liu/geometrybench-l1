#!/bin/bash
# [4090-brep] Full cadrille eval: generate code (300) -> exec to STEP+mesh -> recon metrics.
# Spans two conda envs: cadrille (gen+exec) and brep (eval: pythonocc/scipy/trimesh).
set -e
export HF_HUB_OFFLINE=1
CAD=/root/autodl-tmp/cadrille_work/cadrille
CADPY=/root/miniconda3/envs/cadrille/bin/python
BREPPY=/root/miniconda3/envs/brep/bin/python
EVAL=/root/gbench/cadrille_eval
cd "$CAD"

echo "[$(date +%H:%M:%S)] [1/3] generate CadQuery code (300 models, cadrille-rl)"
rm -rf work_dirs/abc_py && mkdir -p work_dirs/abc_py
$CADPY test.py --split abc_test --mode pc --checkpoint-path maksimko123/cadrille-rl \
       --data-path ./data --py-path work_dirs/abc_py
echo "[$(date +%H:%M:%S)] generated $(ls work_dirs/abc_py/*.py 2>/dev/null | wc -l) code files"

echo "[$(date +%H:%M:%S)] [2/3] exec -> STEP + mesh (subprocess+timeout each)"
rm -rf work_dirs/abc_out
$CADPY "$EVAL/exec_to_brep.py" work_dirs/abc_py work_dirs/abc_out

echo "[$(date +%H:%M:%S)] [3/3] reconstruction metrics vs GT"
$BREPPY "$EVAL/eval_recon.py" /root/autodl-tmp/gbench_work/abc_test_stl \
        work_dirs/abc_out/mesh work_dirs/abc_out/step

echo "[$(date +%H:%M:%S)] [DONE] results: $CAD/work_dirs/abc_out/_RECON_RESULTS.json"
