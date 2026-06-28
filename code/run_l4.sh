#!/bin/bash
# [无卡/brep env] Scale L1 to >=10k per class + add curves.
#   synthetic generation -> rare surfaces; ABC + synthetic -> curves; reuse
#   existing L1 surface extraction for common surfaces. All small models (2G-safe).
# Arg $1 = number of synthetic models (default 5000). All steps resumable.
set -u
N=${1:-5000}
BREPPY=/root/miniconda3/envs/brep/bin/python
EVAL=/root/gbench/cadrille_eval
W=/root/autodl-tmp/l4_work
mkdir -p "$W"
export N_PER_FACE=128 MAX_FACES=400 N_PER_EDGE=128 MAX_EDGES=600

echo "[$(date +%H:%M:%S)] [1/6] generate $N synthetic models"
$BREPPY "$EVAL/gen_synthetic.py" "$W/syn_step" "$N"
echo "[$(date +%H:%M:%S)] syn step on disk: $(ls "$W/syn_step"/*.step 2>/dev/null | wc -l)"

echo "[$(date +%H:%M:%S)] [2/6] extract surfaces (synthetic)"
$BREPPY /root/gbench/extract_batch.py "$W/syn_step" "$W/syn_surf"

echo "[$(date +%H:%M:%S)] [3/6] extract curves (synthetic)"
$BREPPY "$EVAL/extract_curves.py" "$W/syn_step" "$W/syn_curve"

echo "[$(date +%H:%M:%S)] [4/6] extract curves (ABC)"
$BREPPY "$EVAL/extract_curves.py" /root/autodl-tmp/gbench_work/abc_data "$W/abc_curve"

echo "[$(date +%H:%M:%S)] [5/6] assemble (10 classes, cap 12k)"
$BREPPY "$EVAL/assemble_l4.py" "$W/l4_dataset" 128 12000 \
   /root/autodl-tmp/gbench_work/extract_out "$W/syn_surf" -- "$W/syn_curve" "$W/abc_curve"

echo "[$(date +%H:%M:%S)] [6/6] baseline"
$BREPPY "$EVAL/baseline_l4.py" "$W/l4_dataset"

echo "[$(date +%H:%M:%S)] [DONE] dataset at $W/l4_dataset"
