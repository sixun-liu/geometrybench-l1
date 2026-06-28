#!/bin/bash
# [无卡/brep] Round 2: close the freeform_surf gap + by-model split.
#   reuses existing syn_surf/syn_curve/abc_curve; only adds freeform (bspline)
#   surfaces, then re-assembles (with groups.npy) + re-baselines (by-model).
# Arg $1 = number of freeform models (default 1500). Resumable.
set -u
N=${1:-1500}
BREPPY=/root/miniconda3/envs/brep/bin/python
EVAL=/root/gbench/cadrille_eval
W=/root/autodl-tmp/l4_work
export N_PER_FACE=128 MAX_FACES=400
rm -f "$W/l4_dataset/_BASELINE.json"   # so a watcher waits for the NEW baseline

echo "[$(date +%H:%M:%S)] [1/4] generate $N freeform (bspline) models"
$BREPPY "$EVAL/gen_freeform.py" "$W/frf_step" "$N"
echo "[$(date +%H:%M:%S)] frf step: $(find "$W/frf_step" -name '*.step' 2>/dev/null | wc -l)"

echo "[$(date +%H:%M:%S)] [2/4] extract surfaces (freeform)"
$BREPPY /root/gbench/extract_batch.py "$W/frf_step" "$W/frf_surf"

echo "[$(date +%H:%M:%S)] [3/4] re-assemble (extract_out + syn_surf + frf_surf | syn_curve + abc_curve)"
$BREPPY "$EVAL/assemble_l4.py" "$W/l4_dataset" 128 12000 \
   /root/autodl-tmp/gbench_work/extract_out "$W/syn_surf" "$W/frf_surf" -- "$W/syn_curve" "$W/abc_curve"

echo "[$(date +%H:%M:%S)] [4/4] re-baseline (by-model split)"
$BREPPY "$EVAL/baseline_l4.py" "$W/l4_dataset"
echo "[$(date +%H:%M:%S)] [DONE2] dataset at $W/l4_dataset"
