#!/bin/bash
# [无卡/brep] Hard (scanned-like) variant of the scaled 10-class set + re-baseline.
#   occlude(random view) -> resample -> gaussian noise -> renormalize, then
#   by-model baseline on the harder input (same labels/groups).
set -u
PY=/root/miniconda3/envs/brep/bin/python
CODE=/root/gbench/cadrille_eval
W=/root/autodl-tmp/l4_work
rm -f "$W/l4_dataset_hard/_BASELINE.json"

echo "[$(date +%H:%M:%S)] [1/3] make hard variant (noise 0.03 + occlude 0.3)"
$PY "$CODE/make_hard.py" "$W/l4_dataset" "$W/l4_dataset_hard" 0.03 0.3

echo "[$(date +%H:%M:%S)] [2/3] baseline on hard (by-model)"
$PY "$CODE/baseline_l4.py" "$W/l4_dataset_hard"

echo "[$(date +%H:%M:%S)] [3/3] figures (hard)"
$PY "$CODE/viz_l4.py" "$W/l4_dataset_hard"
echo "[$(date +%H:%M:%S)] [DONEHARD]"
