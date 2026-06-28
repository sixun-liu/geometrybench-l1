#!/bin/bash
# [无卡/brep] Re-baseline clean + hard with the SAME bounded RF (2G-safe, comparable).
set -u
PY=/root/miniconda3/envs/brep/bin/python
CODE=/root/gbench/cadrille_eval
W=/root/autodl-tmp/l4_work
for ds in l4_dataset l4_dataset_hard; do
  rm -f "$W/$ds/_BASELINE.json" "$W/$ds/fig_l4_confusion.png"
  echo "[$(date +%H:%M:%S)] baseline $ds"
  $PY "$CODE/baseline_l4.py" "$W/$ds" || echo "FAIL baseline $ds"
  echo "[$(date +%H:%M:%S)] viz $ds"
  $PY "$CODE/viz_l4.py" "$W/$ds" || echo "FAIL viz $ds"
done
echo "[$(date +%H:%M:%S)] REBASE_DONE"
