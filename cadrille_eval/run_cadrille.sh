#!/bin/bash
# [4090] Run cadrille point-cloud inference on the prepared ABC test STLs,
# then export each prediction to STEP/mesh. Run after cadrille_setup.sh.
# Test STLs must be at:  $CAD/data/abc_test/*.stl   (from prep_testset.py)
set -e
export HF_ENDPOINT=https://hf-mirror.com
CAD=/root/autodl-tmp/cadrille_work/cadrille
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cadrille
cd "$CAD"

PYOUT=./work_dirs/abc_py
rm -rf "$PYOUT" && mkdir -p "$PYOUT"

echo "== [1/2] generate CadQuery code from point clouds (RL checkpoint) =="
# loader reads data/abc_test/*.stl -> sample 8192 -> FPS 256 -> generate code
python test.py \
  --split abc_test \
  --mode pc \
  --checkpoint-path maksimko123/cadrille-rl \
  --data-path ./data \
  --py-path "$PYOUT"
echo "generated $(ls "$PYOUT"/*.py 2>/dev/null | wc -l) code files"

echo "== [2/2] exec code -> mesh + STEP (B-Rep) =="
# enable the STEP export line in evaluate.py (it's commented by default)
sed -i "s/^\s*#\s*cq.exporters.export(compound, brep_path)/    cq.exporters.export(compound, brep_path)/" evaluate.py || true
# NOTE: confirm evaluate.py's exact flags on the box; expected outputs land in
# work_dirs/tmp_mesh/*.stl and work_dirs/tmp_brep/*.step
python evaluate.py --pred-py-path "$PYOUT" || \
  echo "!! check evaluate.py args (see its argparse) — finalize live"
echo "DONE — predictions (code + mesh + STEP) under $CAD/work_dirs/"
