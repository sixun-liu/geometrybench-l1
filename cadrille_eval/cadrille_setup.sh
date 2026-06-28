#!/bin/bash
# [4090] Set up cadrille for point-cloud → CadQuery → STEP inference.
# Assumes an AutoDL-style box with conda + a CUDA 12.x driver. China mirrors used.
# We patch test.py to SDPA so flash-attn is NOT required (avoids a slow CUDA build).
set -e
source /etc/network_turbo 2>/dev/null || true   # AutoDL 学术加速：github / pytorch.org / huggingface
export HF_ENDPOINT=https://hf-mirror.com
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"
WORK=/root/autodl-tmp/cadrille_work
mkdir -p "$WORK" && cd "$WORK"

echo "== clone repos (cadrille + cad-recode fallback) =="
[ -d cadrille ]   || git clone https://github.com/col14m/cadrille.git
[ -d cad-recode ] || git clone https://github.com/filaPro/cad-recode.git

echo "== conda env =="
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda create -y -n cadrille python=3.10
conda activate cadrille

echo "== torch 2.5.1 / cu124 =="
$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || \
$PIP torch==2.5.1 torchvision==0.20.1

echo "== core inference deps (pinned per cadrille Dockerfile) =="
$PIP transformers==4.50.3 tokenizers==0.21.0 accelerate==0.34.2 \
     huggingface-hub==0.27.0 safetensors==0.4.5 qwen-vl-utils==0.0.10 einops==0.8.0 \
     numpy==2.2.0 scipy==1.14.1 trimesh==4.5.3 scikit-image==0.25.0 open3d tqdm

echo "== pytorch3d (FPS sampling; needed by stock loader) =="
# try prebuilt first; fall back to source (needs nvcc from a -devel image)
$PIP "git+https://github.com/facebookresearch/pytorch3d@06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f" \
  --no-build-isolation || echo "!! pytorch3d build failed — see PLAN.md, may need prebuilt wheel"

echo "== CadQuery (output → STEP/B-Rep) =="
$PIP cadquery-ocp==7.7.2 manifold3d==3.0.0 ezdxf==1.3.5 casadi==3.6.7 \
     nlopt==2.9.0 multimethod typish path \
     "git+https://github.com/CadQuery/cadquery.git@e99a15df3cf6a88b69101c405326305b5db8ed94"

echo "== patch test.py to SDPA (skip flash-attn) =="
sed -i "s/attn_implementation='flash_attention_2'/attn_implementation='sdpa'/g" cadrille/test.py || true

echo "== pre-download weights via hf-mirror (RL = SOTA) =="
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
for repo in ["maksimko123/cadrille-rl", "Qwen/Qwen2-VL-2B-Instruct"]:
    print("downloading", repo)
    snapshot_download(repo)
print("weights ready")
PY

echo "== sanity: imports =="
python -c "import torch,transformers,cadquery,trimesh,open3d; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "DONE — cadrille env ready at $WORK (conda env 'cadrille')"
