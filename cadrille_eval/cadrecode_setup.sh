#!/bin/bash
# [4090] Separate env for CAD-Recode (2nd method, comparison). It needs
# transformers==4.47.1 (conflicts with cadrille's 4.50.3) -> own conda env.
# Repo already cloned at /root/autodl-tmp/cadrille_work/cad-recode by cadrille_setup.sh.
set -e
source /etc/network_turbo 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache   # 权重放大盘（/ 系统盘只有 30G）
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"
source /root/miniconda3/etc/profile.d/conda.sh
conda create -y -p /root/autodl-tmp/cadrecode_env python=3.10   # 环境也放大盘
conda activate /root/autodl-tmp/cadrecode_env

$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || $PIP torch==2.5.1
$PIP transformers==4.47.1 accelerate safetensors numpy trimesh tqdm einops
$PIP cadquery-ocp==7.7.2 manifold3d ezdxf casadi nlopt multimethod typish path \
     "git+https://github.com/CadQuery/cadquery.git@e99a15df3cf6a88b69101c405326305b5db8ed94"

python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
for r in ["filapro/cad-recode-v1.5", "Qwen/Qwen2-1.5B"]:
    print("downloading", r); snapshot_download(r)
print("weights ready")
PY

python -c "import torch,transformers,cadquery; print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
echo "DONE — cadrecode env ready"
