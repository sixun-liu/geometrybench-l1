#!/usr/bin/env python3
"""
[cadrecode env] Run CAD-Recode (filapro/cad-recode-v1.5) on the test STLs ->
CadQuery code per model. CAD-Recode has no CLI (demo-notebook only), so we pull
its model class straight out of demo.ipynb (cell 2), strip the pytorch3d import
(we do farthest-point sampling in numpy instead), and run the demo's inference
recipe (cell 7/11): center-0 / longest-extent-2 normalization, 256-point FPS,
256 pad tokens + <|im_start|>, generate, slice the code.

Usage: python cadrecode_run.py <stl_dir> <py_out_dir>
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
from transformers import AutoTokenizer

DEMO = "/root/autodl-tmp/cadrille_work/cad-recode/demo.ipynb"


def load_model_class():
    nb = json.load(open(DEMO))
    # cell with the FourierPointEncoder + CADRecode definitions
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "class CADRecode" in "".join(c["source"]))
    bad = ("pytorch3d", "open3d", "skimage", "matplotlib", "cadquery",
           "cKDTree", "import trimesh")          # viz/sampling deps the class doesn't need
    src = "\n".join(l for l in src.splitlines() if not any(b in l for b in bad))
    g = {}
    exec(src, g)
    return g["CADRecode"]


def fps_numpy(points, k):
    n = len(points)
    sel = [0]
    dist = np.full(n, np.inf)
    for _ in range(k - 1):
        dist = np.minimum(dist, np.linalg.norm(points - points[sel[-1]], axis=1))
        sel.append(int(dist.argmax()))
    return points[sel]


def main():
    stl_dir, py_out = sys.argv[1], sys.argv[2]
    os.makedirs(py_out, exist_ok=True)
    device = "cuda"
    CADRecode = load_model_class()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", pad_token="<|im_end|>",
                                        padding_side="left")
    model = CADRecode.from_pretrained("filapro/cad-recode-v1.5", torch_dtype="auto",
                                      attn_implementation="sdpa").eval().to(device)
    im_start = tok("<|im_start|>")["input_ids"][0]

    files = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))
    ok = 0
    for i, stl in enumerate(files):
        mid = os.path.splitext(os.path.basename(stl))[0]
        try:
            m = trimesh.load(stl, force="mesh")
            m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
            m.apply_scale(2.0 / max(m.extents))            # center 0, longest 2
            np.random.seed(0)
            v, _ = trimesh.sample.sample_surface(m, 8192)
            pc = fps_numpy(np.asarray(v), 256).astype(np.float32)
            ids = [tok.pad_token_id] * len(pc) + [im_start]
            att = [-1] * len(pc) + [1]
            with torch.no_grad():
                out = model.generate(
                    input_ids=torch.tensor(ids).unsqueeze(0).to(device),
                    attention_mask=torch.tensor(att).unsqueeze(0).to(device),
                    point_cloud=torch.tensor(pc).unsqueeze(0).to(device),
                    max_new_tokens=768, pad_token_id=tok.pad_token_id)
            s = tok.batch_decode(out)[0]
            b = s.find("<|im_start|>") + 12
            e = s.find("<|endoftext|>")
            code = s[b:e] if e > 0 else s[b:]
            open(os.path.join(py_out, mid + ".py"), "w").write(code)
            ok += 1
        except Exception as ex:
            print(f"FAIL {mid}: {str(ex)[:90]}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"...{i + 1}/{len(files)} ok={ok}", flush=True)
    print(f"DONE {ok}/{len(files)} code files -> {py_out}")


if __name__ == "__main__":
    main()
