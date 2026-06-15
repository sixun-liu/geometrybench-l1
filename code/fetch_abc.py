#!/usr/bin/env python3
"""
Download a capped subset of ABC STEP models from the HF mirror
turiya-ai/abc-cad-dataset-organized (split into simple/ and complex/).

Uses only Python stdlib + hf-mirror.com direct `resolve` URLs — avoids the
huggingface_hub httpx bug and needs no proxy from China.

Usage: python fetch_abc.py <n_simple> <n_complex> <out_dir>
"""
import json
import os
import sys
import urllib.parse
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "turiya-ai/abc-cad-dataset-organized"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=90).read()


def list_step(subdir):
    url = f"{ENDPOINT}/api/datasets/{REPO}/tree/main/{subdir}"
    items = json.loads(get(url))
    return sorted(it["path"] for it in items
                  if it.get("type") == "file" and it["path"].endswith(".step"))


def main():
    n_simple, n_complex, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sel = []
    if n_simple > 0:
        sel += list_step("step_files/simple")[:n_simple]
    if n_complex > 0:
        sel += list_step("step_files/complex")[:n_complex]
    print(f"selected {len(sel)} files")

    ok = 0
    for i, p in enumerate(sel):
        dst = os.path.join(out, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:   # resume-skip
            ok += 1
            continue
        try:
            data = get(f"{ENDPOINT}/datasets/{REPO}/resolve/main/{urllib.parse.quote(p)}")
            with open(dst, "wb") as f:
                f.write(data)
            ok += 1
        except Exception as e:
            print(f"DLFAIL {p}: {e}", file=sys.stderr)
        if (i + 1) % 50 == 0:
            print(f"...{i + 1}/{len(sel)} ok={ok}", flush=True)
    print(f"DONE {ok}/{len(sel)} -> {out}")


if __name__ == "__main__":
    main()
