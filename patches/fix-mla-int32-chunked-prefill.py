#!/usr/bin/env python3
"""Fix garbage output past ~4K-token prompts on int-quantized GLM-5.2 checkpoints.

Credit: Zatz, NVIDIA developer forum thread 375416 post 15 (verbatim change).

Root cause
----------
In the MLA attention path, chunked prefill casts activations to the *weight*
dtype of `kv_b_proj`. For Marlin-packed int checkpoints that weight dtype is
`torch.int32`. Past the default 4096-token chunk size, prefill is chunked and
the cast fires — corrupting activations and producing garbage tokens. The guard
that skips this cast only excluded `torch.uint8` (NVFP4), so the fork author,
who only ran NVFP4, never saw the int32 case.

The fix
-------
Widen the guard to skip both uint8 and int32:

    ) and _kv_b_proj_w_dtype != torch.uint8:
  ->
    ) and _kv_b_proj_w_dtype not in (torch.uint8, torch.int32):

Idempotent: safe to run more than once.
"""

import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py"

OLD = ") and _kv_b_proj_w_dtype != torch.uint8:"
NEW = ") and _kv_b_proj_w_dtype not in (torch.uint8, torch.int32):"


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    if NEW in src:
        print("[fix-mla-int32-chunked-prefill] already applied; nothing to do.")
        return 0

    if OLD not in src:
        print(
            "[fix-mla-int32-chunked-prefill] ERROR: anchor not found in "
            f"{TARGET}. vLLM version mismatch?",
            file=sys.stderr,
        )
        return 1

    patched = src.replace(OLD, NEW, 1)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(patched)

    print(f"[fix-mla-int32-chunked-prefill] applied to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
