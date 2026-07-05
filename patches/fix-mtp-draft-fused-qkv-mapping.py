#!/usr/bin/env python3
"""Enable MTP speculative decoding on compressed-tensors (int-quantized) GLM-5.2.

THE HEADLINE FIX of this deployment — the first published version of it.

Root cause
----------
`DeepSeekMultiTokenPredictorLayer` builds the *draft* model's quant config via
`get_draft_quant_config(vllm_config)`, which returns a **fresh**
`CompressedTensorsConfig` whose `packed_modules_mapping` is **empty**. The
*target* model (`DeepseekV2ForCausalLM`) registers the fused-module mappings it
needs at construction time — notably:

    fused_qkv_a_proj -> [q_a_proj, kv_a_proj_with_mqa]     (when q_lora_rank is set)
    gate_up_proj     -> [gate_proj, up_proj]

but the draft never inherits them. compressed-tensors then cannot match the
fused module against any config group, so it silently builds `fused_qkv_a_proj`
**unquantized** (a plain `.weight`, not a `.weight_packed`). At load time,
`DeepSeekMTP.load_weights` iterates the checkpoint — which *does* have the
packed tensor — and KeyErrors:

    KeyError: 'model.layers.78.mtp_block.self_attn.kv_a_proj_with_mqa.weight_packed'

The fix
-------
Immediately after the draft quant config is built, seed its
`packed_modules_mapping` with the same fused mappings the target uses. Both are
`setdefault`, so this never clobbers an existing mapping.

Applies to vLLM ref `codex/dcp-globaltopk-sharddraft-defaults-20260622`
(`local-inference-lab/vllm`). Discovered 2026-07-05 via checkpoint-vs-loader
forensics; validated by a clean 655K + MTP boot with acceptance length ~3.0.

Note: Zatz (forum thread 375416 post 15) has equivalent *unpublished* fixes
("compressed-tensors quants need 2 extra small draft-loading fixes"). This is
the first published version.

Idempotent: safe to run more than once.
"""

import re
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_mtp.py"

ANCHOR = """        quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(
            config, vllm_config, get_draft_quant_config(vllm_config)
        )
"""

INSERT = """        if quant_config is not None and hasattr(
            quant_config, "packed_modules_mapping"
        ):
            quant_config.packed_modules_mapping.setdefault(
                "gate_up_proj", ["gate_proj", "up_proj"]
            )
            if getattr(config, "q_lora_rank", None) is not None:
                quant_config.packed_modules_mapping.setdefault(
                    "fused_qkv_a_proj", ["q_a_proj", "kv_a_proj_with_mqa"]
                )
"""


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    if 'quant_config.packed_modules_mapping.setdefault(\n                "fused_qkv_a_proj"' in src:
        print("[fix-mtp-draft-fused-qkv-mapping] already applied; nothing to do.")
        return 0

    if ANCHOR not in src:
        print(
            "[fix-mtp-draft-fused-qkv-mapping] ERROR: anchor not found in "
            f"{TARGET}. vLLM version mismatch?",
            file=sys.stderr,
        )
        return 1

    patched = src.replace(ANCHOR, ANCHOR + INSERT, 1)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(patched)

    print(f"[fix-mtp-draft-fused-qkv-mapping] applied to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
