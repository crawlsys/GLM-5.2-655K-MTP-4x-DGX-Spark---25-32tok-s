# patches/

Three source patches, baked into the vLLM image before it is committed and
distributed to the cluster. Each is an **idempotent, anchored Python patcher**:
it locates a specific anchor in the installed vLLM source and rewrites it in
place, and re-running is a no-op once applied. Paths assume the stock install
location `/usr/local/lib/python3.12/dist-packages/vllm/...`.

Run all three against the image's installed vLLM:

```bash
python3 fix-mtp-draft-fused-qkv-mapping.py
python3 fix-mla-int32-chunked-prefill.py
python3 fix-dsa-indexer-block-table.py
```

All target vLLM ref
`local-inference-lab/vllm@codex/dcp-globaltopk-sharddraft-defaults-20260622`.

---

## 1. `fix-mtp-draft-fused-qkv-mapping.py` — the headline fix

**Patches:** `model_executor/models/deepseek_mtp.py`

**First published fix** enabling MTP speculative decoding on compressed-tensors
(int-quantized) GLM-5.2 checkpoints.

The MTP draft builds its quant config from a fresh `CompressedTensorsConfig`
with an **empty** `packed_modules_mapping`. The target model registers
`fused_qkv_a_proj -> [q_a_proj, kv_a_proj_with_mqa]` (and `gate_up_proj ->
[gate_proj, up_proj]`), but the draft never inherits them. compressed-tensors
then can't match the fused module, silently builds it **unquantized** (`.weight`
not `.weight_packed`), and `DeepSeekMTP.load_weights` KeyErrors on
`model.layers.78.mtp_block.self_attn.kv_a_proj_with_mqa.weight_packed`.

The patch seeds the draft's `packed_modules_mapping` with those two mappings
(the `fused_qkv_a_proj` one only when `q_lora_rank` is set), immediately after
the draft quant config is constructed. Both are `setdefault`, so nothing is
clobbered.

Discovered 2026-07-05 via checkpoint-vs-loader forensics; validated by a clean
655K + MTP boot with acceptance length ~3.0. Zatz (thread 375416 post 15) has
equivalent unpublished fixes; this is the first published version.

## 2. `fix-mla-int32-chunked-prefill.py` — credit Zatz

**Patches:** `model_executor/layers/attention/mla_attention.py`

Verbatim from Zatz, thread 375416 post 15. Changes
`) and _kv_b_proj_w_dtype != torch.uint8:` to
`) and _kv_b_proj_w_dtype not in (torch.uint8, torch.int32):`.

Fixes garbage output past 4K-token prompts on int-quantized checkpoints:
chunked prefill casts activations to the Marlin `int32` weight dtype past the
default 4096 chunk size. NVFP4 (uint8) was unaffected, so the fork author never
saw it.

## 3. `fix-dsa-indexer-block-table.py` — co-discovered with ciprianveg

**Patches:** `v1/attention/backends/mla/indexer.py`

The DSA indexer's `expanded_block_table_buffer` is sized from `max_model_len`
alone, but the scheduler's block table can be one block wider (MTP spec tokens /
scheduler overhang), crashing the engine at concurrency >= 3 with
`RuntimeError: The expanded size of the tensor (N) must match the existing size
(N+1)`. Adds `+ 1` to the `max_num_blocks_per_req = cdiv(...)` result before the
buffer is allocated.

Co-discovered independently with forum user ciprianveg (thread 374125 posts
105/107, his `mods/fix-dsa-block-table-dim`).
