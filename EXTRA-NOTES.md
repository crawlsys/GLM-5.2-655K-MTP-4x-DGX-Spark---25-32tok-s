# Extra notes (kvncrw fork) — what we had to figure out on top of Tony's recipe

**Upstream credit:** this is a fork of [tonyd2wild/GLM-5.2-655K-MTP-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.2-655K-MTP-4x-DGX-Spark)
(and the [25–32 tok/s variant](https://github.com/tonyd2wild/GLM-5.2-655K-MTP-4x-DGX-Spark---25-32tok-s)).
Tony cracked the hangs that sank earlier CosmicRaisins/eugr paths. **Use his README first.**
What follows is only the additional landmines we hit bringing QuantTrio up for agentic (tool-calling) work on a switched 4× DGX Spark fabric.

## 1. cutlass-dsl 4.6.0 (not 4.5.2)

After building Tony's stack (local-inference-lab vLLM DCP branch + b12x-from-source@9cd63a72),
model init died in the b12x sparse-indexer TMA path:

```
b12x/.../contiguous_kernel.py → cutlass cute cpasync → ValueError: Operation creation failed
```

**Fix:** bake `nvidia-cutlass-dsl==4.6.0` (and matching `libs-base` / `libs-cu13`) into the image,
even though vLLM pins `==4.5.2`. That was the only change that unblocked TMA op creation on sm_121a
for us. Image tag we used: `vllm-zatz-dcp:cutlass46`.

```bash
# inside a throwaway container of the built image:
pip install nvidia-cutlass-dsl==4.6.0 \
  nvidia-cutlass-dsl-libs-base==4.6.0 \
  nvidia-cutlass-dsl-libs-cu13==4.6.0
# then docker commit with the original ENTRYPOINT restored
```

## 2. Skip the profiling forward with `--kv-cache-memory-bytes`

Other recipes hung forever in the memory-profiling forward pass. Tony's
`KV_CACHE_MEMORY_BYTES=9000000000` (≈9 GB) **skips that pass** and reports
`GPU KV cache size: ~657k tokens`. Do not remove this "just to reclaim a bit of KV."

## 3. Tool parsers are not optional for agent work

Without all three of:

- `--reasoning-parser glm45`
- `--tool-call-parser glm47`
- `--enable-auto-tool-choice`

the endpoint will answer chat prompts and look healthy while **silently failing structured tool calls**.
We verified with a trivial `get_weather("Tokyo")` before wiring the litellm gateway.

## 4. Weights location: never `/var/tmp` (or any tmpfs)

On these boxes `/var/tmp` can be RAM-backed. Staging a 378 GB QuantTrio there will OOM-kill the node.
Put weights on real disk and hardlink into the HF cache path if your launcher expects one.

## 5. Fabric: one rail with an IP; GID index matters

- Prefer the RoCE rail that actually has the fabric IP. A second rail that is UP but has no IP will
  confuse NCCL if you list both HCAs.
- `NCCL_IB_GID_INDEX=3` (routable RoCEv2 GID) + `NCCL_CUMEM_ENABLE=0` + `NCCL_NET=IB` matched our
  switched star fabric. Your GID index may differ — check `show_gids`.

## 6. Boot is quiet for a long time — do not "fix" it

Cold start is **~8–10 minutes** to `Application startup complete`. The head log goes silent during
DCP weight load. We tore down working serves more than once thinking it was wedged. Rule: after
launch, leave it alone for 10 minutes unless you see an explicit traceback / CUDA error.

## 7. Example env

See `scripts/glm-cluster.example.env` — copy, fill the `EDIT` fields, source it before
`launch-ray.sh` / `serve.sh`.

## Throughput (our agentic load, not synthetic bench)

On long-horizon autonomous pentest work (not interactive chat):

- **Decode:** roughly **20–30 tok/s** end-to-end for the agent loop
- **Early discovery** (big context / recon): **prefill-heavy**, ~**60s** mean step latency over multi-hour stretches
- **Late API probing** (short tool loops): **prefill-light**, ~**20s** mean step latency

Not chat-snappy. Plenty for hands-off agentic work that runs while you do something else.

## License / etiquette

Keep Tony's LICENSE and NOTICE. If you publish numbers or patches derived from this fork, link both
his recipe and this notes file.
