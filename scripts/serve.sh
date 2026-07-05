#!/usr/bin/env bash
set -euo pipefail

# Start GLM-5.2 serving inside an already-running 4x DGX Spark Ray cluster.
# Run launch-ray.sh first, then source glm52-qt-dcp4-655k.env and run this.
#
# This wrapper keeps the Spark-specific choices explicit and avoids the
# single-host RTX defaults in upstream serve-glm52.sh.
#
# EDIT markers: HEAD_NAME / HEAD_IP / HS_IFACE must match your cluster.

HEAD_NAME="${HEAD_NAME:-glm-dark-head}"            # EDIT: head container name
HEAD_IP="${HEAD_IP:-192.168.192.1}"                # EDIT: head node RoCE IP
RAY_PORT="${RAY_PORT:-26479}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8210}"
PROFILE="${PROFILE:-custom}"
HS_IFACE="${HS_IFACE:-enp1s0f0np0}"                # EDIT: RoCE interface name
STOP_EXISTING_API="${STOP_EXISTING_API:-1}"

MODEL="/models"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
DCP_SIZE="${DCP_SIZE:-}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
ENABLE_MTP="${ENABLE_MTP:-}"
LOG_FILE="${LOG_FILE:-/tmp/glm52-spark-${PROFILE}.log}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
REASONING_PARSER="${REASONING_PARSER:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"

# DeepSeek-Sparse-Attention per-layer full/sparse pattern (F=full, S=sparse).
INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
if [[ -z "${HF_OVERRIDES:-}" ]]; then
  HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"
fi

case "${PROFILE}" in
  custom)
    : "${DCP_SIZE:?DCP_SIZE is required for PROFILE=custom}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required for PROFILE=custom}"
    : "${MAX_NUM_BATCHED_TOKENS:?MAX_NUM_BATCHED_TOKENS is required for PROFILE=custom}"
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION is required for PROFILE=custom}"
    : "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required for PROFILE=custom}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ENABLE_MTP="${ENABLE_MTP:-1}"
    ;;
  *)
    echo "Unknown PROFILE '${PROFILE}' (this repo ships the 'custom' 655K profile)." >&2
    exit 2
    ;;
esac

# Confirm all 4 GPUs are in the Ray cluster before serving.
docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} | grep -q '/4.0 GPU'"

if [[ "${STOP_EXISTING_API}" == "1" ]]; then
  docker exec "${HEAD_NAME}" bash -lc "pkill -f '[v]llm.entrypoints.openai.api_server' >/dev/null 2>&1 || true"
  sleep 2
fi

docker exec -d \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${HS_IFACE}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_MIN_NCHANNELS=4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER="${USE_B12X_SPARSE_INDEXER}" \
  -e VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}" \
  -e VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}" \
  -e VLLM_DISABLE_TP_MQ_BROADCASTER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e USES_B12X=True \
  -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  "${HEAD_NAME}" bash -lc "$(cat <<EOF
set -euo pipefail
args=(
  python3 -m vllm.entrypoints.openai.api_server
  --model '${MODEL}'
  --tokenizer '${MODEL}'
  --served-model-name '${SERVED_MODEL_NAME}'
  --trust-remote-code
  --download-dir '${MODEL}'
  --load-format '${LOAD_FORMAT}'
  --quantization '${QUANTIZATION}'
  --distributed-executor-backend ray
  --tensor-parallel-size '${TP_SIZE}'
  --decode-context-parallel-size '${DCP_SIZE}'
  --dcp-comm-backend '${DCP_COMM_BACKEND}'
  --dcp-kv-cache-interleave-size '${DCP_KV_CACHE_INTERLEAVE_SIZE}'
  --pipeline-parallel-size '${PP_SIZE}'
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}'
  --max-model-len '${MAX_MODEL_LEN}'
  --max-num-seqs '${MAX_NUM_SEQS}'
  --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}'
  --generation-config vllm
  --hf-overrides '${HF_OVERRIDES}'
  --port '${PORT}'
  --host '${HOST}'
  --no-enable-log-requests
)
if [[ '${ENABLE_PREFIX_CACHING}' == '0' ]]; then
  args+=(--no-enable-prefix-caching)
fi
if [[ '${ENFORCE_EAGER}' == '1' ]]; then
  args+=(--enforce-eager)
fi
if [[ -n '${MAX_CUDAGRAPH_CAPTURE_SIZE}' ]]; then
  args+=(--max-cudagraph-capture-size '${MAX_CUDAGRAPH_CAPTURE_SIZE}')
fi
if [[ -n '${KV_CACHE_MEMORY_BYTES}' ]]; then
  args+=(--kv-cache-memory-bytes '${KV_CACHE_MEMORY_BYTES}')
fi
if [[ '${KV_CACHE_DTYPE}' != 'auto' ]]; then
  args+=(--kv-cache-dtype '${KV_CACHE_DTYPE}')
fi
if [[ '${ATTENTION_BACKEND}' != 'auto' ]]; then
  args+=(--attention-backend '${ATTENTION_BACKEND}')
fi
if [[ '${MOE_BACKEND}' != 'auto' ]]; then
  args+=(--moe-backend '${MOE_BACKEND}')
fi
if [[ -n '${REASONING_PARSER}' ]]; then
  args+=(--reasoning-parser '${REASONING_PARSER}')
fi
if [[ -n '${TOOL_CALL_PARSER}' ]]; then
  args+=(--tool-call-parser '${TOOL_CALL_PARSER}')
fi
if [[ '${ENABLE_AUTO_TOOL_CHOICE}' == '1' ]]; then
  args+=(--enable-auto-tool-choice)
fi
if [[ '${ENABLE_MTP}' == '1' ]]; then
  speculative_config='{"model":"${MODEL}","method":"mtp","num_speculative_tokens":${NUM_SPECULATIVE_TOKENS},"moe_backend":"${MOE_BACKEND}","draft_attention_backend":"${DRAFT_ATTENTION_BACKEND}","draft_sample_method":"${DRAFT_SAMPLE_METHOD}"}'
  args+=(--speculative-config "\${speculative_config}")
fi
if [[ -n '${LONG_PREFILL_TOKEN_THRESHOLD}' ]]; then
  args+=(--long-prefill-token-threshold '${LONG_PREFILL_TOKEN_THRESHOLD}')
fi
if [[ '${ASYNC_SCHEDULING}' == '1' ]]; then
  args+=(--async-scheduling)
fi
printf 'Starting %s on port %s\\n' '${SERVED_MODEL_NAME}' '${PORT}' >'${LOG_FILE}'
printf '%q ' "\${args[@]}" >>'${LOG_FILE}'
printf '\\n' >>'${LOG_FILE}'
exec "\${args[@]}" >>'${LOG_FILE}' 2>&1
EOF
)"

echo "Started ${SERVED_MODEL_NAME}; log: docker exec ${HEAD_NAME} tail -f ${LOG_FILE}"
echo "Endpoint: http://${HEAD_IP}:${PORT}/v1"
echo "Expect boot log: 'GPU KV cache size: 657,664 tokens'"
