#!/usr/bin/env bash
set -euo pipefail

# Spark-specific 4-node Ray launcher for the GLM-5.2 B12X vLLM stack.
# One GB10 GPU per host; NCCL/RDMA over the RoCE fabric; tiny Ray object store;
# no B12X PCIe allreduce. Start this first, then run serve.sh.
#
# EDIT markers: IMAGE, MODEL_DIR, HEAD_IP, WORKER_IPS, HS_IFACE, SSH_KEY, and the
# ssh_dest_for_ip() user mapping must match your cluster.

IMAGE="${IMAGE:-vllm-zatz-dcp:probe}"                          # EDIT: your image tag
MODEL_DIR="${MODEL_DIR:-/var/tmp/models/glm52-int4-int8mix}"   # EDIT: local weights path (every node)
HEAD_IP="${HEAD_IP:-192.168.192.1}"                            # EDIT: head node RoCE IP
RAY_PORT="${RAY_PORT:-26479}"
OBJECT_STORE="${OBJECT_STORE:-134217728}"
OBJECT_SPILLING_DIR="${OBJECT_SPILLING_DIR:-/var/tmp/ray-spill}"
WORKER_IPS="${WORKER_IPS:-192.168.192.2 192.168.192.3 192.168.192.4}"  # EDIT: worker RoCE IPs
SSH_KEY="${SSH_KEY:-/etc/cluster/cluster.key}"                 # EDIT: SSH key path
HS_IFACE="${HS_IFACE:-enp1s0f0np0}"                            # EDIT: RoCE interface name
# NCCL narrowing: keep RDMA on but pin the HCA and cap channels. Per-channel
# buffer/proxy state across 4 ranks is a material memory cost on unified-memory
# Sparks, and bs=1 perf did not change with more channels in testing.
NCCL_IB_HCA="${NCCL_IB_HCA:-roceP2p1s0f0}"                     # EDIT: your RoCE HCA
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
WORKER_NAME="${WORKER_NAME:-glm-dark-worker}"
DROP_CACHES="${DROP_CACHES:-1}"
VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"

ssh_base=(
  ssh
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -i "${SSH_KEY}"
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o BatchMode=yes
  -o ConnectTimeout=10
)

# EDIT: map each worker IP to its ssh login (user@ip). Adjust for your cluster.
ssh_dest_for_ip() {
  case "$1" in
    192.168.192.2) printf "%s\n" "sparkuser2@$1" ;;
    192.168.192.3) printf "%s\n" "sparkuser3@$1" ;;
    192.168.192.4) printf "%s\n" "sparkuser4@$1" ;;
    *) printf "%s\n" "$1" ;;
  esac
}

# Env baked into every container (head + workers). These are the REQUIRED vars.
docker_common=(
  --network host
  --ipc host
  --privileged
  --security-opt label=disable
  --gpus all
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${MODEL_DIR}:/models:ro"
  -e RAY_memory_usage_threshold=0.99
  -e RAY_memory_monitor_refresh_ms=0
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}"
  -e GLOO_SOCKET_IFNAME="${HS_IFACE}"
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="${NCCL_IB_HCA}"
  -e NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS}"
  -e NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS}"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SAFETENSORS_FAST_GPU=1
  -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_DISABLE_TP_MQ_BROADCASTER=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER}"
  -e VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK}"
  -e VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT}"
  -e USES_B12X=True
)

stop_all() {
  docker rm -f "${HEAD_NAME}" >/dev/null 2>&1 || true
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "$(ssh_dest_for_ip "${ip}")" "docker rm -f '${WORKER_NAME}' >/dev/null 2>&1 || true" &
  done
  wait
}

# Drop page caches everywhere — GB10 unified memory stalls big loads in kernel
# reclaim otherwise. (Also worth running in a 60s loop DURING weight load.)
drop_caches_all() {
  if [[ "${DROP_CACHES}" != "1" ]]; then
    return 0
  fi
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "$(ssh_dest_for_ip "${ip}")" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true" &
  done
  wait
}

start_head() {
  docker run -d --name "${HEAD_NAME}" \
    "${docker_common[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    -e HOST_IP="${HEAD_IP}" \
    "${IMAGE}" \
    bash -lc "mkdir -p '${OBJECT_SPILLING_DIR}' && ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --object-store-memory=${OBJECT_STORE} --object-spilling-directory='${OBJECT_SPILLING_DIR}' --num-cpus=1 --num-gpus=1 --include-dashboard=false --include-log-monitor=false --disable-usage-stats --temp-dir=/tmp/ray-vllm-head --block" \
    >/tmp/glm-dark-head.cid
}

start_workers() {
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "$(ssh_dest_for_ip "${ip}")" \
      "docker run -d --name '${WORKER_NAME}' --network host --ipc host --privileged --security-opt label=disable --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -v '${MODEL_DIR}:/models:ro' -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e NCCL_SOCKET_IFNAME='${HS_IFACE}' -e GLOO_SOCKET_IFNAME='${HS_IFACE}' -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA='${NCCL_IB_HCA}' -e NCCL_MAX_NCHANNELS='${NCCL_MAX_NCHANNELS}' -e NCCL_MIN_NCHANNELS='${NCCL_MIN_NCHANNELS}' -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_DISABLE_TP_MQ_BROADCASTER=1 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e VLLM_USE_B12X_SPARSE_INDEXER='${VLLM_USE_B12X_SPARSE_INDEXER}' -e VLLM_DCP_GLOBAL_TOPK='${VLLM_DCP_GLOBAL_TOPK}' -e VLLM_DCP_SHARD_DRAFT='${VLLM_DCP_SHARD_DRAFT}' -e USES_B12X=True -e VLLM_HOST_IP='${ip}' -e HOST_IP='${ip}' '${IMAGE}' bash -lc \"mkdir -p '${OBJECT_SPILLING_DIR}' && ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${ip} --object-store-memory=${OBJECT_STORE} --object-spilling-directory='${OBJECT_SPILLING_DIR}' --num-cpus=1 --num-gpus=1 --include-log-monitor=false --disable-usage-stats --temp-dir=/tmp/ray-vllm-worker --block\" >/tmp/glm-dark-worker.cid" &
  done
  wait
}

wait_cluster() {
  for _ in $(seq 1 60); do
    if docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} 2>/dev/null | grep -q '/4.0 GPU'"; then
      docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} | sed -n '1,80p'"
      return 0
    fi
    sleep 3
  done
  docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} || true"
  return 1
}

stop_all
drop_caches_all
start_head
sleep 5
start_workers
wait_cluster
