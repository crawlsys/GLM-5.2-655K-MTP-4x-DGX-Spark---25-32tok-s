# scripts/

Genericized versions of the cluster's launch/serve recipe. Every hardcoded IP,
user, SSH key path, and interface name has been replaced with a placeholder
marked `# EDIT:` — substitute your own before running.

| file | what |
|------|------|
| `glm52-qt-dcp4-655k.env` | the DCP4 / 655,360-token + MTP k=3 profile. `source` it before `serve.sh`. |
| `launch-ray.sh` | starts the 4-node Ray cluster (head + 3 workers) on the RoCE fabric. Run first. |
| `serve.sh` | launches the vLLM OpenAI API server (`:8210`) inside the running cluster. |

## Order of operations

```bash
# 1. Ray cluster up (waits for 4.0/4.0 GPU)
./launch-ray.sh

# 2. Load the profile
set -a; source glm52-qt-dcp4-655k.env; set +a

# 3. Serve
./serve.sh
# -> http://<HEAD_IP>:8210/v1
# -> boot log should read: GPU KV cache size: 657,664 tokens
```

## What to edit

- **`glm52-qt-dcp4-655k.env`**: `IMAGE`, `MODEL_DIR`, `NCCL_SOCKET_IFNAME`.
- **`launch-ray.sh`**: `IMAGE`, `MODEL_DIR`, `HEAD_IP`, `WORKER_IPS`, `SSH_KEY`,
  `HS_IFACE`, `NCCL_IB_HCA`, and the `ssh_dest_for_ip()` user mapping.
- **`serve.sh`**: `HEAD_NAME`, `HEAD_IP`, `HS_IFACE` (or pass them via env / the
  sourced `.env`).

The `.env` overrides the defaults baked into `serve.sh`, so in practice you edit
the `.env` and only touch the two shell scripts for cluster-topology specifics
(IPs, SSH users, key path, interface, HCA).
