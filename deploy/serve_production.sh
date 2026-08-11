#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 on 8x A100 80GB PCIe (SM80).
# Launched by /etc/systemd/system/deepseek-v4-flash.service.
# The API key is NOT set here -- systemd injects VLLM_API_KEY from the 0600
# EnvironmentFile /etc/deepseek-v4-flash.env, so it never enters the argv.
set -euo pipefail

source /data1/dsv4/v2_attempt/vllm/.venv/bin/activate

# --- CUDA runtime: libcudart.so.12, from the cu129 torch wheel (driver >= 550) ---
export LD_LIBRARY_PATH="/data1/dsv4/v2_attempt/vllm/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# --- keep every writer off the 100%-full root filesystem ---
export HOME=/data1/dsv4/v2_attempt/cache/home
export HF_HOME=/data1/dsv4/v2_attempt/cache/hf
export XDG_CACHE_HOME=/data1/dsv4/v2_attempt/cache
export XDG_CONFIG_HOME=/data1/dsv4/v2_attempt/cache/config
export VLLM_CONFIG_ROOT=/data1/dsv4/v2_attempt/cache/config/vllm
export TRITON_CACHE_DIR=/data1/dsv4/v2_attempt/cache/triton
export VLLM_CACHE_ROOT=/data1/dsv4/v2_attempt/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/data1/dsv4/v2_attempt/cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/data1/dsv4/v2_attempt/cache/flashinfer
export TMPDIR=/data1/dsv4/v2_attempt/cache/tmp
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$VLLM_CONFIG_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TMPDIR"

# --- correctness: this fork auto-enables breakable cudagraph for
# DeepseekV4ForCausalLM when the var is UNSET (vllm/config/vllm.py:1208), which
# is the known cause of silent batch>1 output corruption. Must stay explicit. ---
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# --- PCIe / no-NVLink topology ---
export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SPARSE_RAGGED_FAST_SCAN=1

# --- gcc12 toolchain: required at RUNTIME, not just build time. CUDA 12.4 nvcc
# rejects host gcc > 13, and the system gcc has no C++ frontend installed. ---
export PATH="/data1/dsv4/gxx_env12/bin:$PATH"
export CXX=x86_64-conda-linux-gnu-g++
export CC=x86_64-conda-linux-gnu-gcc
export CUDAHOSTCXX=/data1/dsv4/gxx_env12/bin/x86_64-conda-linux-gnu-g++

cd /data1/dsv4/v2_attempt/vllm
exec .venv/bin/vllm serve /data1/models/DeepSeek-V4-Flash-0731 --trust-remote-code \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 16384 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
  --hf-overrides '{"head_dtype": "float32"}' \
  --override-generation-config '{"top_p": 0.95}' \
  --host 0.0.0.0 --port 8000
