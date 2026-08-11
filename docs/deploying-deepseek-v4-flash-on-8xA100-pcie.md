# Running DeepSeek-V4-Flash-0731 on 8×A100 PCIe

*The model shipped for Hopper and Blackwell tensor cores. This machine has neither — eight Ampere cards on plain PCIe, no NVLink. Here's every wall we hit getting it to production, the numbers that came out the other side, and the things we got wrong the first time and had to correct.*

**Result, up front:**

| | |
|---|---|
| Single-stream decode | ~40 tok/s |
| Aggregate @ 32 concurrent (short prompts) | ~1,070 tok/s |
| Aggregate @ 64 concurrent (short prompts) | ~1,718 tok/s |
| Context window configured | 1,048,576 tokens |
| Largest prompt actually served | 893,046 tokens |
| Long-context concurrency (measured) | ~2 requests in flight at 423K; **serialized at ~893K** |
| Speculative decoding | DSpark, k=7 |

That second-to-last row is the one most guides would omit. It is the single most important number for anyone planning to serve parallel long-context traffic, and we got it wrong in the first version of this document. See [Long-context concurrency](#long-context-concurrency-the-number-that-matters) below.

---

DeepSeek-V4-Flash-0731 is a 284B-parameter mixture-of-experts model — 13B active per token — natively quantized to a mix of FP8 and MXFP4, shipping at 166.9GB on disk with a 1,048,576-token context window. It's a serious model. It was also, as of its release, built for hardware that doesn't exist in most people's server rooms yet: Hopper and Blackwell tensor cores with native FP8 and FP4 support.

We had eight A100 80GB PCIe cards. Ampere. No NVLink — the GPUs talk over plain PCIe, split across two NUMA islands of four. No native FP4. No native FP8 matmul either — Ampere's tensor cores stop at bfloat16 and int8. On paper, this hardware wasn't on the model's supported list at all.

It runs now, correctly, at production-grade throughput, with tool calling and configurable reasoning effort. This is the record of how.

## Before the model: the machine itself

### Three infrastructure bugs that had nothing to do with DeepSeek

**The root disk was full.** `df -h /` reported 0 bytes available on a shared machine with other people's data. The move was to redirect everything new — model weights, virtualenv, pip/HF/Triton/FlashInfer caches, `TMPDIR` — onto a mostly-empty NVMe mount, with a symlink at the path the model was *supposed* to live at.

Be thorough about this: our first pass missed two writers and kept touching `/` for days. FlashInfer's JIT cache follows `$HOME`, not `XDG_CACHE_HOME`, and vLLM's usage stats follow `XDG_CONFIG_HOME`. Both silently succeeded only because the service runs as root and can draw on ext4's reserved blocks. The full set you need to redirect is in the launcher below.

One honest caveat we cannot fix cheaply: the Python **interpreter and stdlib** still live on the root filesystem, because the venv was created from a conda env on that filesystem, so its `bin/python3.12` is a symlink. Nothing is *written* to `/`, but ~245MB of interpreter and C extensions are read from it. A truly self-contained deployment would build the venv with `python -m venv --copies` from an interpreter that also lives on the data disk. Until then this is a **deletion hazard**: the obvious response to a full root filesystem is to prune home directories, and a conda prefix is the obvious target. Leave a DO-NOT-DELETE marker in the interpreter's directory.

**IPv6 was configured but not routable.** Downloads to PyPI and GitHub hung for minutes rather than failing. DNS returned AAAA records, the resolver preferred IPv6, and IPv6 had no route out — so every connection burned a full timeout before falling back. The fix is one line:

```
# /etc/gai.conf — prefer IPv4 when IPv6 has no route
precedence ::ffff:0:0/96  100
```

Note this is a **machine-wide** change: appending a single `precedence` rule replaces glibc's entire default precedence table with your one rule. On a shared box, tell the other users, or scope the fix to your own service instead.

**A proxy helped for some traffic and hurt for other traffic.** ModelScope (where the weights live) was fastest direct. PyPI and GitHub were roughly 3× faster *through* a local proxy. There is no single right setting — measure per destination.

## Getting the model

```bash
export MODELSCOPE_CACHE=/data1/dsv4/cache/modelscope
modelscope download --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /data1/models/DeepSeek-V4-Flash-0731 --max-workers 8
```

166.9GB across 48 safetensors shards. Verify before you trust it: 48 `*.safetensors` files present, `metadata.total_size` in `model.safetensors.index.json` equal to 166,878,536,440, and no `*.incomplete` leftovers.

## The wall

### Why the stock build doesn't run on Ampere at all

The model uses a per-layer compression step ("hyperconnection" prenorm) inside its attention stack. vLLM's implementation calls straight into [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), and DeepGEMM's kernel for that operation — `tf32_hc_prenorm_gemm` — ships compiled for SM90 (Hopper) and SM100/SM120 (Blackwell) only. There is no SM80 kernel. Not a slow fallback — the op does not exist for Ampere.

> **This is a wall, not a flag.** If a model's forward pass calls a fused kernel with no build for your architecture, no launch configuration fixes it. Go looking for a fork.

An open vLLM issue tracks SM8x support for this model, with maintainers on record that Ampere support "better lives in a fork."

## Choosing a fork

We went through two community forks, and the difference between them mattered more than anything we configured.

**1. `Lasimeri/vllm-dsv4-ampere`** — a from-scratch reimplementation of the Hopper-only kernels in Triton and PyTorch. It got the model running end to end after three small patches on our side (a missing MoE quantization case in a weight-loading path, a hard-coded DeepGEMM availability check, and an op call missing an argument a newer vLLM commit had added). Eager: **~4 tok/s**. With CUDA graphs in piecewise mode: **~14 tok/s**.

**2. `haosdent/vllm@dsv4-flash-a100`** — far more actively maintained, incorporating fixes from several contributors, rebased on a current vLLM commit, and crucially including a working Ampere implementation of DeepSeek's own speculative decoding (DSpark). This is what we shipped: **~40 tok/s single-stream**.

**Pin the commit.** A branch name is not a reproducible reference, and correctness results are only meaningful against a specific tree:

```bash
git clone --branch dsv4-flash-a100 --single-branch https://github.com/haosdent/vllm.git
cd vllm && git checkout 01ecc8e4f62ca6f3c2add67eede38aa2548204ce
```

### Building it

This branch carries real CUDA source changes, so it must be compiled — `VLLM_USE_PRECOMPILED=1` would silently give you the unmodified upstream binary without the Ampere fixes.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U uv
uv pip install -r requirements/build/cuda.txt -r requirements/build/rust.txt --torch-backend=auto

export TORCH_CUDA_ARCH_LIST="8.0"          # SM80 only; keeps the compile bounded
export CUDAHOSTCXX=/data1/dsv4/gxx_env12/bin/x86_64-conda-linux-gnu-g++
export CXX=x86_64-conda-linux-gnu-g++
export CC=x86_64-conda-linux-gnu-gcc
export PATH="/data1/dsv4/gxx_env12/bin:$PATH"
export MAX_JOBS=64

pip install -e . --no-build-isolation -v   # ~24 minutes on a 112-core box
```

Three things that will bite you:

- **`CUDAHOSTCXX` is mandatory and separate from `CXX`.** CMake's CUDA compiler probe does not read `CXX`/`CC`; without `CUDAHOSTCXX` it fails at configure time with `nvcc fatal: Failed to preprocess host compiler properties`.
- **gcc version ceiling.** CUDA 12.4's `nvcc` refuses any host compiler newer than gcc 13. Our system had gcc 11 with no C++ frontend and a full root disk, so we installed a portable gcc 12 into an isolated conda prefix (`conda create --prefix /data1/dsv4/gxx_env12 -c conda-forge gxx_linux-64=12 gcc_linux-64=12`) — system untouched, fully reversible. The same toolchain must be on `PATH` **at serve time**, not just build time, because kernels JIT-compile at runtime.
- **Don't let the resolver pick torchvision independently.** A mismatched `torchvision` against this `torch` throws `RuntimeError: operator torchvision::nms does not exist` at import. Install it pinned to the same CUDA build: `uv pip install torchvision --torch-backend=cu129 --no-deps`.

## "Is it actually correct?"

### Token counts are not a correctness check

This fork family has a documented history of producing *fluent-looking garbage* under concurrent load — leaked training fragments, null content, repetition loops — while the API returns a normal 200 with a plausible token count. What actually verifies it:

- **Needle-in-a-haystack.** A unique secret buried mid-document (we tested 20K–893K tokens), asked for by name. Proves the sparse attention indexer and long-context path do real work.
- **Read the generated text under concurrency**, not just the token count. We issued 32 and then 128 concurrent client requests of 600 tokens each and inspected every response for leaked fragments, null content, and repetition. Note the distinction this article insists on elsewhere: those are *client* concurrencies. The scheduler's own peak was `Running: 64 reqs` — always confirm in-flight depth from the log rather than from how many sockets you opened.
- **Distinct secrets per concurrent request**, to catch cross-request bleed as well as plain wrongness.

All clean. But see the concurrency caveat below — we issued 128 client requests and the scheduler ran up to 64 in flight — that part was real; the "8 concurrent" at 423K was not.

### The correctness flag you must not omit

```bash
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

The community traced the batch>1 corruption to a CUDA-graph replay bug fixed by this one variable. **This fork auto-enables it to `1` when the variable is unset** (`vllm/config/vllm.py:1208`) for `DeepseekV4ForCausalLM` — so omitting it from your launcher does not give you the default, it gives you the broken configuration. It must be exported explicitly.

## The TP-vs-PP debate, settled by data

The instinct for PCIe-only hardware: tensor parallelism all-reduces every layer, pipeline parallelism only passes activations at stage boundaries, so PP should win. Two things undercut that here. Most of this model's parameters live in MoE experts, and expert-parallel routing — already enabled — doesn't pay the tensor-sharding cost at all. And an operator on 4×A100 A/B'd TP4 against PP4 on this same fork family: **both corrupted identically** under concurrent load. It was never a parallelism bug; it was the CUDA-graph bug above.

Measured aggregate throughput on TP=8 + expert-parallel, short prompts (re-verified on the shipped configuration):

| Concurrency | Aggregate tok/s |
|---|---|
| 1 | ~40 |
| 8 | ~293 |
| 32 | ~1,070 |
| 64 | ~1,718 |

Per-request speed barely degrades out to 32 concurrent. The PP proposal predicted 500–700 tok/s aggregate; the configuration already running cleared that at 32 concurrent and kept climbing.

## Long-context concurrency: the number that matters

**This is the part we got wrong first time, and the correction is the most useful thing in this document.**

We reported "8 × 423K tokens concurrent" and "four simultaneous ~900K-token requests." Neither was concurrent. The scheduler log tells the truth:

```
22:19:33  Running: 1 reqs  Waiting: 2 reqs  GPU KV cache usage: 25.1%
22:23:23  Running: 1 reqs  Waiting: 1 reqs  GPU KV cache usage: 25.1%
22:27:13  Running: 1 reqs  Waiting: 0 reqs  GPU KV cache usage: 25.1%
```

Four ~893K requests completing exactly 230 seconds apart, one at a time. The "923 seconds for the slowest" we quoted is 4 × 230s of *queueing*. Peak KV cache usage across the entire log never exceeded 25.7% — one request's worth. The 8×423K run peaked at `Running: 2` and served 43% of its prefill from prefix cache (all eight shared one haystack).

The correctness results stand. The implied parallel capacity did not exist.

### Why, and what actually controls it

Raising `--max-num-batched-tokens` from 2048 to 16384 cut admission capacity at 1M context by 68%:

| `--max-num-batched-tokens` | KV memory | KV pool | Max concurrency @ 1M |
|---|---|---|---|
| 2,048 | 47.86 GiB | 9,291,477 tok | **8.86×** |
| 16,384 | 45.70 GiB | 2,944,541 tok | **2.81×** |

Our first explanation — bigger batches reserve activation memory that the KV cache can't use — accounts for only 4.5% of that (47.86 → 45.70 GiB). The real mechanism is a **per-request admission reservation**. This model's compressed layers (`sliding_window: 128`) reserve
`min(chunk_size + max_in_flight_tokens, max_model_len)` blocks *per request*, and
`max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens`
(`vllm/v1/kv_cache_interface.py:522`, `vllm/config/vllm.py:553`). The reservation scales directly with the knob.

Two practical consequences:

1. **`--max-model-len` is a per-request reservation, not just a cap.** Configuring 1M when your workload sends 200K costs you concurrency on *every* request. Set it to what you actually send.
2. **`--max-num-batched-tokens` trades prefill speed against long-context parallelism.** 16384 genuinely helped serialized latency (6×91K: 110s → 71s; 8×423K: one timeout → all 8 complete). It did not relieve serialization, and at 65536 the server won't even start — the KV pool drops below what one 1M request needs.

If you serve parallel long-context traffic, measure `Running:` in the scheduler log. Do not infer concurrency from wall-clock.

## Security: the API key does not protect the inference API

vLLM's auth middleware only guards paths matching `GUARDED_PREFIX = ("/v1", "/v2", "/inference")`. **`POST /invocations` — the SageMaker-compatibility route that dispatches to the exact same chat-completions handler — falls outside it and runs with no credentials at all.**

```
POST /v1/chat/completions   (no key)  ->  401 Unauthorized
POST /invocations           (no key)  ->  200 OK, full completion
```

`/tokenize`, `/detokenize`, `/generative_scoring` and the *mutating* `/scale_elastic_ep` are likewise unguarded. With `--host 0.0.0.0` and no host firewall, that is unrestricted use of the cluster by anyone who can reach the port.

Pick at least one of:

- **Bind to `127.0.0.1`** and front it with an authenticating reverse proxy.
- **Firewall** everything except your client subnet.
- **Widen the guard list** in the fork (what we did, since we needed LAN access and the host had no packet filter installed):

```python
# vllm/entrypoints/serve/utils/server_utils.py
GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/invocations", "/tokenize",
                  "/detokenize", "/generative_scoring", "/scale_elastic_ep",
                  "/is_scaling_elastic_ep", "/load")
```

Verify with an unauthenticated `POST /invocations` returning 401 before calling the deployment closed. Keep `/health`, `/ping` and `/metrics` open so monitoring still works.

**Keep the key out of argv.** `--api-key` on the command line is visible in `ps` to every local user, and it is echoed into the server log at startup. Put it in a `0600` `EnvironmentFile` as `VLLM_API_KEY` instead — `/proc/PID/environ` is owner-only, unlike `/proc/PID/cmdline`. Create it before first start:

```bash
umask 077
printf 'VLLM_API_KEY=%s\n' "dsv4-$(python3 -c 'import secrets;print(secrets.token_urlsafe(36))')" \
  > /etc/deepseek-v4-flash.env
chmod 600 /etc/deepseek-v4-flash.env
```

> **If `VLLM_API_KEY` is unset, there is no authentication at all — not even on `/v1`.**
> vLLM installs the auth middleware conditionally: `if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:` (`entrypoints/openai/api_server.py:310`). With no key and no `--api-key`, the middleware is never added and *every* route is open on `0.0.0.0`, silently and with no warning in the log. Widening `GUARDED_PREFIX` protects nothing in that state. If you run the launcher by hand for a smoke test, export `VLLM_API_KEY` first, or bind to `127.0.0.1`.

**There is no output-token ceiling.** A request that omits `max_tokens` is granted ~1,048,571 output tokens — roughly seven hours of decode holding a KV slot. Cap it client-side, or at a proxy.

## What's actually running

The real entrypoint is a systemd unit, not a bare command. Both parts matter.

```bash
#!/usr/bin/env bash
# /data1/dsv4/v2_attempt/serve_production.sh   (0750)
set -euo pipefail
source /data1/dsv4/v2_attempt/vllm/.venv/bin/activate

export LD_LIBRARY_PATH="/data1/dsv4/v2_attempt/vllm/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# keep every writer off the full root filesystem
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

# correctness — must be explicit, the fork auto-enables =1 when unset
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# PCIe / no-NVLink topology
export NCCL_CUMEM_ENABLE=0
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SPARSE_RAGGED_FAST_SCAN=1

# gcc12 needed at RUNTIME for kernel JIT, not just at build time
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
```

```ini
# /etc/systemd/system/deepseek-v4-flash.service
[Unit]
Description=DeepSeek-V4-Flash-0731 vLLM server (SM80 fork, TP=8, 1M context)
After=network-online.target data1.mount
Wants=network-online.target
# everything lives on /data1 — without this the unit starts before the mount
RequiresMountsFor=/data1
StartLimitIntervalSec=1800
StartLimitBurst=3

[Service]
Type=simple
ExecStart=/data1/dsv4/v2_attempt/serve_production.sh
# 0600, holds VLLM_API_KEY. systemd has NO inline comments -- a trailing
# "# ..." on the next line would become part of the value.
EnvironmentFile=/etc/deepseek-v4-flash.env
Restart=on-failure
RestartSec=30
TimeoutStartSec=900
User=root
StandardOutput=append:/data1/dsv4/v2_attempt/serve_final.log
StandardError=append:/data1/dsv4/v2_attempt/serve_final.log
NoNewPrivileges=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
```

**Put your data mount in `/etc/fstab`.** Ours wasn't — it had been mounted by hand, so `RequiresMountsFor=/data1` alone would not have saved it. Use `nofail` so a missing disk degrades instead of blocking boot:

```
UUID=<uuid> /data1 ext4 defaults,nofail,x-systemd.device-timeout=30s 0 2
```

Non-obvious flags, each of which earned its place:

- **`--reasoning-parser deepseek_v4`** — this model has its own encoding scheme, not a Jinja chat template. Without the matching parser, the thinking block leaks into the answer text instead of arriving in a separate `reasoning` field.
- **`--tool-call-parser deepseek_v4`** — *not* `hermes`, despite that being the common default in generic guides. This model has its own tool-call tag format; the generic parser silently fails to extract calls.
- **`--hf-overrides '{"head_dtype": "float32"}'`** — numerical-stability fix used consistently across working community configs on non-native-FP8 hardware.
- **`--override-generation-config '{"top_p": 0.95}'`** — the model's own `generation_config.json` ships `temperature=1.0, top_p=1.0`, and vLLM honors it. The model card recommends `top_p=0.95` for agentic use; without this override every client inherits full-tail sampling, which is the worst default for a stack whose failure mode is fluent-looking garbage.

### Operating it

- **Cold start is ~4–5 minutes.** `Type=simple` reports the unit active as soon as the process spawns, long before the model is loaded — poll `/health` for readiness, not `systemctl is-active`.
- **The log has no rotation by default,** and systemd holds an append fd on it, so a rename-based rotation would leave the server writing into an unlinked inode. Rotate with copy-truncate. (If `logrotate` isn't installed — ours wasn't, despite `/etc/logrotate.d/` existing — a small `oneshot` service on an hourly timer does the job.)
- **`/metrics` is exposed but nothing scrapes it by default.** Point Prometheus at it, or you will not learn the server died until a user tells you.

## Reasoning effort

Passed per request through vLLM's chat-template-kwargs extension, since it isn't part of the standard OpenAI schema:

```json
{
  "model": "deepseek-v4-flash",
  "messages": [...],
  "chat_template_kwargs": { "thinking": true, "reasoning_effort": "max" }
}
```

There are effectively **three** states, not the four the parameter names imply. Verified by hashing the rendered token IDs:

| request | rendered prompt |
|---|---|
| `thinking:false`, or `reasoning_effort:"none"` | thinking **off** |
| `thinking:true` with `"high"`, `"low"`, or any unrecognised value | identical — 5 tokens |
| `thinking:true` with `"max"` or `"xhigh"` | 84 tokens (adds the effort prefix) |

So `"high"` is a **no-op** — byte-identical to just `thinking:true` — and `"none"` turns thinking *off* rather than falling back to high. Only `"max"`/`"xhigh"` changes the prompt.

## The checklist

- [ ] Redirect every cache and temp path off the root disk — including `HOME` (FlashInfer) and `XDG_CONFIG_HOME` (usage stats), which the obvious variables miss.
- [ ] Add the `gai.conf` IPv4-preference line before diagnosing any download as "just slow" — and tell your co-tenants, it's machine-wide.
- [ ] Measure proxy vs. direct per destination.
- [ ] If a model's forward pass calls a fused kernel with no build for your architecture, that's a wall, not a flag.
- [ ] Pin the fork to a commit, not a branch.
- [ ] Export `VLLM_USE_BREAKABLE_CUDAGRAPH=0` explicitly — unset means *enabled* for this model.
- [ ] Verify correctness by reading actual generated text under concurrency; a fast HTTP 200 with garbled content passes every naive check.
- [ ] Confirm concurrency from `Running:` in the scheduler log, never from wall-clock time.
- [ ] Set `--max-model-len` to what you actually send — it is a per-request KV reservation, not just a cap.
- [ ] Tune `--max-num-batched-tokens` against your *worst* realistic concurrent load; the failure at the ceiling is a crash, not a slowdown.
- [ ] Test `POST /invocations` without a key. If it answers, your API key is decorative.
- [ ] Put the key in a `0600` EnvironmentFile, never in argv.
- [ ] Put your data mount in `fstab` with `nofail`, and add `RequiresMountsFor=` to the unit.
- [ ] Cap output tokens somewhere — the server won't.

---

*Hardware: 8× NVIDIA A100 80GB PCIe, dual 4-GPU NUMA islands, no NVLink. Model: [deepseek-ai/DeepSeek-V4-Flash-0731](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/summary). Serving stack: vLLM fork `haosdent/vllm@dsv4-flash-a100` at commit `01ecc8e4f`, built from source for SM80, with one local patch widening `GUARDED_PREFIX`.*
