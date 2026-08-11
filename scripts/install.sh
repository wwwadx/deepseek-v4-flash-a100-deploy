#!/usr/bin/env bash
# 从零复现 DeepSeek-V4-Flash-0731 @ 8x A100 80GB PCIe (SM80)。
# 幂等；任一步失败即中止。需要 root。
#
# 可覆盖变量：
#   DATA=/data1        数据盘挂载点（模型/构建树/缓存都在这里）
#   BIND=0.0.0.0       监听地址。0.0.0.0 会把模型暴露给整个网段——见 README 安全章节
#   MAX_JOBS=64        编译并行度
set -euo pipefail

DATA=${DATA:-/data1}
BIND=${BIND:-0.0.0.0}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD=$DATA/dsv4/v2_attempt
MODEL=$DATA/models/DeepSeek-V4-Flash-0731
COMMIT=01ecc8e4f62ca6f3c2add67eede38aa2548204ce
PATCH=$REPO/patches/0001-guard-non-v1-routes.patch

say  () { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die  () { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
need () { command -v "$1" >/dev/null 2>&1 || die "缺少 $1。$2"; }

say "0/8 前置检查"
[ "$(id -u)" -eq 0 ] || die "需要 root"
need git       "apt install git"
need conda     "先装 miniconda: https://docs.conda.io/en/latest/miniconda.html"
need modelscope "pip install -U modelscope"
need findmnt   "apt install util-linux"
need nvidia-smi "未检测到 NVIDIA 驱动"
[ -x /usr/local/cuda-12.4/bin/nvcc ] || die "未找到 CUDA 12.4 (/usr/local/cuda-12.4)。本构建针对 12.4 验证。"
GPUS=$(nvidia-smi -L | wc -l)
[ "$GPUS" -eq 8 ] || echo "  警告: 检测到 $GPUS 张 GPU，本配置针对 8 张 TP=8"
# 数据盘必须真的是挂载点，否则后面会把 166GB 权重写进根分区
mountpoint -q "$DATA" || die "$DATA 不是挂载点。先挂上数据盘（见 system/fstab.snippet）。"
AVAIL=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -ge 350 ] || die "$DATA 可用空间 ${AVAIL}G，权重+构建至少需要 350G"
echo "  ✓ 前置检查通过（GPU=$GPUS, $DATA 可用 ${AVAIL}G）"

say "1/8 IPv4 优先（IPv6 无路由时会让下载假死）"
grep -q '^precedence ::ffff:0:0/96  100' /etc/gai.conf \
  || { cat "$REPO/system/gai.conf.snippet" >> /etc/gai.conf; echo "  已追加（注意：这是机器级改动）"; }

say "2/8 数据盘写入 fstab"
UUID=$(findmnt -no UUID "$DATA")
if grep -q "UUID=$UUID" /etc/fstab; then
  echo "  ✓ 已在 fstab"
else
  # 手动挂载的盘重启即消失，服务会永久起不来 —— 这一步不能只是提示
  printf 'UUID=%s %s ext4 defaults,nofail,x-systemd.device-timeout=120s 0 2\n' "$UUID" "$DATA" >> /etc/fstab
  echo "  已写入: UUID=$UUID $DATA"
  findmnt --verify --verbose >/dev/null || die "fstab 校验失败，请手工检查 /etc/fstab"
fi

say "3/8 Python 3.12 与 gcc12 工具链"
# Ubuntu 22.04 只带 python3.10；vLLM 这套构建需要 3.12
PY=$DATA/dsv4/py312/bin/python3.12
[ -x "$PY" ] || conda create -y --prefix "$DATA/dsv4/py312" -c conda-forge python=3.12
# CUDA 12.4 的 nvcc 拒绝 gcc>13；且运行期 kernel JIT 也需要它
GXX=$DATA/dsv4/gxx_env12/bin/x86_64-conda-linux-gnu-g++
[ -x "$GXX" ] || conda create -y --prefix "$DATA/dsv4/gxx_env12" -c conda-forge gxx_linux-64=12 gcc_linux-64=12
echo "  ✓ python=$($PY -V 2>&1) gcc=$($GXX --version | head -1)"


say "4/8 拉取 vLLM 源码并应用鉴权补丁"
# 按 SHA 取，不按分支名取。上游分支 dsv4-flash-a100 是可变的，已经被 force-push 到
# 别的提交上（2026-08-11 实测 tip=12810046，与本 pin 分叉 ahead1/behind1），
# 此时 `clone --branch` + `checkout $COMMIT` 会失败：reference is not a tree。
# 只有这个 SHA 是权威的。
# 这一步刻意排在下载权重之前——取不到源码就该几秒内失败，而不是先花几小时下 166GB。
mkdir -p "$BUILD/vllm"
cd "$BUILD/vllm"
[ -d .git ] || git init -q
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/haosdent/vllm.git
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch --no-tags origin "$COMMIT" \
  || die "取不到 pinned commit $COMMIT。上游可能已删除该对象；见 VERSIONS.md。"
git checkout -q "$COMMIT" || die "checkout $COMMIT 失败（构建树可能有冲突的本地改动）"
# 三态判断：已应用 / 可应用 / 无法应用。绝不能把失败当成"已在位"——
# 那会静默产出一个 /invocations 无需凭据的服务器。
if git apply -R --check "$PATCH" 2>/dev/null; then
  echo "  ✓ 鉴权补丁已在位"
elif git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"; echo "  ✓ 已应用鉴权补丁"
else
  die "鉴权补丁无法应用（commit 是否为 $COMMIT？）。中止，否则会产出无鉴权的服务。"
fi

say "5/8 下载模型（166.9GB）"
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  MODELSCOPE_CACHE=$DATA/dsv4/cache/modelscope \
    modelscope download --model deepseek-ai/DeepSeek-V4-Flash-0731 \
      --local-dir "$MODEL" --max-workers 8
fi
"$PY" - "$MODEL" <<'PY'
import json,sys,glob,os
m=sys.argv[1]
n=len(glob.glob(os.path.join(m,"*.safetensors")))
t=json.load(open(os.path.join(m,"model.safetensors.index.json")))["metadata"]["total_size"]
assert n==48, f"期望 48 个 shard，实际 {n}"
assert t==166878536440, f"total_size 不匹配: {t}"
print(f"  ✓ 模型校验通过: {n} shards, {t} bytes")
PY

say "6/8 编译 vLLM（SM80，约 24 分钟）"
cd "$BUILD/vllm"                                   # 显式声明，别依赖上一步残留的 cwd
if [ ! -f .venv/.build-complete ]; then
  [ -d .venv ] || "$PY" -m venv --copies .venv     # --copies: 不依赖外部解释器目录
  # 构建缓存同样要避开根分区
  export TMPDIR=$BUILD/cache/tmp PIP_CACHE_DIR=$BUILD/cache/pip \
         UV_CACHE_DIR=$BUILD/cache/uv XDG_CACHE_HOME=$BUILD/cache
  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
  export PATH="$DATA/dsv4/gxx_env12/bin:$PATH"
  export CXX=x86_64-conda-linux-gnu-g++ CC=x86_64-conda-linux-gnu-gcc CUDAHOSTCXX=$GXX
  export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=${MAX_JOBS:-64}
  .venv/bin/pip install -qU uv
  .venv/bin/uv pip install -r requirements/build/cuda.txt -r requirements/build/rust.txt --torch-backend=auto
  .venv/bin/pip install -e . --no-build-isolation
  # torchvision 必须匹配 torch 的 CUDA build，否则 import 即报 nms 缺失
  .venv/bin/uv pip install torchvision --torch-backend=cu129 --no-deps
  touch .venv/.build-complete                      # 哨兵放在最后，避免半截构建被当成完成
fi
# 构建依赖是现解析的，没有 lockfile。核对关键包是否和本部署一致——不一致不中止，
# 但要明说，否则"复现"只是复现了流程，不是复现了那套二进制。
if [ -f "$REPO/VERSIONS.lock" ]; then
  MISMATCH=0
  for pkg in torch torchvision flashinfer-python transformers triton; do
    want=$(grep -iE "^${pkg}==" "$REPO/VERSIONS.lock" | head -1)
    got=$(.venv/bin/pip freeze 2>/dev/null | grep -iE "^${pkg}==" | head -1)
    [ "$want" = "$got" ] || { printf '  警告: %s 与本部署不同（本部署 %s，此处 %s）\n' \
      "$pkg" "${want:-未记录}" "${got:-未安装}"; MISMATCH=1; }
  done
  [ "$MISMATCH" -eq 0 ] && echo "  ✓ 关键依赖版本与 VERSIONS.lock 一致" \
    || echo "  → 完整对照见 $REPO/VERSIONS.lock；性能与正确性结论以本部署版本为准"
fi
echo "  ✓ 构建完成"

say "7/8 API key"
# 若 VLLM_API_KEY 为空，vLLM 根本不装鉴权中间件，所有路由全开且日志无提示
if [ ! -f /etc/deepseek-v4-flash.env ]; then
  umask 077
  printf 'VLLM_API_KEY=%s\n' "dsv4-$("$PY" -c 'import secrets;print(secrets.token_urlsafe(36))')" \
    > /etc/deepseek-v4-flash.env
  chmod 600 /etc/deepseek-v4-flash.env
  echo "  已生成 key 并写入 /etc/deepseek-v4-flash.env (0600)"
  echo "  取用: grep -oP '(?<=^VLLM_API_KEY=).*' /etc/deepseek-v4-flash.env"
fi

say "8/8 安装启动脚本与 systemd 单元"
subst () { sed -e "s#/data1#$DATA#g" -e "s#--host 0.0.0.0#--host $BIND#" "$1"; }
subst "$REPO/deploy/serve_production.sh" > "$BUILD/serve_production.sh"; chmod 750 "$BUILD/serve_production.sh"
subst "$REPO/deploy/bin/dsv4-logrotate.sh" > /usr/local/sbin/dsv4-logrotate.sh; chmod 750 /usr/local/sbin/dsv4-logrotate.sh
MOUNTUNIT=$(systemd-escape -p --suffix=mount "$DATA")
for u in "$REPO"/deploy/systemd/*.service "$REPO"/deploy/systemd/*.timer; do
  subst "$u" | sed "s#data1\.mount#$MOUNTUNIT#g" > "/etc/systemd/system/$(basename "$u")"
  chmod 644 "/etc/systemd/system/$(basename "$u")"
done
systemctl daemon-reload
systemctl enable --now dsv4-logrotate.timer
systemctl enable deepseek-v4-flash.service
systemctl start  deepseek-v4-flash.service

if [ "$BIND" = "0.0.0.0" ]; then
  printf '\n\033[33m警告: 服务监听 0.0.0.0，能访问该端口的任何主机都能用这个模型。\n'
  printf '  API key 是唯一防线，且本机可能没有防火墙。见 README「安全」章节。\n'
  printf '  仅本机访问请用: BIND=127.0.0.1 %s\033[0m\n' "$0"
fi

say "等待就绪（冷启动约 4-5 分钟）"
# 探测地址必须跟随 BIND：BIND 为具体 IP 时 loopback 探不到，会把健康的部署误判为失败
PROBE=$([ "$BIND" = "0.0.0.0" ] && echo 127.0.0.1 || echo "$BIND")
for _ in $(seq 1 90); do
  if curl -sf -m 5 -o /dev/null "http://$PROBE:8000/health" 2>/dev/null; then
    echo "  ✓ 就绪"; exec env HOST="http://$PROBE:8000" "$REPO/scripts/verify.sh"
  fi
  systemctl is-active --quiet deepseek-v4-flash.service || die "服务启动失败，见 journalctl -u deepseek-v4-flash"
  sleep 10
done
die "15 分钟内未就绪，见 $BUILD/serve_final.log"
