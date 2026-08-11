#!/usr/bin/env bash
# 部署后自检：就绪 / 安全 / 功能 / 容量。只读，不修改任何东西。
#   DATA=/data1  数据盘挂载点   HOST=http://localhost:8000  服务地址
set -uo pipefail
DATA=${DATA:-/data1}
HOST=${HOST:-http://localhost:8000}
LOG=$DATA/dsv4/v2_attempt/serve_final.log
KEYFILE=/etc/deepseek-v4-flash.env
# 本机 http_proxy 常会劫持 localhost（代理返回 503），一律剥掉
CURL="env -u http_proxy -u https_proxy curl -s"
pass=0; fail=0

chk () { # 描述 期望 实际
  if [ "$2" = "$3" ]; then printf "  [PASS] %s = %s\n" "$1" "$3"; pass=$((pass+1))
  else printf "  [FAIL] %s : 期望=%s 实际=%s\n" "$1" "$2" "$3"; fail=$((fail+1)); fi
}
code () { $CURL -o /dev/null -w '%{http_code}' -m "${2:-10}" -X "${3:-GET}" "$1" \
            ${4:+-H 'Content-Type: application/json'} ${4:+-d "$4"} 2>/dev/null; }

echo "== 就绪 =="
READY=$($CURL -o /dev/null -w '%{http_code}' -m 10 "$HOST/health")
chk "/health 可达" 200 "$READY"
[ "$READY" = "200" ] || { echo "  服务未就绪（冷启动约 4-5 分钟），后续检查略过"; exit 1; }

echo "== 安全: 鉴权须覆盖所有推理/分词/变更路由 =="
for p in /invocations /tokenize /detokenize /generative_scoring /scale_elastic_ep /load /v1/chat/completions; do
  chk "$p 无 key" 401 "$($CURL -o /dev/null -w '%{http_code}' -m 10 -X POST "$HOST$p" -H 'Content-Type: application/json' -d '{}')"
done
echo "== 安全: 监控端点须保持开放 =="
for p in /health /metrics; do
  chk "$p 无 key" 200 "$($CURL -o /dev/null -w '%{http_code}' -m 10 "$HOST$p")"
done

echo "== 配置 =="
chk "key 不在进程 argv" 0 "$(ps aux | grep '[v]llm serve' | grep -c 'api-key')"
if [ -f "$LOG" ]; then
  chk "无未知环境变量告警" 0 "$(tail -c 300000 "$LOG" | grep -c 'Unknown vLLM environment variable')"
else
  printf "  [FAIL] 找不到日志 %s\n" "$LOG"; fail=$((fail+1))
fi
MAINPID=$(systemctl show deepseek-v4-flash -p MainPID --value 2>/dev/null)
if [ -n "$MAINPID" ] && [ "$MAINPID" != 0 ]; then
  chk "breakable cudagraph 显式关闭" 1 \
      "$(tr '\0' '\n' < /proc/$MAINPID/environ 2>/dev/null | grep -c '^VLLM_USE_BREAKABLE_CUDAGRAPH=0')"
fi
chk "unit 有数据盘挂载依赖" "$DATA" "$(systemctl show deepseek-v4-flash -p RequiresMountsFor --value)"

echo "== 功能 =="
KEY=$(grep -oP '(?<=^VLLM_API_KEY=).*' "$KEYFILE" 2>/dev/null || echo "")
if [ -n "$KEY" ]; then
  # 走临时 curl 配置，避免 key 出现在 ps 里（本项目正是为此把 key 移出 argv 的）
  CFG=$(umask 077; mktemp); trap 'rm -f "$CFG"' EXIT
  printf 'header = "Authorization: Bearer %s"\n' "$KEY" > "$CFG"
  post () { $CURL -m 90 --config "$CFG" -H 'Content-Type: application/json' -d "$1" "$HOST/v1/chat/completions"; }
  chk "算术正确 (17*24=408)" True "$(post '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is 17 * 24?"}],"max_tokens":30,"temperature":0}' | python3 -c "import json,sys;print('408' in json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"
  chk "工具调用可用" True "$(post '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Weather in Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":80,"temperature":0}' | python3 -c "import json,sys;print(bool(json.load(sys.stdin)['choices'][0]['message'].get('tool_calls')))" 2>/dev/null)"
else
  echo "  [WARN] 读不到 $KEYFILE，跳过功能检查"
fi

echo "== 容量 =="
[ -f "$LOG" ] && grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -1 | sed 's/^/  /'
echo "  提示: 长上下文并发看日志里的 'Running: N reqs'，不要用墙钟时间推断"

echo; echo "结果: $pass 通过, $fail 失败"; [ "$fail" -eq 0 ]
