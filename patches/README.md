# patches/

## 0001-guard-non-v1-routes.patch

**必须应用。** 不应用会得到一个 `POST /invocations` 无需任何凭据即可推理的服务。

### 问题

vLLM 的鉴权中间件只守卫三个前缀：

```python
GUARDED_PREFIX = ("/v1", "/v2", "/inference")
```

而 `POST /invocations`（SageMaker 兼容路由）分发到与 `/v1/chat/completions`
**完全相同**的 handler，却落在这三个前缀之外 —— `--api-key` 对它无效。
`/tokenize`、`/detokenize`、`/generative_scoring` 同样开放，
状态变更接口 `/scale_elastic_ep` 也是。

### 本补丁

把上述路由加入 `GUARDED_PREFIX`。纯增量修改；`/health`、`/ping`、`/metrics`、
`/version` 与文档路由**刻意保持开放**，以免破坏监控。

### 适用范围

| | |
|---|---|
| 上游 | vLLM (Apache-2.0) |
| 经由 | `haosdent/vllm`（分支 `dsv4-flash-a100` 可变，已 force-push 过，勿按分支名取）|
| 锁定 commit | `01ecc8e4f62ca6f3c2add67eede38aa2548204ce`（用 `git fetch origin <SHA>`）|
| 目标文件 | `vllm/entrypoints/serve/utils/server_utils.py` |

`scripts/install.sh` 会自动三态判断（已应用 / 可应用 / 无法应用），
**无法应用时直接中止**，绝不静默产出无鉴权的服务。

### 手工验证

```bash
# 未带 key 应返回 401；若返回 200，说明补丁没生效
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":3}'
```
