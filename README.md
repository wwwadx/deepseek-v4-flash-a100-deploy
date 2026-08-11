# DeepSeek-V4-Flash-0731 on 8× A100 80GB PCIe

*README 为中文；完整技术长文 `docs/` 为英文。*

在**没有** Hopper/Blackwell（无原生 FP8/FP4）的 Ampere PCIe 机器上，把 284B MoE
模型跑到生产可用：~40 tok/s 单流、32 并发约 1,070 tok/s、1M 上下文、带工具调用。

本仓库包含**从零复现该部署所需的全部内容**。

## 快速开始

```bash
sudo scripts/install.sh     # 幂等；从下模型到起服务
sudo scripts/verify.sh      # 自检（install.sh 结束时会自动调用）：就绪 / 安全 / 功能 / 容量
```

冷启动约 4–5 分钟。`Type=simple` 会在进程刚拉起时就报 active，**判断就绪要轮询
`/health`，不要看 `systemctl is-active`**。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 完整实战文章：踩过的每个坑与原因 |
| `VERSIONS.md` | 锁定的复现坐标（commit / 驱动 / 依赖版本） |
| `deploy/serve_production.sh` | 真正的启动脚本（20 个 export 每一条都是必需的） |
| `deploy/systemd/` | service + 日志轮转 timer |
| `deploy/env/*.example` | key 文件模板（真实 key 永不入库） |
| `patches/` | **必须应用**的 vLLM 补丁 |
| `system/` | fstab 与 gai.conf 片段 |
| `scripts/` | install / verify |

## 三个最容易致命的点

1. **`VLLM_USE_BREAKABLE_CUDAGRAPH=0` 必须显式导出。**
   该 fork 在变量*未设置*时会自动置 1，而 1 正是 batch>1 静默输出损坏的成因。
   漏掉它 = 拿到损坏配置，且日志里看不出问题。

2. **`--api-key` 保护不了推理接口。** vLLM 只守卫 `/v1`、`/v2`、`/inference`；
   `POST /invocations` 走同一个 chat/completions handler 却**完全不需要凭据**。
   `patches/0001` 修的就是这个。另外：**若 `VLLM_API_KEY` 为空，vLLM 干脆不装
   鉴权中间件，所有路由全开**，且日志无任何提示。

3. **数据盘必须写进 `/etc/fstab`。** 手动挂载的盘重启即消失，而 venv、权重、
   工具链全在上面 —— 服务会永久起不来。unit 里的 `RequiresMountsFor=` 救不了
   一个根本没在 fstab 里的挂载点。

## 长上下文并发的真相

`--max-model-len` 是**每请求的 KV 预留**，不只是上限。1M 配置下实测并发上限仅
**2.81×**；把 `--max-num-batched-tokens` 从 2048 提到 16384 会让该值从 8.86×
掉到 2.81×（−68%）。若你的负载是并行长上下文，把 `--max-model-len` 调到实际
需要的长度，收益远大于任何其他调参。

**验证并发一定要看日志里的 `Running: N reqs`，不能用墙钟时间推断** —— 我们最初
就是这样把「4 路 893K 并发」误报了，实际是串行排队。

## 部署前必读的几点

- **数据盘写进 fstab。** 见 `system/fstab.snippet`，填入本机 UUID。手动挂载的盘
  重启即消失。
- **给输出长度设上限。** 服务端没有输出 token 上限，请求若不带 `max_tokens`，
  会被允许生成接近 `--max-model-len` 的长度。请在客户端或反向代理侧限制。
- **确认 GPU 驱动与内核的绑定关系。** 若 `nvidia.ko` 只为当前内核编译且未启用
  DKMS，内核升级后重启会导致 GPU 不可用。部署前用
  `ls /lib/modules/*/updates/dkms/nvidia.ko` 自查，必要时 hold 住内核包。
- **venv 若由别处的解释器创建，它只是个软链壳。** 确认解释器本体所在目录不会被
  当成垃圾清理掉；更稳妥的做法是 `python -m venv --copies`。
- **接一套监控。** `/metrics` 默认暴露但无人抓取。

## 复现坐标

见 `VERSIONS.md`。硬件为 8× A100 80GB PCIe（SM80，无 NVLink，双 NUMA island）。
其他 Ampere 卡只要总显存与拓扑相当应可照搬，但未经验证。
