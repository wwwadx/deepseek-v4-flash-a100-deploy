# 复现坐标（锁定版本）

生成时间：2026-08-11T20:49:56+08:00

## 硬件 / 驱动
- GPU: 8x NVIDIA A100 80GB PCIe (SM80)，无 NVLink，双 4-GPU NUMA island
- 驱动: NVIDIA >= 550（需支持 CUDA 12.4；勿升到需要 CUDA 13 的组合）
- CUDA toolkit: 12.4（系统 nvcc，编译 CUDA 源码用）
- PyTorch wheel: cu129（与系统 12.4 并存，运行期加载 libcudart.so.12）
- OS: Ubuntu 22.04 LTS（5.15 系内核）

## 模型
- deepseek-ai/DeepSeek-V4-Flash-0731
- 大小: 166,878,536,440 bytes / 48 shards
- 来源: ModelScope
- 本地路径: $DATA/models/DeepSeek-V4-Flash-0731（$DATA 默认 /data1，可覆盖）

## 推理引擎（fork，必须从源码编译）
- repo:   https://github.com/haosdent/vllm.git
- branch: dsv4-flash-a100
- commit: 01ecc8e4f62ca6f3c2add67eede38aa2548204ce
- 本地补丁: patches/0001-guard-non-v1-routes.patch（修复鉴权绕过，必须应用）
- 编译目标: TORCH_CUDA_ARCH_LIST=8.0

## 关键 Python 依赖（编译后实际锁定值）
- flashinfer-python                        0.6.15.post1
- torch                                    2.13.0+cu129
- torchvision                              0.28.0+cu129
- transformers                             5.14.1
- triton                                   3.7.1
- vllm  0.1.dev19422+g01ecc8e4f.cu124（源码可编辑安装）
- xgrammar                                 0.2.3

## 工具链
- gcc/g++ 12（CUDA 12.4 的 nvcc 拒绝 >13）
- 安装: conda create --prefix $DATA/dsv4/gxx_env12 -c conda-forge gxx_linux-64=12 gcc_linux-64=12
- 运行期也需要（kernel JIT），不只编译期

## 前置依赖（install.sh 会检查，缺失即中止）

| | |
|---|---|
| Python | 3.12（Ubuntu 22.04 自带 3.10，不可用；install.sh 会用 conda 装 3.12） |
| conda | 需预先安装（miniconda 即可），用于装 python3.12 与 gcc12 |
| modelscope | `pip install -U modelscope`，用于下载权重 |
| gcc/g++ | 12（CUDA 12.4 的 nvcc 拒绝 >13）；**运行期也需要**，kernel 会 JIT |
| git / util-linux | findmnt、mountpoint |
| 数据盘可用空间 | >= 350G（权重 166.9G + 构建树 + 缓存） |
| 编译并行度 | `MAX_JOBS`，默认 64；112 核约 24 分钟 |

> 构建时 `build_rust` 会因缺少 rustc 而跳过并打印警告 —— 本配置下属预期，不影响使用。
