# Qwen3.8-27B：双 RTX 2080 Ti 22GB + NVLink 的 vLLM 抄作业配置

这是一个独立的开源部署项目，面向 2 张魔改 RTX 2080 Ti 22GB、并且两卡之间已连接双 NVLink 的用户。

目标是把一套正在运行的 Qwen3.8-27B 长上下文配置完整公开：硬件、驱动、加速路径、补丁、Jinja 模板、环境变量、systemd 和完整加载参数都在这里。

适合：单机双卡、单请求优先、180K 上下文、个人/小团队 API、长文档与代码任务。

不包含：模型权重、API Key、内网地址、个人目录、SSH 或隧道配置。

## 已验证环境

| 分类 | 参数 |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti 22GB（22,528 MiB / 卡） |
| GPU 架构 | Turing / SM75 / Compute Capability 7.5 |
| GPU 互联 | NV2：每张卡 2 条 NVLink；单条实测约 25.781 GB/s |
| CPU | AMD Ryzen 7 5700X，8 核 16 线程 |
| 内存 | 32 GiB |
| OS / Kernel | Ubuntu 24.04.4 LTS / Linux 6.17.0-29-generic |
| NVIDIA Driver | 580.159.03 |
| CUDA Runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| vLLM | 0.27.1，上游 v0.27.1 commit 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac + 本仓库 patch |
| Transformers / Triton | 5.15.1 / 3.7.1 |
| FlashInfer | 0.6.16 |
| NCCL | 2.29.7 |

## 这套配置的目标

- TP=2：两张卡共同加载 Qwen3.8-27B。
- FP8 权重 + fp8_e4m3 KV Cache：把更多显存留给上下文。
- max-model-len=180000：服务最大上下文 180K；启动日志可用 KV Cache 约 195K tokens。
- max-num-seqs=1：优先长上下文单请求，不按高并发路线配置。
- Prefix Cache + Chunked Prefill：改善固定系统提示词和超长输入。
- MTP=3 + PIECEWISE CUDA Graph：降低部分解码与固定形状调度开销。
- flashqla_legacy：为 SM70/SM75 的 Qwen GDN prefill 提供兼容加速路径。
- Qwen3 thinking、XML tool calling、修复版 chat template：全部包含在启动配置中。

## 目录

~~~text
config/       环境变量样例
docs/         打补丁、加速组件和引用说明
patches/      已验证工作树导出的 vLLM / FlashQLA patch
scripts/      启动、硬件检查、FlashInfer 检查、GDN 辅助脚本
systemd/      常驻服务模板
templates/    qwen3.8-froggeric-v22.3 Jinja 模板源文件
~~~

## 快速开始

### 1. 验证硬件

~~~bash
bash scripts/verify_hardware.sh
~~~

关键拓扑应包含：

~~~text
GPU0  GPU1
GPU0   X   NV2
GPU1  NV2   X
~~~

没有 NV2 也可能运行，但双卡 TP 的通信条件与本配置不同。

### 2. 获取上游源码并套用补丁

按 docs/PATCHING.md 锁定 vLLM、FlashQLA 和 FlashInfer 版本，并应用本仓库 patch。

### 3. 填写你的路径

~~~bash
cp config/vllm.env.example .env
~~~

至少修改：

~~~bash
MODEL_PATH=/你的/Qwen3.8-27B-FP8/模型目录
VLLM_PYTHON=/你的/venv/bin/python
FLASHQLA_PATH=/你的/FlashQLA-SM70-SM75
~~~

CHAT_TEMPLATE 默认指向本仓库内的 templates/qwen3.8-froggeric-v22.3.jinja。

### 4. 启动

~~~bash
bash scripts/run_qwen3.8_27b_sm75.sh
~~~

## 跑通后的性能验收

这一步是“AI 能否真的帮你跑到相近速度”的关键，而不是只看到服务能启动。这里的首字时间严格按模型流式输出的第一个思考字符或答案字符计算；Qwen 开始输出 think 中第一个字符，就视为首字。

~~~bash
python benchmarks/collect_streaming_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen-local \
  --output my-benchmark-result.json
~~~

测试方法、真实原始结果、A/B 数据和验收范围都在 benchmarks/README.md。最终配置的基线是：

| 实际输入约 | 平均 TTFT | 平均 Prefill | 平均 Decode |
| --- | ---: | ---: | ---: |
| 5.94K tokens | 3.94 s | 1,509.2 tok/s | 81.4 tok/s |
| 11.87K tokens | 8.16 s | 1,455.2 tok/s | 80.5 tok/s |
| 29.57K tokens | 22.56 s | 1,310.8 tok/s | 75.1 tok/s |

复现时应先锁定 docs/environment-lock.md，再按 benchmarks/README.md 的方法测试。相同硬件的合理验收范围是约 ±10%；显著偏离时按顺序检查：NVLink 是否为 NV2、TP 是否为 2、补丁是否生效、FlashQLA legacy 是否被日志选中、是否使用 FP8 KV、是否有其他 GPU 占用。

## 完整加载参数与用途

| 参数 | 当前值 | 用途 |
| --- | --- | --- |
| --dtype | half | 运行时 FP16 计算 dtype。 |
| --tensor-parallel-size | 2 | 两张 GPU 做 Tensor Parallel。 |
| --device-ids | 0,1 | 明确使用 GPU 0、1。 |
| --quantization | fp8 | FP8 权重加载。 |
| --kv-cache-dtype | fp8_e4m3 | 用 FP8 E4M3 存 KV Cache，降低 KV 显存。 |
| --max-model-len | 180000 | 单请求上下文上限。 |
| --gpu-memory-utilization | 0.93 | vLLM 目标使用每卡 93% 显存。 |
| --kv-cache-memory-bytes | 4G | 显式限制 KV Cache 显存预算。 |
| --max-num-seqs | 1 | 单并发、长上下文优先。 |
| --max-num-batched-tokens | 4096 | 限制单轮调度 token，平衡峰值显存与延迟。 |
| --enable-prefix-caching | 开启 | 缓存重复系统提示词和前缀。 |
| --enable-chunked-prefill | 开启 | 长输入分块 prefill。 |
| --no-async-scheduling | 开启 | 关闭异步调度，保持这套兼容路径。 |
| --speculative-config | mtp / 3 | 每步最多预测 3 个 token。 |
| --compilation-config | PIECEWISE / [4] | 只捕获 size=4 CUDA Graph。 |
| --additional-config | flashqla_legacy | SM75 GDN prefill 后端。 |
| --reasoning-parser | qwen3 | Qwen3 thinking 输出解析。 |
| --tool-call-parser | qwen3_xml | Qwen3 XML tool calling 解析。 |
| --chat-template | qwen3.8-froggeric-v22.3.jinja | 使用本仓库修复模板。 |
| --override-generation-config | T=0.6，top_p=0.95，top_k=20，repetition=1.06 | 默认采样参数。 |

## 环境变量与加速路径

| 环境变量 | 值 | 作用 |
| --- | --- | --- |
| OMP_NUM_THREADS | 8 | 与 5700X 物理核心数匹配。 |
| VLLM_USE_DEEP_GEMM | 0 | 关闭此配置未使用的 DeepGEMM 路径。 |
| VLLM_USE_FLASHINFER_SAMPLER | 0 | 关闭 FlashInfer top-k/top-p sampler。 |
| VLLM_QWOPUS_MTP_BF16_DRAFT | 1 | Qwen3.5 MTP draft 层兼容设置。 |
| VLLM_SM75_SPEC_SYNC_MODE | safe | SM75 speculative decoding 保守同步模式。 |
| VLLM_USE_V2_MODEL_RUNNER | 1 | 使用 V2 model runner。 |
| PYTHONPATH | FLASHQLA_PATH | 让 vLLM 能导入 FlashQLA SM75 GDN backend。 |

完整来源、固定 commit、补丁边界和引用见 docs/ACCELERATION_AND_ATTRIBUTION.md。完整环境锁定清单见 docs/environment-lock.md。

## systemd 常驻服务

1. 把仓库放到 /opt/qwen3-8-27b-dual-2080ti-vllm，或修改 unit 中路径。
2. 把 config/vllm.env.example 复制为 /etc/qwen3.8-vllm.env 并填写路径。
3. 修改 systemd/qwen3.8-27b-vllm.service.example 中的 Linux 用户。
4. 安装并启动：

~~~bash
sudo cp systemd/qwen3.8-27b-vllm.service.example /etc/systemd/system/qwen3.8-27b-vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3.8-27b-vllm
sudo systemctl status qwen3.8-27b-vllm --no-pager
~~~

## 重要说明

- 模型权重不在本仓库内。请从拥有相应许可的来源获取模型。
- 这是 SM75 / 2080 Ti 的特化配置，不能把它当作 H100、4090、A100 或无 NVLink 双卡的通用最优参数。
- FP8 权重和 FP8 KV Cache 需要自行做业务精度回归，尤其是长上下文、数学、代码和工具调用。
- 本仓库只公开部署配置和已导出的本地补丁；上游组件遵循各自许可证。

## 引用与致谢

感谢并请引用：vLLM、PyTorch、Hugging Face Transformers、FlashInfer、FlashQLA-SM70-SM75、NCCL。详细链接和 commit 在 docs/ACCELERATION_AND_ATTRIBUTION.md。
