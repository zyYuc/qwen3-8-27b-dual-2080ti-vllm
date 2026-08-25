# 性能测试结果与复测方式

这里上传的是实际测试结果，不是推测速度。

## 最终配置的实测数据

文件 2026-08-23-streaming-ab.json 记录 4K、8K、20K 三个输入档位，每档连续串行测试 3 次。请求使用唯一的合成英文上下文，避免 Prefix Cache 影响；输出 128 tokens，流式计时。

### A0：最终配置基线

| 实际输入约 | 平均 TTFT | 平均 Prefill | 平均 Decode |
| --- | ---: | ---: | ---: |
| 5.94K tokens | 3.94 s | 1,509.2 tok/s | 81.4 tok/s |
| 11.87K tokens | 8.16 s | 1,455.2 tok/s | 80.5 tok/s |
| 29.57K tokens | 22.56 s | 1,310.8 tok/s | 75.1 tok/s |

TTFT / 首字时间的严格口径：从 HTTP 请求发出开始，到流式 SSE 第一次收到任一非空 reasoning_content、reasoning 或 content 字符为止。也就是说，模型开始输出 think 内的第一个思考字符就算首字，不等待最终答案正文。

Decode 是从该首个字符到流结束、按 API 返回的 completion_tokens 计算的平均速率。因此 decode 包含 reasoning 与 content 两部分输出。

### 历史 A/B 记录说明

仓库不再把旧的 P3/P4 或 2026-08-21 A/B 数字放入速度基线，因为它们来自非流式请求脚本，TTFT 是按固定预填充速度反推的估算值，并非首个思考字符实测。它们可以说明配置尝试历史，但不能用于首字或 decode 验收。

## 启动验收证据

最终服务启动日志记录：

| 项目 | 实测值 |
| --- | --- |
| 每张卡模型加载 | 14.96 GiB |
| 模型加载耗时 | 约 21.9–26.1 s / TP rank |
| GPU KV Cache 容量 | 216,562 tokens |
| 180K 请求最大并发 | 1.20x |
| GDN prefill | FlashQLA legacy SM70/SM75 kernel 已生效 |

## 如何复测

1. 严格使用 README 固定的 GPU、驱动、vLLM commit、补丁、FlashQLA commit、模板与启动参数。
2. 测试前确认无其他 GPU 任务；GPU0 接显示器时数值可能轻微波动。
3. 每档至少串行运行 3 次，使用唯一 prompt，避免 Prefix Cache 把 prefill 成绩虚高。
4. 测试输出固定为 128 tokens，并记录 prompt_tokens、TTFT、total latency、prefill tok/s、decode tok/s；TTFT 必须按首个 reasoning/content 字符记录。
5. 与上表比较时，允许约 ±10% 波动；超过这个范围先检查 flashqla_legacy、TP=2、NVLink=NV2、FP8 KV 和 MTP。

不同模型权重、显示器占用、温度/功耗墙、显卡改装规格、PCIe/NVLink 状态、驱动与 CUDA 小版本都会改变速度。本仓库给出可比的验收区间，不承诺每台机器得到逐位相同数字。
