# 加速组件、固定版本与引用

本仓库提供同硬件可复现的部署配置、补丁和模板，不包含模型权重。

| 组件 | 本次工作版本 | 用途 | 上游 / 引用 |
| --- | --- | --- | --- |
| vLLM | v0.27.1 / 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac | API、TP、KV Cache、MTP、CUDA Graph | https://github.com/vllm-project/vllm |
| PyTorch | 2.13.0+cu130 | CUDA 运行时与张量计算 | https://github.com/pytorch/pytorch |
| Transformers | 5.15.1 | 模型与 Tokenizer 配置 | https://github.com/huggingface/transformers |
| FlashInfer | 0.6.16 | SM75 attention / decode 后端 | https://github.com/flashinfer-ai/flashinfer |
| FlashQLA | 3ab27d77d8ca01d7a4718903b726add1a8886c0e | Qwen GDN/linear-attention 的 SM70/SM75 legacy prefill | https://github.com/weicj/FlashQLA-SM70-SM75 |
| NCCL | 2.29.7 | 双卡 Tensor Parallel 通信 | https://github.com/NVIDIA/nccl |

## 仓库内提供的本地改动

- patches/vllm-v0.27.1-sm75-qwen3.8.patch：基于 vLLM v0.27.1 的工作树补丁，涵盖 flashqla_legacy GDN prefill、Qwen3.5 MTP 兼容、reasoning 预算、SM75 FlashInfer/采样兼容和 GPU runner 调整。
- patches/flashqla-sm70-sm75-local.patch：基于固定 FlashQLA commit 的本地导出/SM legacy 调整。
- scripts/apply_gdn_flashqla_legacy.py：当 git apply 因上游小版本差异无法套用时，用于补充 GDN legacy backend 的辅助脚本。

补丁来自已验证服务的工作树，不是 vLLM、FlashInfer 或 FlashQLA 的官方发布包。升级任一上游组件后必须重新验证。

## Chat template

templates/qwen3.8-froggeric-v22.3.jinja 是当前实际使用的修复模板源文件，模板版本字段为 qwen3.8-froggeric-v22.3。

- 用途：Qwen3 消息格式、thinking 开关、XML/JSON 工具调用、多轮 tool response、图像/视频 token 拼装。
- 启动参数：--chat-template templates/qwen3.8-froggeric-v22.3.jinja --chat-template-content-format string
- 模型权重、Tokenizer、上游模型许可证仍以模型发布方许可证为准。

