# 运行环境锁定清单

这是 2026-08-24 正在运行服务的关键 Python/CUDA 包版本。优先保持这些版本不变，直到先跑通基准。

~~~text
Python                 3.12.3
NVIDIA Driver          580.159.03
Driver CUDA Runtime    13.0
torch                  2.13.0+cu130
vllm                   0.27.1
transformers           5.15.1
flashinfer-python      0.6.16.post3
triton                 3.7.1
numpy                  2.3.5
tokenizers             0.22.2
safetensors            0.8.0
xgrammar               0.2.3
nvidia-nccl-cu13       2.29.7
nvidia-cudnn-cu13      9.20.0.48
nvidia-cuda-runtime    13.0.96
nvidia-cublas          13.1.1.3
nvidia-cusparselt-cu13 0.8.1
~~~

系统 nvcc 是否在 PATH 并不是 vLLM 运行的唯一判断条件；本环境依靠 PyTorch 的 CUDA 运行时。若从源码编译 vLLM、FlashInfer 或 FlashQLA，仍需要准备匹配的 CUDA 编译工具链。

