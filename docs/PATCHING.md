# 从干净上游源码构建补丁版环境

以下步骤锁定的是这套服务实测使用的提交。先在测试机验证，再替换生产服务。

## vLLM

~~~bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.27.1
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/vllm-v0.27.1-sm75-qwen3.8.patch
python -m pip install -U pip
python -m pip install -e .
~~~

本次验证环境：Python 3.12.3 + PyTorch 2.13.0+cu130。

## FlashQLA-SM70-SM75

~~~bash
git clone https://github.com/weicj/FlashQLA-SM70-SM75.git
cd FlashQLA-SM70-SM75
git checkout 3ab27d77d8ca01d7a4718903b726add1a8886c0e
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/flashqla-sm70-sm75-local.patch
python -m pip install -e .
~~~

运行服务前，确保 FLASHQLA_PATH 指向该目录；启动脚本会把它加入 PYTHONPATH。

## FlashInfer

本次运行环境使用 flashinfer-python==0.6.16.post3。SM75 对 FlashInfer/vLLM 版本较敏感，升级 FlashInfer、vLLM、CUDA、驱动或模型后，都要重新测试短文本、长上下文、工具调用和显存上限。
