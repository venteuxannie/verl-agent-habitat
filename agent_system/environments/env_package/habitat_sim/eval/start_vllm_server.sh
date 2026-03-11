#!/bin/bash
# vLLM 服务启动脚本
# 用法: ./start_vllm_server.sh <model_path_or_name> [port] [tensor_parallel_size]
#
# 示例:
#   ./start_vllm_server.sh OpenGVLab/InternVL2_5-4B
#   ./start_vllm_server.sh /data/tct/models/RL/InternVL3_5-2B-unified-tasks-rl-HM3D 8000 1
#   ./start_vllm_server.sh Qwen/Qwen2-VL-7B-Instruct 8001 2
#
# 支持的模型示例:
#   - OpenGVLab/InternVL2_5-4B
#   - OpenGVLab/InternVL2_5-8B
#   - OpenGVLab/InternVL3-8B
#   - Qwen/Qwen2-VL-7B-Instruct
#   - Qwen/Qwen2.5-VL-7B-Instruct
#   - liuhaotian/llava-v1.6-mistral-7b
#   - 本地模型路径

set -e

# ==========================================
# 环境变量设置
# ==========================================
# 修改 Hugging Face 模型缓存地址 (默认是 ~/.cache/huggingface)
export HF_HOME="/data/tct/hf_cache"
# 如果你需要使用镜像源加速下载（针对国内环境）
export HF_ENDPOINT="https://hf-mirror.com"
# ==========================================

# 默认参数
MODEL_PATH=${1:-"OpenGVLab/InternVL2_5-4B"}
PORT=${2:-8000}
TENSOR_PARALLEL_SIZE=${3:-1}

# 检查并自动激活 conda 环境
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "verl-internvl" ]]; then
    echo "🔄 正在尝试自动激活 conda 环境: verl-internvl..."
    conda activate verl-internvl
    echo "✅ 环境激活成功: ${CONDA_DEFAULT_ENV}"
else
    echo "✅ 环境已激活: ${CONDA_DEFAULT_ENV}"
fi

echo "=========================================="
echo "🚀 启动 vLLM OpenAI 兼容 API 服务"
echo "=========================================="
echo "模型: ${MODEL_PATH}"
echo "端口: ${PORT}"
echo "张量并行: ${TENSOR_PARALLEL_SIZE}"
echo "=========================================="
echo ""
echo "服务启动后，可以使用以下方式调用:"
echo ""
echo "  # Python 调用"
echo "  from openai import OpenAI"
echo "  client = OpenAI(base_url='http://localhost:${PORT}/v1', api_key='empty')"
echo ""
echo "  # eval.py 调用"
echo "  python eval.py --mode model --backend api \\"
echo "      --api_url http://localhost:${PORT}/v1 \\"
echo "      --model_name ${MODEL_PATH##*/} \\"
echo "      --exp_name HM3D-grounding-test"
echo ""
echo "=========================================="
echo ""

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
