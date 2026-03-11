#!/bin/bash
cd /data/tct/ActivePerception/agent_system/environments/env_package/habitat_sim/eval

GPU_ID=7
MODEL_PATH="/data/tct/models/SFT/InternVL3_5-2B-unified-tasks-sft-replicacad"
SCENE_NAME="replicacad_10-any-500-seen"
EXP_SUFFIX="sft_only-0128"

# 根据场景名称确定数据集前缀
if [[ "$SCENE_NAME" == *"HM3D"* ]] || [[ "$SCENE_NAME" == *"hm3d"* ]]; then
    DATASET_PREFIX="HM3D"
else
    DATASET_PREFIX="ReplicaCAD"
fi

# 构建实验名称后缀
if [[ -n "$EXP_SUFFIX" ]]; then
    FULL_SUFFIX="-${EXP_SUFFIX}"
else
    FULL_SUFFIX=""
fi

TASKS=("grounding" "segment" "3dbox")

# 同时启动3种任务
for task in "${TASKS[@]}"; do
    exp_name="${DATASET_PREFIX}-${task}${FULL_SUFFIX}"
    echo "[GPU $GPU_ID] Launching: $exp_name"

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_path "$MODEL_PATH" \
        --input_filename "$SCENE_NAME" \
        --exp_name "$exp_name" &
done

echo "查看进程: ps aux | grep eval.py"
echo "结束进程: pkill -f \"python eval.py\""
#pkill -f "python eval.py"