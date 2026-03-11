import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import random
import argparse
import json
from functools import partial

import ray
from utils.habitat_envs import CreateHabitatEnv

def reshape_bbox(bbox, original_size=(1000, 800), new_size=(800, 640)):
    """调整bbox尺寸"""
    if bbox is None:
        return (0, 0, 0, 0)
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    x1, y1, x2, y2 = bbox
    x1_new = x1 * scale_x
    y1_new = y1 * scale_y
    x2_new = x2 * scale_x
    y2_new = y2 * scale_y
    return (x1_new, y1_new, x2_new, y2_new)

def calculate_iou(box2_gt, box1_pred):
    """计算IoU"""
    if box1_pred is None or box2_gt is None:
        return 0.0
    box1 = reshape_bbox(box1_pred)
    box2 = (box2_gt[0], box2_gt[2], box2_gt[1], box2_gt[3])
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    intersection_width = max(0, x2_inter - x1_inter)
    intersection_height = max(0, y2_inter - y1_inter)
    intersection_area = intersection_width * intersection_height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def run_action_sequence(action_sequence, task_id=0):
    """
    执行动作序列测试
    
    Args:
        action_sequence: 动作列表，如 ['move_forward', 'turn_left', 'turn_right']
        task_id: 任务ID (0-99)
    """
    
    # 环境配置
    dataset_name = "ReplicaCAD"
    seed = 0
    scenes_size = 10
    max_scene_instance = 10
    max_step_length = len(action_sequence)
    
    # 创建环境
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    print(f"Habitat环境创建成功")
    
    # 动作映射
    env_actions = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]
    
    # 验证并转换动作
    action_indices = []
    for action_name in action_sequence:
        if action_name in env_actions:
            action_indices.append(env_actions.index(action_name))
        else:
            print(f"警告: 未知动作 '{action_name}'，使用随机动作替代")
            action_indices.append(random.randint(0, len(env_actions) - 1))
    
    # 加载任务信息
    input_filename = '/data1/tct_data/habitat/eval_data/replicacad_10_iou-small-bbox-2/task_infos.json'
    with open(input_filename, 'r', encoding='utf-8') as f:
        all_task_infos = json.load(f)
    
    # 获取任务信息并重置环境
    task_info = all_task_infos[task_id]
    obs, info = env.reset_eval(sync_info=task_info)

    for step, action_idx in enumerate(action_indices):        
        obs, reward, done, info = env.step(action_idx, 1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Habitat环境动作序列测试工具")
    parser.add_argument('--action_sequence', type=str, default="look_down",
                        help='逗号分隔的动作序列，如 "move_forward,turn_left,turn_right"')
    parser.add_argument('--task_id', type=int, default=18,
                        help='任务ID (0-99)')
    
    args = parser.parse_args()
    
    # 解析动作序列
    action_sequence = [action.strip() for action in args.action_sequence.split(',')]
    
    print("支持的动作: move_forward, turn_left, turn_right, look_up, look_down, stop")
    
    # 运行测试
    run_action_sequence(action_sequence, args.task_id)


if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init(_temp_dir="/data/tct/verl-agent/tmp")
    
    main()
