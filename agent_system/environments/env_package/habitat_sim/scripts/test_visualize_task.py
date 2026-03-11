"""
测试脚本：加载指定的任务并可视化
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import ray

# 导入环境和可视化函数
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.dataset.collect_class_detect_eval_dataset import visualize_class_detect_task

def load_task_by_id(json_file, task_id):
    """从 JSON 文件中加载指定 task_id 的任务"""
    with open(json_file, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    
    for task in all_tasks:
        if task.get('task_id') == task_id:
            return task
    
    raise ValueError(f"Task with task_id={task_id} not found in {json_file}")

def main():
    # 配置
    json_file = '/data1/tct_data/habitat/eval_data/replicacad_10-class-detect/task_infos.json'
    target_task_id = 77
    output_dir = '/data1/tct_data/habitat/test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"加载任务 ID: {target_task_id}")
    print(f"从文件: {json_file}\n")
    
    # 加载任务
    task_info = load_task_by_id(json_file, target_task_id)
    
    print(f"任务信息:")
    print(f"  Task ID: {task_info['task_id']}")
    print(f"  Task Type: {task_info['task_type']}")
    print(f"  Target Category: {task_info['target_category']}")
    print(f"  Scene ID: {task_info['scene_id']}")
    print(f"  Task Prompt: {task_info['task_prompt']}")
    print(f"  Semantic ID: {task_info.get('semantic_id')}")
    print(f"  Instance ID: {task_info.get('instance_id')}")
    
    # 初始化环境
    print("\n初始化环境...")
    dataset_name = "ReplicaCAD"
    seed = 42
    scenes_size = 10
    max_scene_instance = 20
    max_step_length = 10
    
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    
    # 使用 reset_eval 加载任务
    print("\n加载任务到环境...")
    obs, info = env.reset_eval(sync_info=task_info)
    
    print(f"\n环境已加载:")
    print(f"  观察图像大小: {obs.size}")
    print(f"  Conf Score: {info.get('conf_score', 'N/A')}")
    
    # 获取所有同类物体的 GT（使用新的可见性过滤方法）
    print("\n获取同类物体的 GT（带可见性过滤）...")
    try:
        import time
        start_time = time.time()
        
        _, gt_list = env.get_gt_for_class()
        
        elapsed_time = time.time() - start_time
        print(f"找到 {len(gt_list)} 个可见的同类物体（耗时: {elapsed_time:.2f}秒）")
        
        # 打印每个 GT 的详细信息
        for i, gt_item in enumerate(gt_list):
            print(f"\n  GT {i+1}:")
            print(f"    bbox: {gt_item['bbox_gt']}")
            print(f"    category: {gt_item['category']}")
            print(f"    instance_id: {gt_item['instance_id']}")
            print(f"    visible_pixels: {gt_item.get('visible_pixels', 'N/A')}")
            
            # 计算可见度
            x1, y1, x2, y2 = gt_item['bbox_gt']
            bbox_area = (x2 - x1) * (y2 - y1)
            visible_pixels = gt_item.get('visible_pixels', 0)
            if bbox_area > 0 and visible_pixels > 0:
                visibility_ratio = visible_pixels / bbox_area
                print(f"    visibility: {visibility_ratio*100:.1f}% ({visible_pixels}/{int(bbox_area)} pixels)")
    
    except Exception as e:
        print(f"获取 GT 失败: {e}")
        import traceback
        traceback.print_exc()
        gt_list = []
    
    # 获取预测结果
    pred_list = info.get("pred", [])
    print(f"\n预测结果: {len(pred_list)} 个检测")
    for i, pred in enumerate(pred_list):
        if pred.get('category') != 'unknown':
            print(f"  Pred {i+1}: {pred.get('category')} (score: {pred.get('score', 0):.3f})")
    
    # 可视化
    print("\n生成可视化...")
    vis_img = visualize_class_detect_task(
        obs_pil=obs,
        task_prompt=task_info['task_prompt'],
        gt_list=gt_list,
        pred_list=pred_list,
        task_id=target_task_id
    )
    
    # 保存结果
    output_path = os.path.join(output_dir, f'task_{target_task_id}_visualization.png')
    vis_img.save(output_path)
    print(f"\n✓ 可视化结果已保存到: {output_path}")
    
    # 也保存原始观察图像（无标注）
    raw_output_path = os.path.join(output_dir, f'task_{target_task_id}_raw.png')
    obs.save(raw_output_path)
    print(f"✓ 原始图像已保存到: {raw_output_path}")
    
    # 创建对比图（并排显示）
    print("\n创建对比图...")
    comparison_img = Image.new('RGB', (obs.width * 2, obs.height))
    comparison_img.paste(obs, (0, 0))
    comparison_img.paste(vis_img, (obs.width, 0))
    
    # 添加标题
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((obs.width // 4, 5), "Original", fill=(255, 255, 255), font=font, 
             stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((obs.width + obs.width // 4, 5), "Annotated", fill=(255, 255, 255), font=font,
             stroke_width=2, stroke_fill=(0, 0, 0))
    
    comparison_path = os.path.join(output_dir, f'task_{target_task_id}_comparison.png')
    comparison_img.save(comparison_path)
    print(f"✓ 对比图已保存到: {comparison_path}")
    
    # 打印统计摘要
    print("\n" + "="*70)
    print("统计摘要")
    print("="*70)
    print(f"任务 ID: {target_task_id}")
    print(f"类别: {task_info['target_category']}")
    print(f"场景: {task_info['scene_id']}")
    print(f"GT 数量: {len(gt_list)}")
    print(f"预测数量: {len([p for p in pred_list if p.get('category') != 'unknown'])}")
    
    if gt_list:
        avg_visibility = np.mean([
            gt.get('visible_pixels', 0) / ((gt['bbox_gt'][2] - gt['bbox_gt'][0]) * (gt['bbox_gt'][3] - gt['bbox_gt'][1]))
            for gt in gt_list
        ])
        print(f"平均可见度: {avg_visibility*100:.1f}%")
    
    print("="*70)
    
    # 清理
    env.close()
    print("\n测试完成！")

if __name__ == "__main__":
    # 初始化 ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

