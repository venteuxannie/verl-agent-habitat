import os
import sys
import json
import numpy as np
import cv2
from tqdm import tqdm
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import get_dataset_subfolder

import ray

def convert_numpy_to_list(obj):
    """递归地将 numpy 数组和 tuple 转换为 Python list，以便 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def get_scene_path(subfolders, dataset_name, eval_id=0):
    """Constructs the full path to a habitat scene file."""
    data_path = "/data1/tct_data/habitat/data"
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d")
        scene = subfolders[eval_id % len(subfolders)]
        id = scene.split('/')[1].split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)
    elif dataset_name == "ReplicaCAD":
        scene_name = subfolders[eval_id % len(subfolders)]
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def visualize_class_detect_task(obs_pil, task_prompt, gt_list, pred_list, task_id):
    """
    可视化 class-detect 任务：在观测图像上标注所有 GT 和 Pred 检测框
    
    Args:
        obs_pil: PIL Image，原始观测图像
        task_prompt: str，任务类别名称
        gt_list: list，GT 检测列表，每个元素包含 bbox_gt, mask_gt, category, instance_id
        pred_list: list，预测检测列表，每个元素包含 bbox, score, category
        task_id: int，任务ID
        
    Returns:
        PIL Image，标注后的图像
    """
    # 转换为 numpy 数组
    img_np = np.array(obs_pil)
    H, W = img_np.shape[:2]
    
    # 创建叠加层用于显示 masks
    overlay = img_np.copy()
    
    # 颜色列表（用于区分不同的实例）
    gt_colors = [
        [255, 0, 0],      # 红色
        [255, 100, 0],    # 橙红
        [200, 0, 50],     # 深红
        [255, 50, 100],   # 粉红
    ]
    
    pred_colors = [
        [0, 255, 0],      # 绿色
        [0, 255, 100],    # 青绿
        [100, 255, 0],    # 黄绿
        [0, 200, 100],    # 深绿
    ]
    
    # 1. 叠加所有 GT masks（红色系半透明）
    if gt_list:
        for i, gt_item in enumerate(gt_list):
            gt_mask_rle = gt_item.get("mask_gt")
            if gt_mask_rle is not None:
                try:
                    gt_mask = mask_utils.decode(gt_mask_rle)
                    color = gt_colors[i % len(gt_colors)]
                    overlay[gt_mask > 0] = color
                except Exception as e:
                    print(f"  Warning: Failed to decode GT mask {i}: {e}")
    
    # 2. 叠加所有 Pred masks（绿色系半透明）
    if pred_list:
        for i, pred_item in enumerate(pred_list):
            pred_mask_rle = pred_item.get("mask_gt")  # 注意：这里用的是 pred 的 mask
            if pred_mask_rle is not None:
                try:
                    pred_mask = mask_utils.decode(pred_mask_rle)
                    color = pred_colors[i % len(pred_colors)]
                    overlay[pred_mask > 0] = color
                except Exception as e:
                    print(f"  Warning: Failed to decode Pred mask {i}: {e}")
    
    # 混合原图和叠加层
    alpha = 0.3  # 透明度
    blended = cv2.addWeighted(img_np, 1-alpha, overlay, alpha, 0)
    
    # 转回 PIL Image
    result_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(result_img)
    
    # 尝试加载字体
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    # 3. 绘制所有 GT bboxes（红色系粗线）
    if gt_list:
        for i, gt_item in enumerate(gt_list):
            bbox_gt = gt_item.get("bbox_gt")
            if bbox_gt is not None and len(bbox_gt) == 4:
                x1, y1, x2, y2 = bbox_gt
                color_rgb = tuple(gt_colors[i % len(gt_colors)])
                draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=3)
                # 标注实例编号
                label = f"GT{i+1}"
                draw.text((x1 + 2, y1 - 18), label, fill=color_rgb, font=font_tiny)
    
    # 4. 绘制所有 Pred bboxes（黄色细线）
    if pred_list:
        for i, pred_item in enumerate(pred_list):
            bbox = pred_item.get("bbox")
            score = pred_item.get("score", 0.0)
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 255, 0), width=2)
                # 标注置信度
                label = f"P{i+1}:{score:.2f}"
                # 绘制标签背景
                text_bbox = draw.textbbox((0, 0), label, font=font_tiny)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([x1, y1, x1 + text_width + 4, y1 + text_height + 4], 
                             fill=(255, 255, 0))
                draw.text((x1 + 2, y1 + 2), label, fill=(0, 0, 0), font=font_tiny)
    
    # 5. 添加文本标注
    
    # 任务ID和类别（左上角）
    task_text = f"Task ID: {task_id} | Detect all: {task_prompt}"
    draw.text((10, 10), task_text, fill=(255, 255, 255), font=font_large, 
             stroke_width=2, stroke_fill=(0, 0, 0))
    
    # 统计信息（右上角）
    num_gt = len(gt_list) if gt_list else 0
    num_pred = len(pred_list) if pred_list else 0
    avg_conf = np.mean([p.get("score", 0.0) for p in pred_list]) if pred_list else 0.0
    
    stats_text = f"GT: {num_gt} | Pred: {num_pred}"
    text_bbox = draw.textbbox((0, 0), stats_text, font=font_small)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text((W - text_width - 10, 10), stats_text, fill=(255, 255, 0), font=font_small,
             stroke_width=1, stroke_fill=(0, 0, 0))
    
    if avg_conf > 0:
        conf_text = f"Avg Conf: {avg_conf:.3f}"
        text_bbox = draw.textbbox((0, 0), conf_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, 35), conf_text, fill=(0, 255, 0), font=font_small,
                 stroke_width=1, stroke_fill=(0, 0, 0))
    
    # 6. 绘制图例（左下角）
    legend_y = H - 110
    draw.text((10, legend_y), "Legend:", fill=(255, 255, 255), font=font_small,
             stroke_width=1, stroke_fill=(0, 0, 0))
    
    y_offset = legend_y + 25
    # GT
    draw.text((10, y_offset), "Ground Truth:", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    y_offset += 18
    draw.rectangle([(15, y_offset), (35, y_offset + 10)], fill=(255, 0, 0))
    draw.text((40, y_offset - 2), "Mask", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    draw.rectangle([(85, y_offset), (105, y_offset + 10)], outline=(255, 0, 0), width=2)
    draw.text((110, y_offset - 2), "Bbox", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    
    # Pred
    y_offset += 20
    draw.text((10, y_offset), "Prediction:", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    y_offset += 18
    draw.rectangle([(15, y_offset), (35, y_offset + 10)], fill=(0, 255, 0))
    draw.text((40, y_offset - 2), "Mask", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    draw.rectangle([(85, y_offset), (105, y_offset + 10)], outline=(255, 255, 0), width=2)
    draw.text((110, y_offset - 2), "Bbox", fill=(255, 255, 255), font=font_tiny,
             stroke_width=1, stroke_fill=(0, 0, 0))
    
    return result_img

def main():
    # --- Configuration ---
    dataset_name = "ReplicaCAD"
    seed = 42
    scenes_size = 10
    max_scene_instance = 20  # 每个场景最多20个任务
    max_step_length = 10
    
    task_type = "class-detect"  # 固定为 class-detect 任务
    
    # 输出文件
    output_dir = f'/data1/tct_data/habitat/eval_data/replicacad_{scenes_size}-{task_type}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'task_infos.json')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # 初始化环境
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    
    # 存储所有任务信息
    all_task_infos = []
    
    # 获取场景列表
    scene_subfolder = get_dataset_subfolder(dataset_name)[:scenes_size]
    total_tasks = scenes_size * max_scene_instance
    
    print(f"开始生成 {task_type} 任务的评估数据集...")
    print(f"场景数量: {scenes_size}")
    print(f"每个场景的任务数: {max_scene_instance}")
    print(f"总任务数: {total_tasks}\n")
    
    task_count = 0
    
    for scene_idx in tqdm(range(scenes_size), desc="Processing Scenes"):
        scene_id = get_scene_path(scene_subfolder, dataset_name, scene_idx)
        
        for instance_idx in range(max_scene_instance):
            try:
                obs, info = env.reset(
                    seed=seed + task_count,
                    is_unique=True,
                    sync_info=None,
                    special_scene_id=scene_id,
                    task_type=task_type
                )

                # 获取所有同类物体的 GT（用于可视化）
                try:
                    _, gt_list = env.get_gt_for_class()
                except Exception as e:
                    print(f"  Warning: Failed to get GT for class: {e}")
                    gt_list = []
                
                # 准备任务信息（用于 reset_eval）
                task_info_raw = {
                    "task_id": task_count,  # 添加全局唯一的任务ID
                    "task_type": info["task_type"],
                    "target_category": info["target_category"],
                    "semantic_id": info["semantic_id"],
                    "instance_id": info["instance_id"],
                    "obj_handle": info["obj_handle"],
                    "obj_rotation": info["obj_rotation"],
                    "obj_translation": info["obj_translation"],
                    "gt": info["gt"],  # Now includes both bbox_gt and mask_gt
                    "task_prompt": info["task_prompt"],
                    "conf_score": info["conf_score"],
                    "phi": info["phi"],
                    "scene_id": info["scene_id"],
                    "agent_pos": info["agent_pos"],
                }
                # 转换 numpy 数组为 list，以便 JSON 序列化
                task_info = convert_numpy_to_list(task_info_raw)                

                
                # 获取预测结果（从 info 中）
                pred_list = info.get("pred", [])
                
                # 可视化任务
                vis_img = visualize_class_detect_task(
                    obs_pil=obs,
                    task_prompt=info["task_prompt"],
                    gt_list=gt_list,
                    pred_list=pred_list,
                    task_id=task_count
                )
                
                # 保存可视化图像
                image_filename = os.path.join(images_dir, f"task_{task_count:04d}.png")
                vis_img.save(image_filename)
                
                all_task_infos.append(task_info)
                
                # 打印详细信息
                num_gt = len(gt_list)
                num_pred = len(pred_list) if pred_list and pred_list[0].get('category') != 'unknown' else 0
                print(f"  Task {instance_idx + 1} (ID:{task_count}): {info['task_prompt']} | GT:{num_gt} Pred:{num_pred}")
                
                task_count += 1
                
                # 每10个任务打印一次进度
                if task_count % 10 == 0:
                    print(f"\n已生成 {task_count}/{total_tasks} 个任务")
                    print(f"最新任务 - 场景: {scene_id}")
                    print(f"目标类别: {info['target_category']}\n")
                
            except Exception as e:
                print(f"\n警告: 场景 {scene_idx} 的任务 {instance_idx} 生成失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存到JSON文件
    print(f"\n正在保存数据到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_task_infos, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✓ 成功生成 {len(all_task_infos)} 个 {task_type} 任务")
    print(f"{'='*70}")
    print(f"任务信息已保存到: {output_file}")
    print(f"可视化图像已保存到: {images_dir}/")
    print(f"  - 格式: task_XXXX.png (XXXX = task_id)")
    print(f"  - 共 {len(all_task_infos)} 张图像")
    print(f"{'='*70}")
    
    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"  总任务数: {len(all_task_infos)}")
    print(f"  任务ID范围: 0 ~ {task_count - 1}")
    
    # 统计类别分布
    category_counts = {}
    for task in all_task_infos:
        cat = task['target_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    unique_categories = set(category_counts.keys())
    print(f"  唯一类别数: {len(unique_categories)}")
    
    print("\n类别分布:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # 打印可视化说明
    print(f"\n可视化说明:")
    print(f"  颜色编码:")
    print(f"    - 红色系: Ground Truth (多个实例用不同深度的红色)")
    print(f"    - 绿色系: Predicted Masks")
    print(f"    - 黄色: Predicted Bboxes (带置信度)")
    print(f"  标注信息:")
    print(f"    - 左上角: Task ID 和目标类别")
    print(f"    - 右上角: 统计信息 (GT 数量, Pred 数量, 平均置信度)")
    print(f"    - 左下角: 图例")
    print(f"  实例标记:")
    print(f"    - GT1, GT2, ... : Ground Truth 实例编号")
    print(f"    - P1:0.XX, P2:0.XX, ... : Prediction 编号和置信度")
    
    # 清理
    env.close()
    print("\n数据集生成完成！")

if __name__ == "__main__":
    # 初始化 ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    try:
        main()
    finally:
        ray.shutdown()

