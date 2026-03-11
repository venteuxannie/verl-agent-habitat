"""
生成 Segment 任务的测试数据集

该脚本从 Habitat 环境中随机采样场景和对象，
为每个任务保存必要的同步信息（场景ID、对象信息、agent位置等），
以便后续用于 eval_segment.py 进行一致性评估。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import random
from tqdm import tqdm
import ray
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import get_dataset_subfolder

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

def visualize_task(obs_pil, task_prompt, gt_dict, pred_dict, task_id):
    """
    可视化任务：在观测图像上标注完整的 GT 和 Pred 信息
    
    Args:
        obs_pil: PIL Image，原始观测图像
        task_prompt: str，任务描述
        gt_dict: dict，包含 bbox_gt 和 mask_gt
        pred_dict: dict，包含 grounding_bbox, grounding_score, segment_score, segment_mask
        task_id: int，任务ID
        
    Returns:
        PIL Image，标注后的图像
    """
    # 转换为 numpy 数组
    img_np = np.array(obs_pil)
    H, W = img_np.shape[:2]
    
    # 创建叠加层
    overlay = img_np.copy()
    
    # 提取数据
    gt_mask_rle = gt_dict.get("mask_gt") if gt_dict else None
    gt_bbox = gt_dict.get("bbox_gt") if gt_dict else None
    
    if pred_dict:
        pred_mask_rle = pred_dict.get("segment_mask")
        pred_bbox = pred_dict.get("grounding_bbox")
        grounding_score = pred_dict.get("grounding_score", 0.0)
        segment_score = pred_dict.get("segment_score", 0.0)
    else:
        pred_mask_rle = None
        pred_bbox = None
        grounding_score = 0.0
        segment_score = 0.0
    
    # 1. 叠加 GT mask (红色半透明)
    if gt_mask_rle is not None:
        try:
            gt_mask = mask_utils.decode(gt_mask_rle)
            overlay[gt_mask > 0] = [255, 0, 0]  # 红色
        except Exception as e:
            print(f"  Warning: Failed to decode GT mask: {e}")
    
    # 2. 叠加 pred mask (绿色半透明)
    if pred_mask_rle is not None:
        try:
            pred_mask = mask_utils.decode(pred_mask_rle)
            overlay[pred_mask > 0] = [0, 255, 0]  # 绿色
        except Exception as e:
            print(f"  Warning: Failed to decode pred mask: {e}")
    
    # 混合原图和叠加层
    alpha = 0.4  # 透明度
    blended = cv2.addWeighted(img_np, 1-alpha, overlay, alpha, 0)
    
    # 转回 PIL Image
    result_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(result_img)
    
    # 绘制 bboxes
    # GT bbox (红色粗线)
    if gt_bbox is not None and len(gt_bbox) == 4:
        xmin, xmax, ymin, ymax = gt_bbox
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=3)
    
    # Pred bbox (黄色细线)
    if pred_bbox is not None:
        if len(pred_bbox) == 4:
            # 格式: (xmin, ymin, xmax, ymax) 或 (xmin, xmax, ymin, ymax)
            # 根据third_party.py，grounding_bbox应该是 (xmin, ymin, xmax, ymax)
            x1, y1, x2, y2 = pred_bbox
            if x2 < x1:  # 如果是 (xmin, xmax, ymin, ymax) 格式
                x1, x2, y1, y2 = pred_bbox[0], pred_bbox[2], pred_bbox[1], pred_bbox[3]
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 255, 0), width=2)
    
    # 尝试加载字体（如果失败则使用默认字体）
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    # 3. 添加文本标注（无背景框）
    
    # 任务ID（左上角）
    task_id_text = f"Task ID: {task_id}"
    draw.text((10, 10), task_id_text, fill=(255, 255, 255), font=font_large)
    
    # 分数信息（右上角，多行）
    score_y = 5
    score_line_height = 25
    
    # Grounding Score
    if grounding_score > 0:
        ground_text = f"Ground: {grounding_score:.3f}"
        text_bbox = draw.textbbox((0, 0), ground_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, score_y + 5), ground_text, fill=(255, 255, 0), font=font_small)
        score_y += score_line_height + 5
    
    # Segment Score
    if segment_score > 0:
        seg_text = f"Segment: {segment_score:.3f}"
        text_bbox = draw.textbbox((0, 0), seg_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, score_y + 5), seg_text, fill=(0, 255, 0), font=font_small)
        score_y += score_line_height + 5
    
    # Overall Confidence (平均分)
    if grounding_score > 0 or segment_score > 0:
        avg_conf = (grounding_score + segment_score) / 2.0
        conf_text = f"Avg: {avg_conf:.3f}"
        text_bbox = draw.textbbox((0, 0), conf_text, font=font_large)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, score_y + 5), conf_text, fill=(255, 255, 255), font=font_large)
    
    # 任务描述（底部）
    # 处理长文本，自动换行
    max_chars_per_line = 80
    if len(task_prompt) > max_chars_per_line:
        prompt_line1 = task_prompt[:max_chars_per_line]
        prompt_line2 = task_prompt[max_chars_per_line:max_chars_per_line*2]
        draw.text((10, H - 55), prompt_line1, fill=(255, 255, 255), font=font_small)
        if prompt_line2:
            draw.text((10, H - 35), prompt_line2, fill=(255, 255, 255), font=font_small)
    else:
        draw.text((10, H - 30), task_prompt, fill=(255, 255, 255), font=font_small)
    
    # 图例（左下角）- 更新为包含更多信息
    legend_y = H - 130
    legend_width = 180
    legend_height = 120
    draw.text((10, legend_y + 5), "Legend:", fill=(255, 255, 255), font=font_small)
    
    # GT
    y_offset = legend_y + 25
    draw.text((10, y_offset), "Ground Truth:", fill=(255, 255, 255), font=font_tiny)
    y_offset += 18
    # GT mask (红色填充)
    draw.rectangle([(15, y_offset), (35, y_offset + 10)], fill=(255, 0, 0))
    draw.text((40, y_offset - 2), "Mask", fill=(255, 255, 255), font=font_tiny)
    # GT bbox (红色框)
    draw.rectangle([(95, y_offset), (115, y_offset + 10)], outline=(255, 0, 0), width=2)
    draw.text((120, y_offset - 2), "Bbox", fill=(255, 255, 255), font=font_tiny)
    
    # Pred
    y_offset += 20
    draw.text((10, y_offset), "Prediction:", fill=(255, 255, 255), font=font_tiny)
    y_offset += 18
    # Pred mask (绿色填充)
    draw.rectangle([(15, y_offset), (35, y_offset + 10)], fill=(0, 255, 0))
    draw.text((40, y_offset - 2), "Mask", fill=(255, 255, 255), font=font_tiny)
    # Pred bbox (黄色框)
    draw.rectangle([(95, y_offset), (115, y_offset + 10)], outline=(255, 255, 0), width=2)
    draw.text((120, y_offset - 2), "Bbox", fill=(255, 255, 255), font=font_tiny)
    
    return result_img

def main():
    # --- Configuration ---
    dataset_name = "ReplicaCAD"
    seed = 42  # 使用固定种子以确保可重复性
    scenes_size = 10
    max_scene_instance = 20  # 每个场景采样20个任务
    max_step_length = 10
    
    output_filename = '/data1/tct_data/habitat/eval_data/replicacad_10new-segment/task_infos.json'
    images_dir = '/data1/tct_data/habitat/eval_data/replicacad_10new-segment/images'
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化 Ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    # 创建环境（强制为 segment 任务）
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    print(f"Habitat environment created for dataset: {dataset_name}")
    
    all_task_infos = []
    task_id_counter = 0  # 全局任务ID计数器
    
    # scene_subfolder = env.scene_subfolder # Traning Scenes
    scene_subfolder = get_dataset_subfolder(dataset_name)[10:20] # New Scenes
    # 遍历所有场景
    for scene_idx in tqdm(range(scenes_size), desc="Processing scenes"):
        special_scene_id = get_scene_path(scene_subfolder, dataset_name, eval_id=scene_idx)
        print(f"\nLoading scene {scene_idx + 1}/{scenes_size}: {special_scene_id}")
        
        # 在每个场景中采样多个任务
        for instance_idx in range(max_scene_instance):
            try:
                # Reset 环境并强制设置为 segment 任务
                obs, info = env.reset(seed=seed + scene_idx * 1000 + instance_idx, 
                                      is_unique=True, 
                                      sync_info=None, 
                                      special_scene_id=special_scene_id,
                                      task_type="segment")
                
                # 构造任务信息字典，包含所有评估所需的信息
                task_info_raw = {
                    "task_id": task_id_counter,  # 添加全局唯一的任务ID
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
                
                # 可视化并保存图像
                gt_dict = info["gt"]  # 包含 bbox_gt 和 mask_gt
                pred_dict = info.get("pred")  # 包含完整的预测信息
                
                vis_img = visualize_task(
                    obs_pil=obs,
                    task_prompt=info["task_prompt"],
                    gt_dict=gt_dict,
                    pred_dict=pred_dict,
                    task_id=task_id_counter
                )
                
                # 保存可视化图像
                image_filename = os.path.join(images_dir, f"task_{task_id_counter:04d}.png")
                vis_img.save(image_filename)
                
                all_task_infos.append(task_info)
                print(f"  Task {instance_idx + 1} (ID:{task_id_counter}): {info['task_prompt'][:50]}... (conf={info['conf_score']:.3f})")
                
                # 递增任务ID
                task_id_counter += 1
                
            except Exception as e:
                print(f"  Error generating task {instance_idx + 1} in scene {scene_idx}: {e}")
                continue
    
    # 保存所有任务信息到 JSON 文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_task_infos, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ 成功生成 {len(all_task_infos)} 个 segment 任务")
    print(f"{'='*70}")
    print(f"任务信息已保存到: {output_filename}")
    print(f"可视化图像已保存到: {images_dir}/")
    print(f"  - 格式: task_XXXX.png (XXXX = task_id)")
    print(f"  - 共 {len(all_task_infos)} 张图像")
    print(f"{'='*70}")
    
    # 打印一些统计信息
    task_prompts = [task["task_prompt"] for task in all_task_infos]
    categories = [task["target_category"] for task in all_task_infos]
    unique_categories = set(categories)
    
    print(f"\n统计信息:")
    print(f"  总任务数: {len(all_task_infos)}")
    print(f"  任务ID范围: 0 ~ {task_id_counter - 1}")
    print(f"  唯一类别数: {len(unique_categories)}")
    print(f"  类别列表: {sorted(unique_categories)}")
    
    # 打印可视化说明
    print(f"\n可视化说明:")
    print(f"  颜色编码:")
    print(f"    - 红色: Ground Truth (Mask 填充 + Bbox 粗框)")
    print(f"    - 绿色: Predicted Mask (填充)")
    print(f"    - 黄色: Predicted Bbox (细框)")
    print(f"  标注信息:")
    print(f"    - 左上角: Task ID")
    print(f"    - 右上角: 分数信息 (Ground Score, Segment Score, Average)")
    print(f"    - 底部: Task Prompt (任务描述)")
    print(f"    - 左下角: 图例")
    
    env.close()

if __name__ == "__main__":
    main()

