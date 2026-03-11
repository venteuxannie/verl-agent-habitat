"""
生成 3D-Box 任务的测试数据集

该脚本从 Habitat 环境中随机采样场景和对象，
为每个任务保存必要的同步信息（场景ID、对象信息、agent位置等），
以便后续用于评估进行一致性评估。
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
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import visualize_bbox_on_image
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

def visualize_task(obs_pil, task_prompt, gt_bbox_3d, pred_bbox_3d, task_id, hfov=90.0):
    """
    可视化任务：在观测图像上标注 GT 和 Pred 的 3D bounding box
    
    Args:
        obs_pil: PIL Image，原始观测图像
        task_prompt: str，任务描述
        gt_bbox_3d: dict，GT 的 3D bbox（相机坐标系）
        pred_bbox_3d: dict，预测的 3D bbox（相机坐标系）
        task_id: int，任务ID
        hfov: float，水平视场角
        
    Returns:
        PIL Image，标注后的图像
    """
    # 转换为 numpy 数组
    img_np = np.array(obs_pil)
    H, W = img_np.shape[:2]
    
    # 绘制 GT bbox (红色)
    if gt_bbox_3d is not None:
        img_np = visualize_bbox_on_image(
            img_np, 
            gt_bbox_3d, 
            hfov=hfov, 
            color=(255, 0, 0), 
            thickness=3,
            use_obb=True
        )
    
    # 绘制 Pred bbox (绿色)
    if pred_bbox_3d is not None:
        img_np = visualize_bbox_on_image(
            img_np, 
            pred_bbox_3d, 
            hfov=hfov, 
            color=(0, 255, 0), 
            thickness=2,
            use_obb=True
        )
    
    # 转回 PIL Image
    result_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(result_img)
    
    # 尝试加载字体（如果失败则使用默认字体）
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    # 任务ID（左上角）
    task_id_text = f"Task ID: {task_id}"
    draw.text((10, 10), task_id_text, fill=(255, 255, 255), font=font_large)
    
    # 显示 3D bbox 信息（右上角）
    info_y = 5
    line_height = 25
    
    # GT bbox 信息
    if gt_bbox_3d is not None:
        gt_center = gt_bbox_3d.get('center', [0, 0, 0])
        gt_size = gt_bbox_3d.get('size', [0, 0, 0])
        gt_text = f"GT Size: [{gt_size[0]:.2f}, {gt_size[1]:.2f}, {gt_size[2]:.2f}]m"
        text_bbox = draw.textbbox((0, 0), gt_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, info_y + 5), gt_text, fill=(255, 0, 0), font=font_small)
        info_y += line_height
    
    # Pred bbox 信息
    if pred_bbox_3d is not None:
        pred_center = pred_bbox_3d.get('center', [0, 0, 0])
        pred_size = pred_bbox_3d.get('size', [0, 0, 0])
        pred_text = f"Pred Size: [{pred_size[0]:.2f}, {pred_size[1]:.2f}, {pred_size[2]:.2f}]m"
        text_bbox = draw.textbbox((0, 0), pred_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, info_y + 5), pred_text, fill=(0, 255, 0), font=font_small)
        info_y += line_height
        
        # 显示点云统计
        num_points = pred_bbox_3d.get('num_points_filtered', 0)
        points_text = f"Points: {num_points}"
        text_bbox = draw.textbbox((0, 0), points_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((W - text_width - 10, info_y + 5), points_text, fill=(255, 255, 255), font=font_small)
    
    # 任务描述（底部）
    max_chars_per_line = 80
    if len(task_prompt) > max_chars_per_line:
        prompt_line1 = task_prompt[:max_chars_per_line]
        prompt_line2 = task_prompt[max_chars_per_line:max_chars_per_line*2]
        draw.text((10, H - 55), prompt_line1, fill=(255, 255, 255), font=font_small)
        if prompt_line2:
            draw.text((10, H - 35), prompt_line2, fill=(255, 255, 255), font=font_small)
    else:
        draw.text((10, H - 30), task_prompt, fill=(255, 255, 255), font=font_small)
    
    # 图例（左下角）
    legend_y = H - 100
    draw.text((10, legend_y + 5), "Legend:", fill=(255, 255, 255), font=font_small)
    
    # GT bbox (红色框)
    y_offset = legend_y + 25
    draw.text((10, y_offset), "Ground Truth:", fill=(255, 255, 255), font=font_tiny)
    y_offset += 18
    draw.rectangle([(15, y_offset), (50, y_offset + 15)], outline=(255, 0, 0), width=3)
    draw.text((60, y_offset), "3D Bbox", fill=(255, 255, 255), font=font_tiny)
    
    # Pred bbox (绿色框)
    y_offset += 25
    draw.text((10, y_offset), "Prediction:", fill=(255, 255, 255), font=font_tiny)
    y_offset += 18
    draw.rectangle([(15, y_offset), (50, y_offset + 15)], outline=(0, 255, 0), width=2)
    draw.text((60, y_offset), "3D Bbox", fill=(255, 255, 255), font=font_tiny)
    
    return result_img

def main():
    # --- Configuration ---
    dataset_name = "ReplicaCAD"
    seed = 42  # 使用固定种子以确保可重复性
    scenes_size = 10
    max_scene_instance = 20  # 每个场景采样20个任务
    max_step_length = 10
    
    output_filename = '/data1/tct_data/habitat/eval_data/replicacad_10new-3dbox/task_infos.json'
    images_dir = '/data1/tct_data/habitat/eval_data/replicacad_10new-3dbox/images'
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 初始化 Ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    # 创建环境（强制为 3d-box 任务）
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
                # Reset 环境并强制设置为 3d-box 任务
                obs, info = env.reset(seed=seed + scene_idx * 1000 + instance_idx, 
                                      is_unique=True, 
                                      sync_info=None, 
                                      special_scene_id=special_scene_id,
                                      task_type="3d-box")
                
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
                    "gt": info["gt"],  # 包含 bbox_3d_gt (相机坐标系)
                    "task_prompt": info["task_prompt"],
                    "conf_score": info["conf_score"],
                    "phi": info["phi"],
                    "scene_id": info["scene_id"],
                    "agent_pos": info["agent_pos"],
                }
                
                # 转换 numpy 数组为 list，以便 JSON 序列化
                task_info = convert_numpy_to_list(task_info_raw)
                
                # 可视化并保存图像
                gt_bbox_3d = info["gt"].get("bbox_3d_gt") if info.get("gt") else None
                pred_bbox_3d = info.get("pred")  # pred 就是 bbox_3d
                
                vis_img = visualize_task(
                    obs_pil=obs,
                    task_prompt=info["task_prompt"],
                    gt_bbox_3d=gt_bbox_3d,
                    pred_bbox_3d=pred_bbox_3d,
                    task_id=task_id_counter,
                    hfov=90.0
                )
                
                # 保存可视化图像
                image_filename = os.path.join(images_dir, f"task_{task_id_counter:04d}.png")
                vis_img.save(image_filename)
                
                all_task_infos.append(task_info)
                
                # 打印任务信息
                if pred_bbox_3d is not None:
                    pred_size = pred_bbox_3d.get('size', [0, 0, 0])
                    num_points = pred_bbox_3d.get('num_points_filtered', 0)
                    print(f"  Task {instance_idx + 1} (ID:{task_id_counter}): {info['task_prompt'][:40]}... "
                          f"(conf={info['conf_score']:.3f}, size=[{pred_size[0]:.2f},{pred_size[1]:.2f},{pred_size[2]:.2f}]m, pts={num_points})")
                else:
                    print(f"  Task {instance_idx + 1} (ID:{task_id_counter}): {info['task_prompt'][:40]}... (conf={info['conf_score']:.3f})")
                
                # 递增任务ID
                task_id_counter += 1
                
            except Exception as e:
                print(f"  Error generating task {instance_idx + 1} in scene {scene_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存所有任务信息到 JSON 文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_task_infos, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ 成功生成 {len(all_task_infos)} 个 3d-box 任务")
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
    print(f"    - 红色: Ground Truth 3D Bbox (粗框)")
    print(f"    - 绿色: Predicted 3D Bbox (细框)")
    print(f"  标注信息:")
    print(f"    - 左上角: Task ID")
    print(f"    - 右上角: 3D Bbox 尺寸信息 (米)")
    print(f"    - 底部: Task Prompt (任务描述)")
    print(f"    - 左下角: 图例")
    print(f"  注意: 3D Bbox 投影到 2D 图像平面显示")
    
    env.close()

if __name__ == "__main__":
    main()


