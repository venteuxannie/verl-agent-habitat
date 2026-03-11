import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import random
import time
import base64
import io
import json
import argparse
from functools import partial
from typing import Dict, List, Tuple, Optional

import requests
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)
import ray
from pycocotools import mask as mask_utils
# --- Rule-based policy functions (imported from collect_sft_dataset_rule.py) ---
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import (
    POLICY_THRESHOLDS,
    get_rule_based_action,
    extract_bbox_from_mask,
    extract_bbox_from_3dbox
)
# --- Custom Module Imports ---
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.dataset.legacy.collect_3dbox_eval_dataset import visualize_task
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import compute_bbox_iou_3d, predict_3d_bbox_from_mask
from agent_system.environments.env_package.habitat_sim.utils.third_party import call_grounding_segment_pipeline
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_UNIFIED_COT_TEMPLATE
from agent_system.environments.env_package.habitat_sim.projection import habitat_projection as unified_habitat_projection
from agent_system.multi_turn_rollout.utils import process_image

# ========================================
# 3D Bounding Box Evaluation Metrics
# ========================================

def calculate_3d_iou(pred_bbox_3d, gt_bbox_3d):
    """
    计算两个3D bounding box的IoU (使用OBB模式)
    
    Args:
        pred_bbox_3d: 预测的3D bbox字典
        gt_bbox_3d: Ground truth 3D bbox字典
        
    Returns:
        IoU score (float)
    """
    if pred_bbox_3d is None or gt_bbox_3d is None:
        return 0.0
    
    try:
        iou = compute_bbox_iou_3d(pred_bbox_3d, gt_bbox_3d, use_obb=True)
        return float(iou)
    except Exception as e:
        print(f"Error calculating 3D IoU: {e}")
        return 0.0

def calculate_center_distance_error(pred_bbox_3d, gt_bbox_3d):
    """
    计算3D bbox中心点的欧氏距离误差
    
    Args:
        pred_bbox_3d: 预测的3D bbox字典，包含'center'或'obb_center'字段
        gt_bbox_3d: Ground truth 3D bbox字典
        
    Returns:
        center distance error (float), 单位：米
    """
    if pred_bbox_3d is None or gt_bbox_3d is None:
        return float('inf')
    
    try:
        # 尝试获取中心点坐标
        pred_center = pred_bbox_3d.get('obb_center')
        gt_center = gt_bbox_3d.get('obb_center')
        
        if pred_center is None or gt_center is None:
            return float('inf')
        
        # 计算欧氏距离
        pred_center = np.array(pred_center)
        gt_center = np.array(gt_center)
        distance = np.linalg.norm(pred_center - gt_center)
        
        return float(distance)
    except Exception as e:
        print(f"Error calculating center distance: {e}")
        return float('inf')

def calculate_rotation_error(pred_bbox_3d, gt_bbox_3d):
    """
    计算3D bbox的旋转误差（绕Y轴的角度差）
    
    Args:
        pred_bbox_3d: 预测的3D bbox字典
        gt_bbox_3d: Ground truth 3D bbox字典
        
    Returns:
        rotation error in degrees (float)
    """
    if pred_bbox_3d is None or gt_bbox_3d is None:
        return float('inf')
    
    try:
        # 获取OBB的旋转轴
        pred_axes = pred_bbox_3d.get('obb_axes')
        gt_axes = gt_bbox_3d.get('obb_axes')
        
        if pred_axes is None or gt_axes is None:
            return float('inf')
        
        # 提取主方向（X轴）在XZ平面的投影
        pred_x_axis = np.array(pred_axes[0])  # (3,)
        gt_x_axis = np.array(gt_axes[0])
        
        # 投影到XZ平面
        pred_xz = np.array([pred_x_axis[0], pred_x_axis[2]])
        gt_xz = np.array([gt_x_axis[0], gt_x_axis[2]])
        
        # 归一化
        pred_xz = pred_xz / (np.linalg.norm(pred_xz) + 1e-8)
        gt_xz = gt_xz / (np.linalg.norm(gt_xz) + 1e-8)
        
        # 计算角度差（使用点积）
        cos_angle = np.clip(np.dot(pred_xz, gt_xz), -1.0, 1.0)
        angle_rad = np.arccos(abs(cos_angle))  # 取绝对值因为方向可能相反
        angle_deg = np.degrees(angle_rad) % 90.0
        
        return float(angle_deg)
    except Exception as e:
        print(f"Error calculating rotation error: {e}")
        return float('inf')

def calculate_size_error(pred_bbox_3d, gt_bbox_3d):
    """
    计算3D bbox尺寸的相对误差
    
    Args:
        pred_bbox_3d: 预测的3D bbox字典
        gt_bbox_3d: Ground truth 3D bbox字典
        
    Returns:
        size error (float), 归一化的L2误差
    """
    if pred_bbox_3d is None or gt_bbox_3d is None:
        return float('inf')
    
    try:
        # 获取尺寸信息 (优先使用obb_extents的两倍，否则使用size)
        pred_extents = pred_bbox_3d.get('obb_extents')
        gt_extents = gt_bbox_3d.get('obb_extents')
        
        if pred_extents is None or gt_extents is None:
            pred_size = pred_bbox_3d.get('size')
            gt_size = gt_bbox_3d.get('size')
            if pred_size is None or gt_size is None:
                return float('inf')
        else:
            pred_size = 2 * np.array(pred_extents)
            gt_size = 2 * np.array(gt_extents)
        
        # 计算相对误差
        pred_size = np.array(pred_size)
        gt_size = np.array(gt_size)
        
        # 归一化的L2误差
        size_diff = pred_size - gt_size
        relative_error = np.linalg.norm(size_diff) / (np.linalg.norm(gt_size) + 1e-8)
        
        return float(relative_error)
    except Exception as e:
        print(f"Error calculating size error: {e}")
        return float('inf')

# ========================================
# Model Inference & Projection
# ========================================

# Note: habitat_projection is now imported from projection.py as unified_habitat_projection
# It returns (pred_task_types, pred_task_prompts, action_indices, valids)

def load_internvl_image(image: Image.Image, config, input_size=448, max_num=12):
    """Load and preprocess image for InternVL models using dynamic preprocessing."""
    import math
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        
        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    # Use the config to get the image size if available
    if hasattr(config, 'force_image_size') and config.force_image_size:
        input_size = config.force_image_size
    elif hasattr(config, 'vision_config') and hasattr(config.vision_config, 'image_size'):
        input_size = config.vision_config.image_size
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values.to(torch.bfloat16)

def load_model_and_processor(model_path: str):
    """Loads the VL model (Qwen-VL or InternVL) and its associated processor."""
    from transformers import AutoModel, AutoConfig
    
    # Detect model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type.lower()
    
    if "internvl" in model_type:
        print("Loading InternVL model from", model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        # InternVL uses tokenizer, not processor in the standard way
        try:
            from verl.utils.tokenizer import hf_processor
            processor = hf_processor(model_path, trust_remote_code=True)
            if processor is None:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        except ImportError:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    elif "qwen2.5" in model_type or "qwen2_5" in model_type:
        print("Loading Qwen2.5-VL model from", model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "qwen2" in model_type:
        print("Loading Qwen2-VL model from", model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please provide a valid Qwen2, Qwen2.5, or InternVL model path.")
    
    return model, processor, model_type

def inference(model, processor, image: Image.Image, prompt: str, model_type: str, max_new_tokens: int = 1024) -> str:
    """Performs inference with the model given an image and a text prompt."""
    if "internvl" in model_type:
        # InternVL-specific inference
        if hasattr(processor, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text_prompt = prompt
        
        if "<image>" in text_prompt and not hasattr(model, 'chat'):
            text_prompt = text_prompt.replace("<image>", "<IMG_CONTEXT>")
        
        if hasattr(model, 'chat'):
            from transformers import AutoTokenizer
            if not isinstance(processor, AutoTokenizer):
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, use_fast=False)
            else:
                tokenizer = processor
            
            pixel_values = load_internvl_image(image, model.config)
            if hasattr(model, 'device'):
                pixel_values = pixel_values.to(model.device)
            else:
                pixel_values = pixel_values.cuda()
            
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return [response] if isinstance(response, str) else response
        else:
            if hasattr(processor, '__call__'):
                inputs = processor(
                    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
                ).to(model.device)
            else:
                inputs = processor(
                    text_prompt, return_tensors="pt"
                ).to(model.device)
                pixel_values = load_internvl_image(image, model.config)
                if hasattr(model, 'device'):
                    pixel_values = pixel_values.to(model.device)
                else:
                    pixel_values = pixel_values.cuda()
                inputs['pixel_values'] = pixel_values
            
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return output_text if output_text else ""
    else:
        # Qwen-VL inference
        messages = [{"role": "user", "content": prompt}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)
        
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return output_text if output_text else ""

def build_text_obs(infos: List[Dict]) -> List[str]:
    """Builds the text observation (prompt) for the agent using the unified template."""
    postprocess_text_obs = []
    for i in range(len(infos)):
        # Use task_description if available, otherwise fall back to task_prompt
        task_description = infos[i].get('task_description', infos[i].get('task_prompt', ''))
        conf_score = infos[i].get('conf_score') or 0.0
        
        prompt = HABITAT_UNIFIED_COT_TEMPLATE.format(
            task_description=task_description,
            conf_score=conf_score
        )
        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def image_to_base64_str(obs, pred_bbox_3d=None, gt_bbox_3d=None, task_prompt=None, task_id=None):
    """将图像转换为Base64编码的字符串，并可选择叠加3D bbox可视化"""
    if not isinstance(obs, Image.Image):
        img = Image.fromarray(obs)
    else:
        img = obs.copy()
    
    # 可视化3D bbox结果
    if pred_bbox_3d is not None or gt_bbox_3d is not None:
        img = visualize_task(img, task_prompt, gt_bbox_3d, pred_bbox_3d, task_id)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

# ========================================
# Main Evaluation Logic
# ========================================

def main(args):
    """主评估逻辑函数 - 专门用于 3D-Box 任务"""
    
    # --- Configuration ---
    mode = args.mode
    model_path = args.model_path
    log_filename = args.log_filename
    
    # 自动生成 txt 日志文件名
    txt_log_filename = os.path.splitext(log_filename)[0] + ".txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    dataset_name = "ReplicaCAD"
    seed = 0
    scenes_size = 10
    max_scene_instance = 10
    max_step_length = 10

    # --- Load Model and Processor (only if mode is 'model') ---
    model, processor, model_type = None, None, None
    if mode == 'model':
        if not model_path:
            raise ValueError("model_path is required when mode='model'")
        model, processor, model_type = load_model_and_processor(model_path)
        print(f"Model and processor loaded successfully. Model type: {model_type}")
    else:
        print(f"Running in {mode.upper()} mode, skipping model loading.")

    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    projection_f = partial(unified_habitat_projection, env_name='habitat')
    print("Habitat environment created.")

    # --- 初始化列表用于存储评估指标和日志 ---
    html_log_entries = []
    txt_log_entries = []
    
    # 3D-Box specific metrics
    all_initial_ious = []
    all_final_ious = []
    all_initial_center_errors = []
    all_final_center_errors = []
    all_initial_rotation_errors = []
    all_final_rotation_errors = []
    all_initial_size_errors = []
    all_final_size_errors = []
    all_initial_confs = []
    all_final_confs = []
    
    # Task type prediction tracking
    all_first_predictions = []       # 记录每个任务第一次预测结果
    task_type_correct_indices = []   # 记录 task_type 预测正确的任务索引

    input_filename = args.input_filename
    input_file_path = f'/data/tct/habitat/eval_data/{input_filename}/task_infos_3dbox.json'
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_task_infos = json.load(f)
        print(f"成功从 {input_file_path} 中加载了 {len(all_task_infos)} 条任务信息。\n")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}。请先生成测试任务文件。")
        all_task_infos = []
        return

    # ----------------- 主循环开始 -----------------
    total_tasks = len(all_task_infos)
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing 3D-Box Tasks")):
        task_info["task_type"] = "3d-box"
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测 3D bbox
        response = call_grounding_segment_pipeline(obs, info.get("task_prompt"))
        segment_mask = response.get("segment_mask")
        bbox_3d = predict_3d_bbox_from_mask(
            mask_rle=segment_mask,
            depth_obs= env.sim.get_sensor_observations()["depth"],
            agent_state=env.sim.get_agent(0).get_state(),
            hfov=90.0,
            denoise=True,
            align_to_ground=True
        )
        
        # conf_score 来自 grounding_score 和 segment_score 的平均
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        geometric_confidence = bbox_3d.get("geometric_confidence", 0.0)
        
        initial_pred_bbox_3d = bbox_3d
        initial_conf = (grounding_score + segment_score + geometric_confidence) / 3.0
        
        # 获取 GT 3D bbox
        gt_info = info.get("gt", {})
        gt_bbox_3d = gt_info.get("bbox_3d_gt")
        
        # 计算初始指标
        initial_iou = calculate_3d_iou(initial_pred_bbox_3d, gt_bbox_3d)
        initial_center_error = calculate_center_distance_error(initial_pred_bbox_3d, gt_bbox_3d)
        initial_rotation_error = calculate_rotation_error(initial_pred_bbox_3d, gt_bbox_3d)
        initial_size_error = calculate_size_error(initial_pred_bbox_3d, gt_bbox_3d)
        
        all_initial_ious.append(initial_iou)
        all_initial_center_errors.append(initial_center_error if initial_center_error != float('inf') else 10.0)  # Cap at 10m
        all_initial_rotation_errors.append(initial_rotation_error if initial_rotation_error != float('inf') else 180.0)
        all_initial_size_errors.append(initial_size_error if initial_size_error != float('inf') else 2.0)
        all_initial_confs.append(initial_conf)
        
        # 记录开始信息
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}\n"
            f"Initial Metrics - IoU: {initial_iou:.4f}, Center Error: {initial_center_error:.4f}m, "
            f"Rotation Error: {initial_rotation_error:.2f}°, Size Error: {initial_size_error:.4f}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        episode_header_html = f"""
        <h2>--- Episode Start: Task {idx+1}/{total_tasks} (Scene: {info.get('scene_id', 'N/A')}) ---</h2>
        <p><b>Task Prompt:</b> {info.get('task_prompt', 'N/A')}</p>
        <p><b>Initial Metrics:</b> IoU: {initial_iou:.4f}, Center Error: {initial_center_error:.4f}m, 
        Rotation Error: {initial_rotation_error:.2f}°, Size Error: {initial_size_error:.4f}</p>
        """
        html_log_entries.append(episode_header_html)
        
        done = False
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            print(f"Task: {info['task_prompt']}")
            
            prompt = build_text_obs([info])[0]
            
            # --- 根据模式选择动作 ---
            # Get ground truth task_type and task_prompt for non-model modes
            gt_task_type = info.get("task_type", "3d-box")
            gt_task_prompt = info.get("task_prompt", "")
            
            if mode == 'model':
                # 模型推理模式：模型预测 task_type, task_prompt, action
                # For Qwen models, replace <image> with special tokens
                if "qwen" in model_type:
                    prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                # For InternVL, keep <image> as is (it will be handled in inference function)
                text_actions = inference(model, processor, process_image(obs), prompt, model_type)
                # unified_habitat_projection returns (pred_task_types, pred_task_prompts, actions, valids)
                pred_task_types, pred_task_prompts, actions, valids = projection_f(text_actions)
                action_name = env.action_space[actions[0]]
            elif mode == 'rule':
                # 基于规则的策略：使用 ground truth task_type 和 task_prompt
                image_width, image_height = obs.size
                task_type = gt_task_type
                pred = info.get("pred")
                
                # 根据任务类型提取正确的bbox
                bbox_for_action = None
                if task_type == "grounding":
                    bbox_for_action = pred
                elif task_type == "class-detect":
                    if pred and isinstance(pred, list) and len(pred) > 0 and pred[0].get("category") != "unknown":
                        bbox_for_action = pred[0]["bbox"]
                elif task_type == "segment":
                    bbox_for_action = extract_bbox_from_mask(pred.get("segment_mask") if pred else None)
                elif task_type == "detect":
                    bbox_for_action = None
                elif task_type == "3d-box":
                    bbox_for_action = extract_bbox_from_3dbox(pred, image_size=(image_width, image_height), hfov=90.0)
                
                action_name = get_rule_based_action(
                    bbox=bbox_for_action,
                    image_width=image_width,
                    image_height=image_height,
                    action_space=env.action_space,
                    thresholds=POLICY_THRESHOLDS
                )
                text_actions = [f'{{"task_type": "{gt_task_type}", "task_prompt": "{gt_task_prompt}", "action": "{action_name}"}}']
                pred_task_types = [gt_task_type]
                pred_task_prompts = [gt_task_prompt]
                actions = [env.action_space.index(action_name)]
                valids = [1]
            elif mode == 'forward':
                # 向前动作模式：使用 ground truth task_type 和 task_prompt
                action_name = "move_forward"
                text_actions = [f'{{"task_type": "{gt_task_type}", "task_prompt": "{gt_task_prompt}", "action": "{action_name}"}}']
                pred_task_types = [gt_task_type]
                pred_task_prompts = [gt_task_prompt]
                actions = [0]
                valids = [1]
            else:  # mode == 'random'
                # 随机动作模式：使用 ground truth task_type 和 task_prompt，随机选择 action
                text_actions = ["random"]
                pred_task_types, pred_task_prompts, actions, valids = projection_f(text_actions)
                # Override with ground truth task_type and task_prompt for random mode
                pred_task_types = [gt_task_type]
                pred_task_prompts = [gt_task_prompt]
                action_name = env.action_space[actions[0]]
            
            # 记录步骤信息
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Pred Task Type: '{pred_task_types[0]}', Pred Task Prompt: '{pred_task_prompts[0][:50]}...'\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)
            
            # 记录第一次预测结果（仅在 k==0 时）
            if k == 0:
                is_task_type_correct = (pred_task_types[0] == gt_task_type)
                all_first_predictions.append({
                    "task_id": task_info.get("task_id", idx),
                    "gt_task_type": gt_task_type,
                    "pred_task_type": pred_task_types[0],
                    "gt_task_prompt": gt_task_prompt,
                    "pred_task_prompt": pred_task_prompts[0],
                    "is_task_type_correct": is_task_type_correct
                })
                if is_task_type_correct:
                    task_type_correct_indices.append(idx)

            # 可视化当前步骤
            gt_info = info.get("gt", {})
            gt_bbox_3d = gt_info.get("bbox_3d_gt") if gt_info else None
            
            if k == 0:
                pred_bbox_3d = initial_pred_bbox_3d
            else:
                pred_bbox_3d = info.get("pred")
            base64_image = image_to_base64_str(obs, pred_bbox_3d, gt_bbox_3d, info.get('task_prompt'), idx+1)
            
            step_html = f"""
            <div class="step-container">
                <h3>STEP {k+1}/{max_step_length}</h3>
                <p><b>Task:</b> {info['task_prompt']}</p>
                <img src="{base64_image}" alt="Agent view at step {k+1}" class="agent-view" style="max-width: 800px;">
                <div class="model-output">
                    <p><b>Model Output:</b> <code>{text_actions[0]}</code></p>
                    <p><b>Pred Task Type:</b> '{pred_task_types[0]}', <b>Pred Task Prompt:</b> '{pred_task_prompts[0][:50]}...'</p>
                    <p><b>Predicted Action:</b> '{action_name}' (Valid: {bool(valids[0])})</p>
                </div>
            </div>
            """
            html_log_entries.append(step_html)
            
            # env.step now requires (pred_task_type, pred_task_prompt, action_index, is_valid_action)
            obs, reward, done, info = env.step(
                pred_task_types[0], 
                pred_task_prompts[0], 
                actions[0], 
                valids[0]
            )

            if done:
                # 最终步骤的图像
                final_gt_info = info.get("gt", {})
                final_gt_bbox_3d = final_gt_info.get("bbox_3d_gt") if final_gt_info else None
                final_pred_bbox_3d = info.get("pred")
                final_base64_image = image_to_base64_str(obs, final_pred_bbox_3d, final_gt_bbox_3d, info.get('task_prompt'))
                
                final_step_html = f"""
                <div class="step-container">
                    <h3>FINAL STEP (Episode End)</h3>
                    <p><b>Task:</b> {info['task_prompt']}</p>
                    <img src="{final_base64_image}" alt="Final agent view" class="agent-view" style="max-width: 800px;">
                    <div class="model-output">
                        <p><b>Model Output:</b> <code>{text_actions[0]}</code></p>
                        <p><b>Predicted Action:</b> '{action_name}' (Valid: {bool(valids[0])})</p>
                        <p><b>Reward:</b> {reward}</p>
                    </div>
                </div>
                """
                html_log_entries.append(final_step_html)
                break
        
        # 计算最终指标
        final_pred_bbox_3d = info.get("pred")
        final_conf = info.get("conf_score", 0.0)
        final_gt_bbox_3d = info.get("gt", {}).get("bbox_3d_gt")
        
        final_iou = calculate_3d_iou(final_pred_bbox_3d, final_gt_bbox_3d)
        final_center_error = calculate_center_distance_error(final_pred_bbox_3d, final_gt_bbox_3d)
        final_rotation_error = calculate_rotation_error(final_pred_bbox_3d, final_gt_bbox_3d)
        final_size_error = calculate_size_error(final_pred_bbox_3d, final_gt_bbox_3d)
        
        all_final_ious.append(final_iou)
        all_final_center_errors.append(final_center_error if final_center_error != float('inf') else 10.0)
        all_final_rotation_errors.append(final_rotation_error if final_rotation_error != float('inf') else 180.0)
        all_final_size_errors.append(final_size_error if final_size_error != float('inf') else 2.0)
        all_final_confs.append(final_conf)

        # 计算指标变化
        iou_change = final_iou - initial_iou
        center_error_change = final_center_error - initial_center_error
        rotation_error_change = final_rotation_error - initial_rotation_error
        size_error_change = final_size_error - initial_size_error
        conf_change = final_conf - initial_conf
        
        # 文本日志格式
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"IoU: {initial_iou:.4f} → {final_iou:.4f} ({iou_change:+.4f})\n"
                f"Center Error: {initial_center_error:.4f}m → {final_center_error:.4f}m ({center_error_change:+.4f}m)\n"
                f"Rotation Error: {initial_rotation_error:.2f}° → {final_rotation_error:.2f}° ({rotation_error_change:+.2f}°)\n"
                f"Size Error: {initial_size_error:.4f} → {final_size_error:.4f} ({size_error_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"IoU: {initial_iou:.4f} → {final_iou:.4f} ({iou_change:+.4f})\n"
                f"Center Error: {initial_center_error:.4f}m → {final_center_error:.4f}m ({center_error_change:+.4f}m)\n"
                f"Rotation Error: {initial_rotation_error:.2f}° → {final_rotation_error:.2f}° ({rotation_error_change:+.2f}°)\n"
                f"Size Error: {initial_size_error:.4f} → {final_size_error:.4f} ({size_error_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        
        print(episode_finish_str)
        txt_log_entries.append(episode_finish_str + "\n")
        
        # HTML日志格式 - 带颜色编码
        def get_colored_change(change, lower_is_better=False):
            """为变化值添加颜色（lower_is_better=True表示误差类指标）"""
            if lower_is_better:
                if change < 0:  # 误差减小是好的
                    return f'<span style="color: green; font-weight: bold;">{change:+.4f}</span>'
                elif change > 0:
                    return f'<span style="color: red; font-weight: bold;">{change:+.4f}</span>'
                else:
                    return f'<span style="color: gray;">{change:+.4f}</span>'
            else:
                if change > 0:  # IoU增加是好的
                    return f'<span style="color: green; font-weight: bold;">{change:+.4f}</span>'
                elif change < 0:
                    return f'<span style="color: red; font-weight: bold;">{change:+.4f}</span>'
                else:
                    return f'<span style="color: gray;">{change:+.4f}</span>'
        
        if done:
            finish_html = f"""
            <div class="episode-summary">
                <h3>Episode finished at step {k+1} with reward {reward}</h3>
                <div class="metrics-comparison">
                    <p><b>IoU:</b> {initial_iou:.4f} → <span style="font-weight: bold;">{final_iou:.4f}</span> ({get_colored_change(iou_change, False)})</p>
                    <p><b>Center Error:</b> {initial_center_error:.4f}m → <span style="font-weight: bold;">{final_center_error:.4f}m</span> ({get_colored_change(center_error_change, True)})</p>
                    <p><b>Rotation Error:</b> {initial_rotation_error:.2f}° → <span style="font-weight: bold;">{final_rotation_error:.2f}°</span> ({get_colored_change(rotation_error_change, True)})</p>
                    <p><b>Size Error:</b> {initial_size_error:.4f} → <span style="font-weight: bold;">{final_size_error:.4f}</span> ({get_colored_change(size_error_change, True)})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({get_colored_change(conf_change, False)})</p>
                </div>
            </div>
            <hr>
            """
        else:
            finish_html = f"""
            <div class="episode-summary">
                <h3>Episode timed out at step {k+1}</h3>
                <div class="metrics-comparison">
                    <p><b>IoU:</b> {initial_iou:.4f} → <span style="font-weight: bold;">{final_iou:.4f}</span> ({get_colored_change(iou_change, False)})</p>
                    <p><b>Center Error:</b> {initial_center_error:.4f}m → <span style="font-weight: bold;">{final_center_error:.4f}m</span> ({get_colored_change(center_error_change, True)})</p>
                    <p><b>Rotation Error:</b> {initial_rotation_error:.2f}° → <span style="font-weight: bold;">{final_rotation_error:.2f}°</span> ({get_colored_change(rotation_error_change, True)})</p>
                    <p><b>Size Error:</b> {initial_size_error:.4f} → <span style="font-weight: bold;">{final_size_error:.4f}</span> ({get_colored_change(size_error_change, True)})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({get_colored_change(conf_change, False)})</p>
                </div>
            </div>
            <hr>
            """
        
        html_log_entries.append(finish_html)

    # ----------------- 计算并打印总体统计信息 -----------------
    
    # --- 计算 task_type 预测正确率 ---
    num_task_type_correct = len(task_type_correct_indices)
    task_type_accuracy = num_task_type_correct / total_tasks if total_tasks > 0 else 0.0
    print(f"\nTask Type Accuracy: {num_task_type_correct}/{total_tasks} = {task_type_accuracy:.4f}")
    
    # --- 筛选出 task_type 预测正确的任务的指标 ---
    correct_initial_ious = [all_initial_ious[i] for i in task_type_correct_indices]
    correct_final_ious = [all_final_ious[i] for i in task_type_correct_indices]
    correct_initial_center_errors = [all_initial_center_errors[i] for i in task_type_correct_indices]
    correct_final_center_errors = [all_final_center_errors[i] for i in task_type_correct_indices]
    correct_initial_rotation_errors = [all_initial_rotation_errors[i] for i in task_type_correct_indices]
    correct_final_rotation_errors = [all_final_rotation_errors[i] for i in task_type_correct_indices]
    correct_initial_size_errors = [all_initial_size_errors[i] for i in task_type_correct_indices]
    correct_final_size_errors = [all_final_size_errors[i] for i in task_type_correct_indices]
    
    avg_initial_iou = np.mean(all_initial_ious)
    avg_final_iou = np.mean(all_final_ious)
    avg_initial_center_error = np.mean(all_initial_center_errors)
    avg_final_center_error = np.mean(all_final_center_errors)
    avg_initial_rotation_error = np.mean(all_initial_rotation_errors)
    avg_final_rotation_error = np.mean(all_final_rotation_errors)
    avg_initial_size_error = np.mean(all_initial_size_errors)
    avg_final_size_error = np.mean(all_final_size_errors)
    avg_initial_conf = np.mean(all_initial_confs)
    avg_final_conf = np.mean(all_final_confs)
    
    # Task type 正确任务的平均指标
    if correct_initial_ious:
        correct_avg_initial_iou = np.mean(correct_initial_ious)
        correct_avg_final_iou = np.mean(correct_final_ious)
        correct_avg_initial_center_error = np.mean(correct_initial_center_errors)
        correct_avg_final_center_error = np.mean(correct_final_center_errors)
        correct_avg_initial_rotation_error = np.mean(correct_initial_rotation_errors)
        correct_avg_final_rotation_error = np.mean(correct_final_rotation_errors)
        correct_avg_initial_size_error = np.mean(correct_initial_size_errors)
        correct_avg_final_size_error = np.mean(correct_final_size_errors)
    else:
        correct_avg_initial_iou = correct_avg_final_iou = 0.0
        correct_avg_initial_center_error = correct_avg_final_center_error = 0.0
        correct_avg_initial_rotation_error = correct_avg_final_rotation_error = 0.0
        correct_avg_initial_size_error = correct_avg_final_size_error = 0.0

    # 计算百分比变化
    iou_pct = ((avg_final_iou - avg_initial_iou) / (avg_initial_iou + 1e-8)) * 100
    center_error_pct = ((avg_final_center_error - avg_initial_center_error) / (avg_initial_center_error + 1e-8)) * 100
    rotation_error_pct = ((avg_final_rotation_error - avg_initial_rotation_error) / (avg_initial_rotation_error + 1e-8)) * 100
    size_error_pct = ((avg_final_size_error - avg_initial_size_error) / (avg_initial_size_error + 1e-8)) * 100
    conf_pct = ((avg_final_conf - avg_initial_conf) / (avg_initial_conf + 1e-8)) * 100
    
    # Task type 正确任务的百分比变化
    correct_iou_pct = ((correct_avg_final_iou - correct_avg_initial_iou) / (correct_avg_initial_iou + 1e-8)) * 100
    correct_center_error_pct = ((correct_avg_final_center_error - correct_avg_initial_center_error) / (correct_avg_initial_center_error + 1e-8)) * 100
    correct_rotation_error_pct = ((correct_avg_final_rotation_error - correct_avg_initial_rotation_error) / (correct_avg_initial_rotation_error + 1e-8)) * 100
    correct_size_error_pct = ((correct_avg_final_size_error - correct_avg_initial_size_error) / (correct_avg_initial_size_error + 1e-8)) * 100
    
    summary_str = f"""
========================================
    3D-Box Task Evaluation Summary
========================================
Total Tasks: {total_tasks}

Task Type Prediction:
  Accuracy: {num_task_type_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)

--- All Tasks ---
3D IoU:
  Initial: {avg_initial_iou:.4f}
  Final:   {avg_final_iou:.4f}
  Change:  {avg_final_iou - avg_initial_iou:+.4f} ({iou_pct:+.2f}%)

Center Distance Error (m):
  Initial: {avg_initial_center_error:.4f}
  Final:   {avg_final_center_error:.4f}
  Change:  {avg_final_center_error - avg_initial_center_error:+.4f} ({center_error_pct:+.2f}%)

Rotation Error (degrees):
  Initial: {avg_initial_rotation_error:.2f}
  Final:   {avg_final_rotation_error:.2f}
  Change:  {avg_final_rotation_error - avg_initial_rotation_error:+.2f} ({rotation_error_pct:+.2f}%)

Size Error (relative):
  Initial: {avg_initial_size_error:.4f}
  Final:   {avg_final_size_error:.4f}
  Change:  {avg_final_size_error - avg_initial_size_error:+.4f} ({size_error_pct:+.2f}%)

Confidence Score:
  Initial: {avg_initial_conf:.4f}
  Final:   {avg_final_conf:.4f}
  Change:  {avg_final_conf - avg_initial_conf:+.4f} ({conf_pct:+.2f}%)

--- Only Task Type Correct ({num_task_type_correct} tasks) ---
3D IoU:
  Initial: {correct_avg_initial_iou:.4f}
  Final:   {correct_avg_final_iou:.4f}
  Change:  {correct_avg_final_iou - correct_avg_initial_iou:+.4f} ({correct_iou_pct:+.2f}%)

Center Distance Error (m):
  Initial: {correct_avg_initial_center_error:.4f}
  Final:   {correct_avg_final_center_error:.4f}
  Change:  {correct_avg_final_center_error - correct_avg_initial_center_error:+.4f} ({correct_center_error_pct:+.2f}%)

Rotation Error (degrees):
  Initial: {correct_avg_initial_rotation_error:.2f}
  Final:   {correct_avg_final_rotation_error:.2f}
  Change:  {correct_avg_final_rotation_error - correct_avg_initial_rotation_error:+.2f} ({correct_rotation_error_pct:+.2f}%)

Size Error (relative):
  Initial: {correct_avg_initial_size_error:.4f}
  Final:   {correct_avg_final_size_error:.4f}
  Change:  {correct_avg_final_size_error - correct_avg_initial_size_error:+.4f} ({correct_size_error_pct:+.2f}%)
========================================
"""
    
    print(summary_str)
    txt_log_entries.append(summary_str)

    # --- 保存第一次预测结果到 JSON 文件 ---
    first_predictions_output_path = f'/data/tct/habitat/eval_data/{input_filename}/first_predictions_3dbox.json'
    os.makedirs(os.path.dirname(first_predictions_output_path), exist_ok=True)
    with open(first_predictions_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_first_predictions, f, indent=4, ensure_ascii=False)
    print(f"\n✅ First predictions saved to '{first_predictions_output_path}'")

    # 保存文本日志
    with open(txt_log_filename, "w", encoding="utf-8") as f:
        f.write("".join(txt_log_entries))
    print(f"文本日志已保存到: {txt_log_filename}")

    # 生成并保存 HTML 日志
    html_summary = f"""
    <div class="summary">
        <h2>3D-Box Task Evaluation Summary</h2>
        <p><b>Total Tasks:</b> {total_tasks}</p>
        <table>
            <tr><th>Metric</th><th>Initial</th><th>Final</th><th>Change</th></tr>
            <tr><td>3D IoU</td><td>{avg_initial_iou:.4f}</td><td>{avg_final_iou:.4f}</td><td>{avg_final_iou - avg_initial_iou:+.4f}</td></tr>
            <tr><td>Center Error (m)</td><td>{avg_initial_center_error:.4f}</td><td>{avg_final_center_error:.4f}</td><td>{avg_final_center_error - avg_initial_center_error:+.4f}</td></tr>
            <tr><td>Rotation Error (°)</td><td>{avg_initial_rotation_error:.2f}</td><td>{avg_final_rotation_error:.2f}</td><td>{avg_final_rotation_error - avg_initial_rotation_error:+.2f}</td></tr>
            <tr><td>Size Error</td><td>{avg_initial_size_error:.4f}</td><td>{avg_final_size_error:.4f}</td><td>{avg_final_size_error - avg_initial_size_error:+.4f}</td></tr>
            <tr><td>Confidence</td><td>{avg_initial_conf:.4f}</td><td>{avg_final_conf:.4f}</td><td>{avg_final_conf - avg_initial_conf:+.4f}</td></tr>
        </table>
    </div>
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D-Box Task Evaluation Log</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f4;
            }}
            .summary {{
                background-color: #fff;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            .step-container {{
                background-color: #fff;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .agent-view {{
                max-width: 100%;
                border: 2px solid #ddd;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .model-output {{
                background-color: #f9f9f9;
                padding: 10px;
                border-left: 4px solid #4CAF50;
                margin-top: 10px;
            }}
            .episode-summary {{
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .metrics-comparison {{
                margin-top: 10px;
            }}
            hr {{
                border: none;
                border-top: 2px solid #ccc;
                margin: 30px 0;
            }}
        </style>
    </head>
    <body>
        {html_summary}
        {''.join(html_log_entries)}
    </body>
    </html>
    """
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 日志已保存到: {log_filename}")
    
    print("\n评估完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Habitat environment evaluation with different action selection modes.")
    parser.add_argument('--mode', type=str, choices=['model', 'random', 'rule', 'forward'], default='model',
                        help='Action selection mode: "model" for VL model inference, "random" for random actions, "rule" for rule-based policy.')
    parser.add_argument('--model_path', type=str, default='/data/tct/models/RL/InternVL3_5-2B-unified-tasks-rl', 
                        help='Path to the VL model checkpoint directory. Required when --mode=model.')
    parser.add_argument('--input_filename', type=str, 
                        default='replicacad_10-any-500-seen',
                        help='Path to the input_filename.')
    parser.add_argument(
        "--log_filename",
        type=str,
        default="/data/tct/ActivePerception/eval_logs/3dbox.html",
        help="Path to save the HTML log file"
    )
    
    args = parser.parse_args()
    
    # 初始化 ray
    ray.init(_temp_dir="/data/tct/ActivePerception/tmp", ignore_reinit_error=True)
    
    main(args)

