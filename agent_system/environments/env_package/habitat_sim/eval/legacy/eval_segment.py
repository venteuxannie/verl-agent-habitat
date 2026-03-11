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

# --- Custom Module Imports ---
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.utils.third_party import call_grounding_segment_pipeline
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_UNIFIED_COT_TEMPLATE
from agent_system.environments.env_package.habitat_sim.projection import habitat_projection as unified_habitat_projection
from agent_system.multi_turn_rollout.utils import process_image

# --- Rule-based policy functions (imported from collect_sft_dataset_rule.py) ---
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import (
    POLICY_THRESHOLDS,
    get_rule_based_action,
    extract_bbox_from_mask,
    extract_bbox_from_3dbox
)

# ========================================
# Segmentation Evaluation Metrics
# ========================================

def calculate_mask_iou(pred_mask_rle, gt_mask_rle):
    """
    计算两个 RLE 格式 mask 的 IoU
    
    Args:
        pred_mask_rle: 预测的 mask (RLE 格式)
        gt_mask_rle: Ground truth mask (RLE 格式)
        
    Returns:
        IoU score (float)
    """
    if pred_mask_rle is None or gt_mask_rle is None:
        return 0.0
    
    try:
        # 解码 RLE masks
        pred_mask = mask_utils.decode(pred_mask_rle).astype(np.bool_)
        gt_mask = mask_utils.decode(gt_mask_rle).astype(np.bool_)
        
        # 计算交集和并集
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    except Exception as e:
        print(f"Error calculating mask IoU: {e}")
        return 0.0

def calculate_dice_score(pred_mask_rle, gt_mask_rle):
    """
    计算 Dice Score (F1 Score 的变体)
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    if pred_mask_rle is None or gt_mask_rle is None:
        return 0.0
    
    try:
        pred_mask = mask_utils.decode(pred_mask_rle).astype(np.bool_)
        gt_mask = mask_utils.decode(gt_mask_rle).astype(np.bool_)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        pred_area = pred_mask.sum()
        gt_area = gt_mask.sum()
        
        if pred_area + gt_area == 0:
            return 0.0
        
        dice = 2 * intersection / (pred_area + gt_area)
        return float(dice)
    except Exception as e:
        print(f"Error calculating Dice score: {e}")
        return 0.0

def calculate_pixel_accuracy(pred_mask_rle, gt_mask_rle):
    """
    计算像素级准确率
    
    Accuracy = (TP + TN) / Total
    """
    if pred_mask_rle is None or gt_mask_rle is None:
        return 0.0
    
    try:
        pred_mask = mask_utils.decode(pred_mask_rle).astype(np.bool_)
        gt_mask = mask_utils.decode(gt_mask_rle).astype(np.bool_)
        
        correct = (pred_mask == gt_mask).sum()
        total = pred_mask.size
        
        accuracy = correct / total if total > 0 else 0.0
        return float(accuracy)
    except Exception as e:
        print(f"Error calculating pixel accuracy: {e}")
        return 0.0

def calculate_precision_recall(pred_mask_rle, gt_mask_rle):
    """
    计算 Precision 和 Recall
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """
    if pred_mask_rle is None or gt_mask_rle is None:
        return 0.0, 0.0
    
    try:
        pred_mask = mask_utils.decode(pred_mask_rle).astype(np.bool_)
        gt_mask = mask_utils.decode(gt_mask_rle).astype(np.bool_)
        
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, ~gt_mask).sum()
        fn = np.logical_and(~pred_mask, gt_mask).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return float(precision), float(recall)
    except Exception as e:
        print(f"Error calculating precision/recall: {e}")
        return 0.0, 0.0

def create_gt_mask_from_bbox(bbox_gt, image_size=(800, 640)):
    """
    从 bbox_gt 创建一个简单的矩形 mask (用于可视化对比)
    注意：这不是真正的 ground truth mask，只是一个近似
    
    Args:
        bbox_gt: (xmin, xmax, ymin, ymax)
        image_size: (width, height)
        
    Returns:
        RLE format mask
    """
    if bbox_gt is None or sum(bbox_gt) == 0:
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
    else:
        xmin, xmax, ymin, ymax = bbox_gt
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    
    rle = mask_utils.encode(np.asfortranarray(mask))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle

# ========================================
# Visualization Helpers
# ========================================

def overlay_mask_on_image(image: Image.Image, mask_rle, color=(0, 255, 0), alpha=0.5):
    """
    在图像上叠加分割 mask
    
    Args:
        image: PIL Image
        mask_rle: RLE format mask
        color: RGB tuple for mask color
        alpha: transparency (0=transparent, 1=opaque)
        
    Returns:
        PIL Image with mask overlay
    """
    if mask_rle is None:
        return image
    
    try:
        # 解码 mask
        mask = mask_utils.decode(mask_rle)
        
        # 创建彩色 overlay
        image_np = np.array(image)
        overlay = image_np.copy()
        
        # 在 mask 区域应用颜色
        overlay[mask > 0] = color
        
        # 混合原图和 overlay
        blended = cv2.addWeighted(image_np, 1-alpha, overlay, alpha, 0)
        
        return Image.fromarray(blended)
    except Exception as e:
        print(f"Error overlaying mask: {e}")
        return image

def visualize_segmentation_result(image: Image.Image, pred_mask_rle, gt_bbox, task_prompt):
    """
    可视化分割结果：显示预测 mask 和 GT bbox
    
    Args:
        image: 原始图像
        pred_mask_rle: 预测的 mask
        gt_bbox: GT bounding box (xmin, xmax, ymin, ymax)
        task_prompt: 任务描述
        
    Returns:
        PIL Image with visualization
    """
    import cv2
    
    img = image.copy()
    
    # 1. 绘制 GT bbox (红色)
    if gt_bbox is not None and sum(gt_bbox) > 0:
        img = draw_bbox_with_text(img, gt_bbox, text="GT", color="red", title=task_prompt)
    
    # 2. 叠加预测 mask (绿色半透明)
    if pred_mask_rle is not None:
        try:
            mask = mask_utils.decode(pred_mask_rle)
            image_np = np.array(img)
            
            # 创建绿色 overlay
            overlay = image_np.copy()
            overlay[mask > 0] = [0, 255, 0]  # 绿色
            
            # 混合
            blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
            img = Image.fromarray(blended)
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    return img

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
        # But we'll use the custom processor from verl if available
        try:
            from verl.utils.tokenizer import hf_processor
            processor = hf_processor(model_path, trust_remote_code=True)
            if processor is None:
                # Fallback to tokenizer
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
        # Check if processor has apply_chat_template method
        if hasattr(processor, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # Fallback for tokenizer-only case
            text_prompt = prompt
        
        # Replace <image> tag with <IMG_CONTEXT> for older InternVL versions
        if "<image>" in text_prompt and not hasattr(model, 'chat'):
            text_prompt = text_prompt.replace("<image>", "<IMG_CONTEXT>")
        
        # Use InternVL's chat method if available
        if hasattr(model, 'chat'):
            # Load image using InternVL's expected format
            from transformers import AutoTokenizer
            if not isinstance(processor, AutoTokenizer):
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, use_fast=False)
            else:
                tokenizer = processor
            
            # For InternVL, we need to preprocess the image properly
            # Using dynamic_preprocess if available
            pixel_values = load_internvl_image(image, model.config)
            if hasattr(model, 'device'):
                pixel_values = pixel_values.to(model.device)
            else:
                pixel_values = pixel_values.cuda()
            
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return [response] if isinstance(response, str) else response
        else:
            # Use standard generate method with processor
            if hasattr(processor, '__call__'):
                inputs = processor(
                    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
                ).to(model.device)
            else:
                # Tokenizer-only case
                inputs = processor(
                    text_prompt, return_tensors="pt"
                ).to(model.device)
                # Need to add pixel_values manually
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

def image_to_base64_str(obs, pred_mask_rle=None, gt_bbox=None, task_prompt=None):
    """将图像转换为Base64编码的字符串，并可选择叠加分割结果"""
    import cv2
    
    if not isinstance(obs, Image.Image):
        img = Image.fromarray(obs)
    else:
        img = obs.copy()
    
    # 可视化分割结果
    if pred_mask_rle is not None or gt_bbox is not None:
        img = visualize_segmentation_result(img, pred_mask_rle, gt_bbox, task_prompt)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

# ========================================
# Main Evaluation Logic
# ========================================

def main(args):
    """主评估逻辑函数 - 专门用于 Segment 任务"""
    
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
    
    # Segmentation specific metrics
    all_initial_mask_ious = []
    all_final_mask_ious = []
    all_initial_dice_scores = []
    all_final_dice_scores = []
    all_initial_pixel_accs = []
    all_final_pixel_accs = []
    all_initial_precisions = []
    all_final_precisions = []
    all_initial_recalls = []
    all_final_recalls = []
    all_initial_confs = []
    all_final_confs = []
    
    # Task type prediction tracking
    all_first_predictions = []       # 记录每个任务第一次预测结果
    task_type_correct_indices = []   # 记录 task_type 预测正确的任务索引

    input_filename = args.input_filename
    input_file_path = f'/data/tct/habitat/eval_data/{input_filename}/task_infos_segment.json'
     
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
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Segment Tasks")):
        task_info["task_type"] = "segment"
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测 mask
        response = call_grounding_segment_pipeline(obs, info.get("task_prompt"))
        # conf_score 来自 grounding_score 和 segment_score 的平均
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        initial_conf = (grounding_score + segment_score) / 2.0
        initial_pred_mask = response.get("segment_mask")  # RLE format
        
        # 获取 GT 信息
        gt_info = info.get("gt", {})
        gt_bbox = gt_info.get("bbox_gt")
        gt_mask = gt_info.get("mask_gt")
        
        # 如果没有真实的 GT mask，从 bbox 创建近似的
        if gt_mask is None and gt_bbox is not None:
            gt_mask = create_gt_mask_from_bbox(gt_bbox)
        
        # 计算初始指标
        initial_mask_iou = calculate_mask_iou(initial_pred_mask, gt_mask)
        initial_dice = calculate_dice_score(initial_pred_mask, gt_mask)
        initial_pixel_acc = calculate_pixel_accuracy(initial_pred_mask, gt_mask)
        initial_precision, initial_recall = calculate_precision_recall(initial_pred_mask, gt_mask)
        
        all_initial_mask_ious.append(initial_mask_iou)
        all_initial_dice_scores.append(initial_dice)
        all_initial_pixel_accs.append(initial_pixel_acc)
        all_initial_precisions.append(initial_precision)
        all_initial_recalls.append(initial_recall)
        all_initial_confs.append(initial_conf)
        
        # 记录开始信息
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}\n"
            f"Initial Metrics - mIoU: {initial_mask_iou:.4f}, Dice: {initial_dice:.4f}, "
            f"PixelAcc: {initial_pixel_acc:.4f}, Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        episode_header_html = f"""
        <h2>--- Episode Start: Task {idx+1}/{total_tasks} (Scene: {info.get('scene_id', 'N/A')}) ---</h2>
        <p><b>Task Prompt:</b> {info.get('task_prompt', 'N/A')}</p>
        <p><b>Initial Metrics:</b> mIoU: {initial_mask_iou:.4f}, Dice: {initial_dice:.4f}, 
        PixelAcc: {initial_pixel_acc:.4f}, Precision: {initial_precision:.4f}, Recall: {initial_recall:.4f}</p>
        """
        html_log_entries.append(episode_header_html)
        
        done = False
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            print(f"Task: {info['task_prompt']}")
            
            prompt = build_text_obs([info])[0]
            
            # --- 根据模式选择动作 ---
            # Get ground truth task_type and task_prompt for non-model modes
            gt_task_type = info.get("task_type", "segment")
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

            # 从 gt 字典中获取 bbox_gt
            gt_info = info.get("gt", {})
            if gt_info is None:
                gt_bbox = None
            else:
                gt_bbox = gt_info.get("bbox_gt")
                
            if k == 0:
                base64_image_pred = image_to_base64_str(obs, initial_pred_mask, gt_bbox, info.get('task_prompt'))
            else:
                base64_image_pred = image_to_base64_str(obs, info.get("pred")["segment_mask"], gt_bbox, info.get('task_prompt'))
            # base64_image_gt = image_to_base64_str(obs, gt_info.get("mask_gt"), None, None)
            step_html = f"""
            <div class="step-container">
                <h3>STEP {k+1}/{max_step_length}</h3>
                <p><b>Task:</b> {info['task_prompt']}</p>
                <img src="{base64_image_pred}" alt="Agent view at step {k+1}" class="agent-view" style="max-width: 800px;">
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
                final_gt_bbox = final_gt_info.get("bbox_gt")
                final_base64_image_pred = image_to_base64_str(obs, info.get("pred")["segment_mask"], final_gt_bbox, info.get('task_prompt'))
                # final_base64_image_gt = image_to_base64_str(obs, final_gt_info.get("mask_gt"), None, None)
                final_step_html = f"""
                <div class="step-container">
                    <h3>FINAL STEP (Episode End)</h3>
                    <p><b>Task:</b> {info['task_prompt']}</p>
                    <img src="{final_base64_image_pred}" alt="Final agent view" class="agent-view" style="max-width: 800px;">
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
        final_pred_mask = info.get("pred")["segment_mask"]
        final_conf = info.get("conf_score", 0.0)
        final_gt_mask = info.get("gt", {}).get("mask_gt")
        
        final_mask_iou = calculate_mask_iou(final_pred_mask, final_gt_mask)
        final_dice = calculate_dice_score(final_pred_mask, final_gt_mask)
        final_pixel_acc = calculate_pixel_accuracy(final_pred_mask, final_gt_mask)
        final_precision, final_recall = calculate_precision_recall(final_pred_mask, final_gt_mask)
        
        all_final_mask_ious.append(final_mask_iou)
        all_final_dice_scores.append(final_dice)
        all_final_pixel_accs.append(final_pixel_acc)
        all_final_precisions.append(final_precision)
        all_final_recalls.append(final_recall)
        all_final_confs.append(final_conf)

        # 计算指标变化
        miou_change = final_mask_iou - initial_mask_iou
        dice_change = final_dice - initial_dice
        pixel_acc_change = final_pixel_acc - initial_pixel_acc
        precision_change = final_precision - initial_precision
        recall_change = final_recall - initial_recall
        conf_change = final_conf - initial_conf
        
        # 文本日志格式
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"mIoU: {initial_mask_iou:.4f} → {final_mask_iou:.4f} ({miou_change:+.4f})\n"
                f"Dice: {initial_dice:.4f} → {final_dice:.4f} ({dice_change:+.4f})\n"
                f"PixelAcc: {initial_pixel_acc:.4f} → {final_pixel_acc:.4f} ({pixel_acc_change:+.4f})\n"
                f"Precision: {initial_precision:.4f} → {final_precision:.4f} ({precision_change:+.4f})\n"
                f"Recall: {initial_recall:.4f} → {final_recall:.4f} ({recall_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"mIoU: {initial_mask_iou:.4f} → {final_mask_iou:.4f} ({miou_change:+.4f})\n"
                f"Dice: {initial_dice:.4f} → {final_dice:.4f} ({dice_change:+.4f})\n"
                f"PixelAcc: {initial_pixel_acc:.4f} → {final_pixel_acc:.4f} ({pixel_acc_change:+.4f})\n"
                f"Precision: {initial_precision:.4f} → {final_precision:.4f} ({precision_change:+.4f})\n"
                f"Recall: {initial_recall:.4f} → {final_recall:.4f} ({recall_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        
        print(episode_finish_str)
        txt_log_entries.append(episode_finish_str + "\n")
        
        # HTML日志格式 - 带颜色编码
        def get_colored_change(change):
            if change > 0:
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
                    <p><b>mIoU:</b> {initial_mask_iou:.4f} → <span style="font-weight: bold;">{final_mask_iou:.4f}</span> ({get_colored_change(miou_change)})</p>
                    <p><b>Dice:</b> {initial_dice:.4f} → <span style="font-weight: bold;">{final_dice:.4f}</span> ({get_colored_change(dice_change)})</p>
                    <p><b>PixelAcc:</b> {initial_pixel_acc:.4f} → <span style="font-weight: bold;">{final_pixel_acc:.4f}</span> ({get_colored_change(pixel_acc_change)})</p>
                    <p><b>Precision:</b> {initial_precision:.4f} → <span style="font-weight: bold;">{final_precision:.4f}</span> ({get_colored_change(precision_change)})</p>
                    <p><b>Recall:</b> {initial_recall:.4f} → <span style="font-weight: bold;">{final_recall:.4f}</span> ({get_colored_change(recall_change)})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({get_colored_change(conf_change)})</p>
                </div>
            </div>
            <hr>
            """
        else:
            finish_html = f"""
            <div class="episode-summary">
                <h3>Episode timed out at step {k+1}</h3>
                <div class="metrics-comparison">
                    <p><b>mIoU:</b> {initial_mask_iou:.4f} → <span style="font-weight: bold;">{final_mask_iou:.4f}</span> ({get_colored_change(miou_change)})</p>
                    <p><b>Dice:</b> {initial_dice:.4f} → <span style="font-weight: bold;">{final_dice:.4f}</span> ({get_colored_change(dice_change)})</p>
                    <p><b>PixelAcc:</b> {initial_pixel_acc:.4f} → <span style="font-weight: bold;">{final_pixel_acc:.4f}</span> ({get_colored_change(pixel_acc_change)})</p>
                    <p><b>Precision:</b> {initial_precision:.4f} → <span style="font-weight: bold;">{final_precision:.4f}</span> ({get_colored_change(precision_change)})</p>
                    <p><b>Recall:</b> {initial_recall:.4f} → <span style="font-weight: bold;">{final_recall:.4f}</span> ({get_colored_change(recall_change)})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({get_colored_change(conf_change)})</p>
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
    correct_initial_mask_ious = [all_initial_mask_ious[i] for i in task_type_correct_indices]
    correct_final_mask_ious = [all_final_mask_ious[i] for i in task_type_correct_indices]
    correct_initial_dice_scores = [all_initial_dice_scores[i] for i in task_type_correct_indices]
    correct_final_dice_scores = [all_final_dice_scores[i] for i in task_type_correct_indices]
    correct_initial_pixel_accs = [all_initial_pixel_accs[i] for i in task_type_correct_indices]
    correct_final_pixel_accs = [all_final_pixel_accs[i] for i in task_type_correct_indices]
    correct_initial_precisions = [all_initial_precisions[i] for i in task_type_correct_indices]
    correct_final_precisions = [all_final_precisions[i] for i in task_type_correct_indices]
    correct_initial_recalls = [all_initial_recalls[i] for i in task_type_correct_indices]
    correct_final_recalls = [all_final_recalls[i] for i in task_type_correct_indices]
    
    avg_initial_miou = np.mean(all_initial_mask_ious)
    avg_final_miou = np.mean(all_final_mask_ious)
    avg_initial_dice = np.mean(all_initial_dice_scores)
    avg_final_dice = np.mean(all_final_dice_scores)
    avg_initial_pixel_acc = np.mean(all_initial_pixel_accs)
    avg_final_pixel_acc = np.mean(all_final_pixel_accs)
    avg_initial_precision = np.mean(all_initial_precisions)
    avg_final_precision = np.mean(all_final_precisions)
    avg_initial_recall = np.mean(all_initial_recalls)
    avg_final_recall = np.mean(all_final_recalls)
    avg_initial_conf = np.mean(all_initial_confs)
    avg_final_conf = np.mean(all_final_confs)
    
    # Task type 正确任务的平均指标
    if correct_initial_mask_ious:
        correct_avg_initial_miou = np.mean(correct_initial_mask_ious)
        correct_avg_final_miou = np.mean(correct_final_mask_ious)
        correct_avg_initial_dice = np.mean(correct_initial_dice_scores)
        correct_avg_final_dice = np.mean(correct_final_dice_scores)
        correct_avg_initial_pixel_acc = np.mean(correct_initial_pixel_accs)
        correct_avg_final_pixel_acc = np.mean(correct_final_pixel_accs)
        correct_avg_initial_precision = np.mean(correct_initial_precisions)
        correct_avg_final_precision = np.mean(correct_final_precisions)
        correct_avg_initial_recall = np.mean(correct_initial_recalls)
        correct_avg_final_recall = np.mean(correct_final_recalls)
    else:
        correct_avg_initial_miou = correct_avg_final_miou = 0.0
        correct_avg_initial_dice = correct_avg_final_dice = 0.0
        correct_avg_initial_pixel_acc = correct_avg_final_pixel_acc = 0.0
        correct_avg_initial_precision = correct_avg_final_precision = 0.0
        correct_avg_initial_recall = correct_avg_final_recall = 0.0

    # 计算百分比变化
    miou_pct = ((avg_final_miou - avg_initial_miou) / (avg_initial_miou + 1e-8)) * 100
    dice_pct = ((avg_final_dice - avg_initial_dice) / (avg_initial_dice + 1e-8)) * 100
    pixel_acc_pct = ((avg_final_pixel_acc - avg_initial_pixel_acc) / (avg_initial_pixel_acc + 1e-8)) * 100
    precision_pct = ((avg_final_precision - avg_initial_precision) / (avg_initial_precision + 1e-8)) * 100
    recall_pct = ((avg_final_recall - avg_initial_recall) / (avg_initial_recall + 1e-8)) * 100
    conf_pct = ((avg_final_conf - avg_initial_conf) / (avg_initial_conf + 1e-8)) * 100
    
    # Task type 正确任务的百分比变化
    correct_miou_pct = ((correct_avg_final_miou - correct_avg_initial_miou) / (correct_avg_initial_miou + 1e-8)) * 100
    correct_dice_pct = ((correct_avg_final_dice - correct_avg_initial_dice) / (correct_avg_initial_dice + 1e-8)) * 100
    correct_pixel_acc_pct = ((correct_avg_final_pixel_acc - correct_avg_initial_pixel_acc) / (correct_avg_initial_pixel_acc + 1e-8)) * 100
    correct_precision_pct = ((correct_avg_final_precision - correct_avg_initial_precision) / (correct_avg_initial_precision + 1e-8)) * 100
    correct_recall_pct = ((correct_avg_final_recall - correct_avg_initial_recall) / (correct_avg_initial_recall + 1e-8)) * 100

    summary_str = f"""
========================================
    Segment Task Evaluation Summary
========================================
Total Tasks: {total_tasks}

Task Type Prediction:
  Accuracy: {num_task_type_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)

--- All Tasks ---
Mask IoU:
  Initial: {avg_initial_miou:.4f}
  Final:   {avg_final_miou:.4f}
  Change:  {avg_final_miou - avg_initial_miou:+.4f} ({miou_pct:+.2f}%)

Dice Score:
  Initial: {avg_initial_dice:.4f}
  Final:   {avg_final_dice:.4f}
  Change:  {avg_final_dice - avg_initial_dice:+.4f} ({dice_pct:+.2f}%)

Pixel Accuracy:
  Initial: {avg_initial_pixel_acc:.4f}
  Final:   {avg_final_pixel_acc:.4f}
  Change:  {avg_final_pixel_acc - avg_initial_pixel_acc:+.4f} ({pixel_acc_pct:+.2f}%)

Precision:
  Initial: {avg_initial_precision:.4f}
  Final:   {avg_final_precision:.4f}
  Change:  {avg_final_precision - avg_initial_precision:+.4f} ({precision_pct:+.2f}%)

Recall:
  Initial: {avg_initial_recall:.4f}
  Final:   {avg_final_recall:.4f}
  Change:  {avg_final_recall - avg_initial_recall:+.4f} ({recall_pct:+.2f}%)

Confidence Score:
  Initial: {avg_initial_conf:.4f}
  Final:   {avg_final_conf:.4f}
  Change:  {avg_final_conf - avg_initial_conf:+.4f} ({conf_pct:+.2f}%)

--- Only Task Type Correct ({num_task_type_correct} tasks) ---
Mask IoU:
  Initial: {correct_avg_initial_miou:.4f}
  Final:   {correct_avg_final_miou:.4f}
  Change:  {correct_avg_final_miou - correct_avg_initial_miou:+.4f} ({correct_miou_pct:+.2f}%)

Dice Score:
  Initial: {correct_avg_initial_dice:.4f}
  Final:   {correct_avg_final_dice:.4f}
  Change:  {correct_avg_final_dice - correct_avg_initial_dice:+.4f} ({correct_dice_pct:+.2f}%)

Pixel Accuracy:
  Initial: {correct_avg_initial_pixel_acc:.4f}
  Final:   {correct_avg_final_pixel_acc:.4f}
  Change:  {correct_avg_final_pixel_acc - correct_avg_initial_pixel_acc:+.4f} ({correct_pixel_acc_pct:+.2f}%)

Precision:
  Initial: {correct_avg_initial_precision:.4f}
  Final:   {correct_avg_final_precision:.4f}
  Change:  {correct_avg_final_precision - correct_avg_initial_precision:+.4f} ({correct_precision_pct:+.2f}%)

Recall:
  Initial: {correct_avg_initial_recall:.4f}
  Final:   {correct_avg_final_recall:.4f}
  Change:  {correct_avg_final_recall - correct_avg_initial_recall:+.4f} ({correct_recall_pct:+.2f}%)
========================================
"""
    
    print(summary_str)
    txt_log_entries.append(summary_str)

    # --- 保存第一次预测结果到 JSON 文件 ---
    first_predictions_output_path = f'/data/tct/habitat/eval_data/{input_filename}/first_predictions_segment.json'
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
        <h2>Segment Task Evaluation Summary</h2>
        <p><b>Total Tasks:</b> {total_tasks}</p>
        <table>
            <tr><th>Metric</th><th>Initial</th><th>Final</th><th>Change</th></tr>
            <tr><td>Mask IoU</td><td>{avg_initial_miou:.4f}</td><td>{avg_final_miou:.4f}</td><td>{avg_final_miou - avg_initial_miou:+.4f}</td></tr>
            <tr><td>Dice Score</td><td>{avg_initial_dice:.4f}</td><td>{avg_final_dice:.4f}</td><td>{avg_final_dice - avg_initial_dice:+.4f}</td></tr>
            <tr><td>Pixel Accuracy</td><td>{avg_initial_pixel_acc:.4f}</td><td>{avg_final_pixel_acc:.4f}</td><td>{avg_final_pixel_acc - avg_initial_pixel_acc:+.4f}</td></tr>
            <tr><td>Precision</td><td>{avg_initial_precision:.4f}</td><td>{avg_final_precision:.4f}</td><td>{avg_final_precision - avg_initial_precision:+.4f}</td></tr>
            <tr><td>Recall</td><td>{avg_initial_recall:.4f}</td><td>{avg_final_recall:.4f}</td><td>{avg_final_recall - avg_initial_recall:+.4f}</td></tr>
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
        <title>Segment Task Evaluation Log</title>
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
        default="/data/tct/ActivePerception/eval_logs/segment.html",
        help="Path to save the HTML log file"
    )
    
    args = parser.parse_args()
    
    # 初始化 ray
    ray.init(_temp_dir="/data/tct/ActivePerception/tmp", ignore_reinit_error=True)
    
    main(args)

