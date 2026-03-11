import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)
import ray

# --- Custom Module Imports ---
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_VISUAL_CLASS_DETECT_COT_TEMPLATE
from agent_system.multi_turn_rollout.utils import process_image

# ========================================
# Class Detection Evaluation Metrics
# ========================================

def calculate_bbox_iou(bbox1, bbox2):
    """
    计算两个 bounding box 的 IoU
    
    Args:
        bbox1, bbox2: (x1, y1, x2, y2) format
        
    Returns:
        IoU score (float)
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_detections_to_gt(pred_detections: List[Dict], gt_detections: List[Dict], iou_threshold: float = 0.5):
    """
    将预测的检测结果与 GT 进行匹配
    
    Args:
        pred_detections: 预测的检测列表，每个元素包含 {bbox, score, category}
        gt_detections: GT 检测列表，每个元素包含 {bbox, category}
        iou_threshold: IoU 阈值，用于判断是否匹配
        
    Returns:
        matches: List of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: List of pred indices that were not matched
        unmatched_gts: List of gt indices that were not matched
    """
    if not pred_detections or not gt_detections:
        return [], list(range(len(pred_detections))), list(range(len(gt_detections)))
    
    # 计算所有预测和 GT 之间的 IoU
    iou_matrix = np.zeros((len(pred_detections), len(gt_detections)))
    for i, pred in enumerate(pred_detections):
        pred_bbox = pred['bbox']
        for j, gt in enumerate(gt_detections):
            gt_bbox = gt['bbox']
            iou_matrix[i, j] = calculate_bbox_iou(pred_bbox, gt_bbox)
    
    matches = []
    unmatched_preds = set(range(len(pred_detections)))
    unmatched_gts = set(range(len(gt_detections)))
    
    # 贪心匹配：从最高 IoU 开始匹配
    while True:
        if iou_matrix.size == 0:
            break
        
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        
        pred_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches.append((pred_idx, gt_idx, max_iou))
        
        unmatched_preds.discard(pred_idx)
        unmatched_gts.discard(gt_idx)
        
        # 移除已匹配的行和列
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0
    
    return matches, list(unmatched_preds), list(unmatched_gts)

def calculate_precision_recall_f1(pred_detections: List[Dict], gt_detections: List[Dict], iou_threshold: float = 0.5):
    """
    计算 Precision, Recall, F1-Score
    
    Args:
        pred_detections: 预测的检测列表
        gt_detections: GT 检测列表
        iou_threshold: IoU 阈值
        
    Returns:
        precision, recall, f1_score
    """
    if not pred_detections and not gt_detections:
        return 1.0, 1.0, 1.0  # 都为空，完美匹配
    
    if not pred_detections:
        return 0.0, 0.0, 0.0  # 没有预测，recall=0
    
    if not gt_detections:
        return 0.0, 0.0, 0.0  # 没有GT，precision=0
    
    matches, unmatched_preds, unmatched_gts = match_detections_to_gt(
        pred_detections, gt_detections, iou_threshold
    )
    
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def calculate_average_precision(pred_detections: List[Dict], gt_detections: List[Dict], iou_threshold: float = 0.5):
    """
    计算 Average Precision (AP)
    
    AP 是在不同置信度阈值下的 precision-recall 曲线下面积
    """
    if not gt_detections:
        return 0.0 if pred_detections else 1.0
    
    if not pred_detections:
        return 0.0
    
    # 按置信度降序排序
    sorted_preds = sorted(pred_detections, key=lambda x: x['score'], reverse=True)
    
    # 标记每个预测是否为 TP
    tp_flags = []
    matched_gt_indices = set()
    
    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_detections):
            if gt_idx in matched_gt_indices:
                continue
            
            iou = calculate_bbox_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp_flags.append(True)
            matched_gt_indices.add(best_gt_idx)
        else:
            tp_flags.append(False)
    
    # 计算累积的 TP 和 FP
    cumulative_tp = np.cumsum(tp_flags)
    cumulative_fp = np.cumsum([not flag for flag in tp_flags])
    
    # 计算 precision 和 recall
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp)
    recalls = cumulative_tp / len(gt_detections)
    
    # 计算 AP (使用 11-point interpolation)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precision_at_recall = precisions[recalls >= t]
        if len(precision_at_recall) > 0:
            ap += precision_at_recall.max()
    ap /= 11
    
    return ap

def calculate_detection_metrics(pred_detections: List[Dict], gt_detections: List[Dict], iou_threshold: float = 0.5):
    """
    综合计算所有检测指标
    
    Returns:
        metrics: Dict containing all metrics
    """
    precision, recall, f1 = calculate_precision_recall_f1(pred_detections, gt_detections, iou_threshold)
    ap = calculate_average_precision(pred_detections, gt_detections, iou_threshold)
    
    # 计算平均置信度
    avg_confidence = np.mean([d['score'] for d in pred_detections]) if pred_detections else 0.0
    
    # 计算检测数量
    num_pred = len(pred_detections)
    num_gt = len(gt_detections)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'average_precision': ap,
        'avg_confidence': avg_confidence,
        'num_detections': num_pred,
        'num_gt': num_gt
    }
    
    return metrics

# ========================================
# Visualization Helpers
# ========================================

def draw_multiple_bboxes_on_image(image: Image.Image, detections: List[Dict], color_map: Dict = None, title: str = None):
    """
    在图像上绘制多个检测框
    
    Args:
        image: PIL Image
        detections: List of {bbox: (x1, y1, x2, y2), score: float, category: str}
        color_map: Dict mapping category to color (default: green for all)
        title: Optional title text
        
    Returns:
        PIL Image with bboxes drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 默认颜色
    default_color = "green"
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        score = det.get('score', 0.0)
        category = det.get('category', 'unknown')
        
        x1, y1, x2, y2 = bbox
        color = color_map.get(category, default_color) if color_map else default_color
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签
        label = f"{category}: {score:.2f}"
        
        # 计算标签背景大小
        bbox_text = draw.textbbox((x1, y1), label, font=small_font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制标签背景
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        
        # 绘制标签文本
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=small_font)
    
    # 绘制标题
    if title:
        draw.text((10, 10), title, fill="white", font=font, stroke_width=2, stroke_fill="black")
    
    return img

def visualize_detection_result(image: Image.Image, pred_detections: List[Dict], gt_detections: List[Dict], task_prompt: str):
    """
    可视化检测结果：同时显示预测和GT
    
    Args:
        image: 原始图像
        pred_detections: 预测的检测列表
        gt_detections: GT 检测列表
        task_prompt: 任务描述
        
    Returns:
        PIL Image with visualization
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 绘制 GT (红色虚线)
    for gt in gt_detections:
        bbox = gt['bbox']
        x1, y1, x2, y2 = bbox
        
        # 绘制虚线矩形（通过绘制多个短线段实现）
        dash_length = 10
        for x in range(int(x1), int(x2), dash_length * 2):
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill="red", width=2)
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill="red", width=2)
        for y in range(int(y1), int(y2), dash_length * 2):
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill="red", width=2)
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill="red", width=2)
        
        # 绘制 GT 标签
        label = "GT"
        draw.text((x1 + 2, y2 + 2), label, fill="red", font=small_font, stroke_width=1, stroke_fill="white")
    
    # 绘制预测 (绿色实线)
    for pred in pred_detections:
        bbox = pred['bbox']
        score = pred.get('score', 0.0)
        x1, y1, x2, y2 = bbox
        
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # 绘制预测标签
        label = f"Pred: {score:.2f}"
        bbox_text = draw.textbbox((x1, y1), label, font=small_font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="green")
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=small_font)
    
    # 绘制标题
    if task_prompt:
        title = f"Task: Detect all '{task_prompt}'"
        draw.text((10, 10), title, fill="white", font=font, stroke_width=2, stroke_fill="black")
    
    return img

# ========================================
# Model Inference & Projection
# ========================================

def habitat_projection(text_actions: List[str], env_name: str) -> (List[int], List[int]):
    """Projects the model's free-form text output to a discrete action index."""
    output_indices = []
    valids = []
    action_list = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]
    
    for string in text_actions:
        if not isinstance(string, str):
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
            continue
        
        string = string.lower()
        action_index = string.find('"action":')
        string = string[action_index:]
        contained_actions = []
        
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        
        contained_actions = list(set(contained_actions))
        
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            output_indices.append(action_list.index(contained_actions[0]))
            valids.append(1)
        else:
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
            
    return output_indices, valids

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
    """Builds the text observation (prompt) for the agent - CLASS-DETECT task specific."""
    postprocess_text_obs = []
    for i in range(len(infos)):
        prompt = HABITAT_VISUAL_CLASS_DETECT_COT_TEMPLATE.format(
            task_caption=infos[i]['task_prompt'],
            conf_score=infos[i]['conf_score']
        )
        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def image_to_base64_str(obs, pred_detections=None, gt_detections=None, task_prompt=None):
    """将图像转换为Base64编码的字符串，并可选择叠加检测结果"""
    if not isinstance(obs, Image.Image):
        img = Image.fromarray(obs)
    else:
        img = obs.copy()
    
    # 可视化检测结果
    if pred_detections is not None or gt_detections is not None:
        img = visualize_detection_result(img, pred_detections or [], gt_detections or [], task_prompt)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

# ========================================
# GT Generation Helper
# ========================================

def get_gt_detections_from_env(env) -> List[Dict]:
    """
    从环境中获取所有同类物体的 GT 检测信息（用于 class-detect 任务）
    
    Args:
        env: Habitat environment instance
        
    Returns:
        List of GT detections, each containing:
            - bbox: (x1, y1, x2, y2) format
            - category: object category
            - mask_gt: RLE format mask (optional)
            - instance_id: instance ID (optional)
    """
    try:
        _, gt_list = env.get_gt_for_class()
        
        if not gt_list:
            return []
        
        # 转换 GT 格式为检测格式
        gt_detections = []
        for gt_item in gt_list:
            bbox_gt = gt_item.get('bbox_gt')
            if bbox_gt is None:
                continue
            
            # bbox_gt 已经是 (x1, y1, x2, y2) 格式
            gt_det = {
                'bbox': bbox_gt,
                'category': gt_item.get('category', 'unknown'),
            }
            
            # 可选：保留 mask 和 instance_id 信息
            if 'mask_gt' in gt_item:
                gt_det['mask_gt'] = gt_item['mask_gt']
            if 'instance_id' in gt_item:
                gt_det['instance_id'] = gt_item['instance_id']
            
            gt_detections.append(gt_det)
        
        return gt_detections
        
    except Exception as e:
        print(f"Error getting GT from environment: {e}")
        # Fallback: 从 info 中获取单个 GT（向后兼容）
        return []

def convert_pred_to_detection_list(pred) -> List[Dict]:
    """
    将 pred 格式转换为标准的检测列表格式
    
    pred 格式: List[{category, score, bbox}]
    返回格式: List[{category, score, bbox: (x1, y1, x2, y2)}]
    """
    if not pred or pred[0].get('category') == 'unknown':
        return []
    
    detections = []
    for item in pred:
        bbox = item['bbox']
        # bbox 格式应该已经是 (x1, y1, x2, y2)
        detections.append({
            'category': item['category'],
            'score': item['score'],
            'bbox': bbox
        })
    
    return detections

# ========================================
# Main Evaluation Logic
# ========================================

def main(args):
    """主评估逻辑函数 - 专门用于 Class-Detect 任务"""
    
    # --- Configuration ---
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
    iou_threshold = 0.5  # IoU threshold for matching

    # --- Load Model and Processor ---
    model, processor, model_type = load_model_and_processor(model_path)
    print(f"Model and processor loaded successfully. Model type: {model_type}")

    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    projection_f = partial(habitat_projection, env_name='habitat')
    print("Habitat environment created.")

    # --- 初始化列表用于存储评估指标和日志 ---
    html_log_entries = []
    txt_log_entries = []
    
    # Class-detect specific metrics
    all_initial_precisions = []
    all_final_precisions = []
    all_initial_recalls = []
    all_final_recalls = []
    all_initial_f1_scores = []
    all_final_f1_scores = []
    all_initial_aps = []
    all_final_aps = []
    all_initial_num_dets = []
    all_final_num_dets = []
    all_initial_confs = []
    all_final_confs = []

    input_filename = '/data1/tct_data/habitat/eval_data/replicacad_10-class-detect/task_infos.json'
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            all_task_infos = json.load(f)
        print(f"成功从 {input_filename} 中加载了 {len(all_task_infos)} 条任务信息。\n")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_filename}。请先生成测试任务文件。")
        all_task_infos = []
        return

    # ----------------- 主循环开始 -----------------
    total_tasks = len(all_task_infos)
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Class-Detect Tasks")):
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测
        initial_pred = info.get("pred", [])
        initial_pred_detections = convert_pred_to_detection_list(initial_pred)
        
        # 获取 GT 信息（从环境中获取所有同类物体）
        gt_detections = get_gt_detections_from_env(env)
        
        # 计算初始指标
        initial_metrics = calculate_detection_metrics(initial_pred_detections, gt_detections, iou_threshold)
        
        all_initial_precisions.append(initial_metrics['precision'])
        all_initial_recalls.append(initial_metrics['recall'])
        all_initial_f1_scores.append(initial_metrics['f1_score'])
        all_initial_aps.append(initial_metrics['average_precision'])
        all_initial_num_dets.append(initial_metrics['num_detections'])
        all_initial_confs.append(initial_metrics['avg_confidence'])
        
        # 记录开始信息
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: Detect all '{info.get('task_prompt', 'N/A')}'\n"
            f"Initial Metrics - Precision: {initial_metrics['precision']:.4f}, Recall: {initial_metrics['recall']:.4f}, "
            f"F1: {initial_metrics['f1_score']:.4f}, AP: {initial_metrics['average_precision']:.4f}, "
            f"Detections: {initial_metrics['num_detections']}/{initial_metrics['num_gt']}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        episode_header_html = f"""
        <h2>--- Episode Start: Task {idx+1}/{total_tasks} (Scene: {info.get('scene_id', 'N/A')}) ---</h2>
        <p><b>Task Prompt:</b> Detect all '{info.get('task_prompt', 'N/A')}'</p>
        <p><b>Initial Metrics:</b> Precision: {initial_metrics['precision']:.4f}, Recall: {initial_metrics['recall']:.4f}, 
        F1: {initial_metrics['f1_score']:.4f}, AP: {initial_metrics['average_precision']:.4f}, 
        Detections: {initial_metrics['num_detections']}/{initial_metrics['num_gt']}</p>
        """
        html_log_entries.append(episode_header_html)
        
        done = False
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            print(f"Task: Detect all '{info['task_prompt']}'")
            
            prompt = build_text_obs([info])[0]
            # For Qwen models, replace <image> with special tokens
            if "qwen" in model_type:
                prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            
            text_actions = inference(model, processor, process_image(obs), prompt, model_type)
            actions, valids = projection_f(text_actions, env_name="habitat")
            action_name = env.action_space[actions[0]]
            
            # 记录步骤信息
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)

            # 可视化当前检测结果
            current_pred = info.get("pred", [])
            current_pred_detections = convert_pred_to_detection_list(current_pred)
            base64_image_pred = image_to_base64_str(obs, current_pred_detections, gt_detections, info.get('task_prompt'))
            
            step_html = f"""
            <div class="step-container">
                <h3>STEP {k+1}/{max_step_length}</h3>
                <p><b>Task:</b> Detect all '{info['task_prompt']}'</p>
                <p><b>Current Detections:</b> {len(current_pred_detections)} objects detected</p>
                <img src="{base64_image_pred}" alt="Agent view at step {k+1}" class="agent-view" style="max-width: 800px;">
                <div class="model-output">
                    <p><b>Model Output:</b> <code>{text_actions[0]}</code></p>
                    <p><b>Predicted Action:</b> '{action_name}' (Valid: {bool(valids[0])})</p>
                </div>
            </div>
            """
            html_log_entries.append(step_html)
            
            obs, reward, done, info = env.step(actions[0], valids[0])

            if done:
                # 最终步骤的图像
                final_pred = info.get("pred", [])
                final_pred_detections = convert_pred_to_detection_list(final_pred)
                # 重新获取 GT（因为可能视角变化导致可见物体变化）
                gt_detections = get_gt_detections_from_env(env)
                final_base64_image_pred = image_to_base64_str(obs, final_pred_detections, gt_detections, info.get('task_prompt'))
                
                final_step_html = f"""
                <div class="step-container">
                    <h3>FINAL STEP (Episode End)</h3>
                    <p><b>Task:</b> Detect all '{info['task_prompt']}'</p>
                    <p><b>Final Detections:</b> {len(final_pred_detections)} objects detected</p>
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
        final_metrics = calculate_detection_metrics(final_pred_detections, gt_detections, iou_threshold)
        
        all_final_precisions.append(final_metrics['precision'])
        all_final_recalls.append(final_metrics['recall'])
        all_final_f1_scores.append(final_metrics['f1_score'])
        all_final_aps.append(final_metrics['average_precision'])
        all_final_num_dets.append(final_metrics['num_detections'])
        all_final_confs.append(final_metrics['avg_confidence'])

        # 计算指标变化
        precision_change = final_metrics['precision'] - initial_metrics['precision']
        recall_change = final_metrics['recall'] - initial_metrics['recall']
        f1_change = final_metrics['f1_score'] - initial_metrics['f1_score']
        ap_change = final_metrics['average_precision'] - initial_metrics['average_precision']
        conf_change = final_metrics['avg_confidence'] - initial_metrics['avg_confidence']
        
        # 文本日志格式
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"Precision: {initial_metrics['precision']:.4f} → {final_metrics['precision']:.4f} ({precision_change:+.4f})\n"
                f"Recall: {initial_metrics['recall']:.4f} → {final_metrics['recall']:.4f} ({recall_change:+.4f})\n"
                f"F1-Score: {initial_metrics['f1_score']:.4f} → {final_metrics['f1_score']:.4f} ({f1_change:+.4f})\n"
                f"AP: {initial_metrics['average_precision']:.4f} → {final_metrics['average_precision']:.4f} ({ap_change:+.4f})\n"
                f"Detections: {initial_metrics['num_detections']}/{initial_metrics['num_gt']} → {final_metrics['num_detections']}/{final_metrics['num_gt']}\n"
                f"Avg Conf: {initial_metrics['avg_confidence']:.4f} → {final_metrics['avg_confidence']:.4f} ({conf_change:+.4f})"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"Precision: {initial_metrics['precision']:.4f} → {final_metrics['precision']:.4f} ({precision_change:+.4f})\n"
                f"Recall: {initial_metrics['recall']:.4f} → {final_metrics['recall']:.4f} ({recall_change:+.4f})\n"
                f"F1-Score: {initial_metrics['f1_score']:.4f} → {final_metrics['f1_score']:.4f} ({f1_change:+.4f})\n"
                f"AP: {initial_metrics['average_precision']:.4f} → {final_metrics['average_precision']:.4f} ({ap_change:+.4f})\n"
                f"Detections: {initial_metrics['num_detections']}/{initial_metrics['num_gt']} → {final_metrics['num_detections']}/{final_metrics['num_gt']}\n"
                f"Avg Conf: {initial_metrics['avg_confidence']:.4f} → {final_metrics['avg_confidence']:.4f} ({conf_change:+.4f})"
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
                    <p><b>Precision:</b> {initial_metrics['precision']:.4f} → <span style="font-weight: bold;">{final_metrics['precision']:.4f}</span> ({get_colored_change(precision_change)})</p>
                    <p><b>Recall:</b> {initial_metrics['recall']:.4f} → <span style="font-weight: bold;">{final_metrics['recall']:.4f}</span> ({get_colored_change(recall_change)})</p>
                    <p><b>F1-Score:</b> {initial_metrics['f1_score']:.4f} → <span style="font-weight: bold;">{final_metrics['f1_score']:.4f}</span> ({get_colored_change(f1_change)})</p>
                    <p><b>AP:</b> {initial_metrics['average_precision']:.4f} → <span style="font-weight: bold;">{final_metrics['average_precision']:.4f}</span> ({get_colored_change(ap_change)})</p>
                    <p><b>Detections:</b> {initial_metrics['num_detections']}/{initial_metrics['num_gt']} → <span style="font-weight: bold;">{final_metrics['num_detections']}/{final_metrics['num_gt']}</span></p>
                    <p><b>Avg Conf:</b> {initial_metrics['avg_confidence']:.4f} → <span style="font-weight: bold;">{final_metrics['avg_confidence']:.4f}</span> ({get_colored_change(conf_change)})</p>
                </div>
            </div>
            <hr>
            """
        else:
            finish_html = f"""
            <div class="episode-summary">
                <h3>Episode timed out at step {k+1}</h3>
                <div class="metrics-comparison">
                    <p><b>Precision:</b> {initial_metrics['precision']:.4f} → <span style="font-weight: bold;">{final_metrics['precision']:.4f}</span> ({get_colored_change(precision_change)})</p>
                    <p><b>Recall:</b> {initial_metrics['recall']:.4f} → <span style="font-weight: bold;">{final_metrics['recall']:.4f}</span> ({get_colored_change(recall_change)})</p>
                    <p><b>F1-Score:</b> {initial_metrics['f1_score']:.4f} → <span style="font-weight: bold;">{final_metrics['f1_score']:.4f}</span> ({get_colored_change(f1_change)})</p>
                    <p><b>AP:</b> {initial_metrics['average_precision']:.4f} → <span style="font-weight: bold;">{final_metrics['average_precision']:.4f}</span> ({get_colored_change(ap_change)})</p>
                    <p><b>Detections:</b> {initial_metrics['num_detections']}/{initial_metrics['num_gt']} → <span style="font-weight: bold;">{final_metrics['num_detections']}/{final_metrics['num_gt']}</span></p>
                    <p><b>Avg Conf:</b> {initial_metrics['avg_confidence']:.4f} → <span style="font-weight: bold;">{final_metrics['avg_confidence']:.4f}</span> ({get_colored_change(conf_change)})</p>
                </div>
            </div>
            <hr>
            """
        
        html_log_entries.append(finish_html)

    # ----------------- 计算并打印总体统计信息 -----------------
    avg_initial_precision = np.mean(all_initial_precisions)
    avg_final_precision = np.mean(all_final_precisions)
    avg_initial_recall = np.mean(all_initial_recalls)
    avg_final_recall = np.mean(all_final_recalls)
    avg_initial_f1 = np.mean(all_initial_f1_scores)
    avg_final_f1 = np.mean(all_final_f1_scores)
    avg_initial_ap = np.mean(all_initial_aps)
    avg_final_ap = np.mean(all_final_aps)
    avg_initial_num_dets = np.mean(all_initial_num_dets)
    avg_final_num_dets = np.mean(all_final_num_dets)
    avg_initial_conf = np.mean(all_initial_confs)
    avg_final_conf = np.mean(all_final_confs)

    summary_str = f"""
========================================
  Class-Detect Task Evaluation Summary
========================================
Total Tasks: {total_tasks}
IoU Threshold: {iou_threshold}

Precision:
  Initial: {avg_initial_precision:.4f}
  Final:   {avg_final_precision:.4f}
  Change:  {avg_final_precision - avg_initial_precision:+.4f}

Recall:
  Initial: {avg_initial_recall:.4f}
  Final:   {avg_final_recall:.4f}
  Change:  {avg_final_recall - avg_initial_recall:+.4f}

F1-Score:
  Initial: {avg_initial_f1:.4f}
  Final:   {avg_final_f1:.4f}
  Change:  {avg_final_f1 - avg_initial_f1:+.4f}

Average Precision (AP):
  Initial: {avg_initial_ap:.4f}
  Final:   {avg_final_ap:.4f}
  Change:  {avg_final_ap - avg_initial_ap:+.4f}

Number of Detections:
  Initial: {avg_initial_num_dets:.2f}
  Final:   {avg_final_num_dets:.2f}
  Change:  {avg_final_num_dets - avg_initial_num_dets:+.2f}

Confidence Score:
  Initial: {avg_initial_conf:.4f}
  Final:   {avg_final_conf:.4f}
  Change:  {avg_final_conf - avg_initial_conf:+.4f}
========================================
"""
    
    print(summary_str)
    txt_log_entries.append(summary_str)

    # 保存文本日志
    with open(txt_log_filename, "w", encoding="utf-8") as f:
        f.write("".join(txt_log_entries))
    print(f"文本日志已保存到: {txt_log_filename}")

    # 生成并保存 HTML 日志
    html_summary = f"""
    <div class="summary">
        <h2>Class-Detect Task Evaluation Summary</h2>
        <p><b>Total Tasks:</b> {total_tasks}</p>
        <p><b>IoU Threshold:</b> {iou_threshold}</p>
        <table>
            <tr><th>Metric</th><th>Initial</th><th>Final</th><th>Change</th></tr>
            <tr><td>Precision</td><td>{avg_initial_precision:.4f}</td><td>{avg_final_precision:.4f}</td><td>{avg_final_precision - avg_initial_precision:+.4f}</td></tr>
            <tr><td>Recall</td><td>{avg_initial_recall:.4f}</td><td>{avg_final_recall:.4f}</td><td>{avg_final_recall - avg_initial_recall:+.4f}</td></tr>
            <tr><td>F1-Score</td><td>{avg_initial_f1:.4f}</td><td>{avg_final_f1:.4f}</td><td>{avg_final_f1 - avg_initial_f1:+.4f}</td></tr>
            <tr><td>Average Precision (AP)</td><td>{avg_initial_ap:.4f}</td><td>{avg_final_ap:.4f}</td><td>{avg_final_ap - avg_initial_ap:+.4f}</td></tr>
            <tr><td>Num Detections</td><td>{avg_initial_num_dets:.2f}</td><td>{avg_final_num_dets:.2f}</td><td>{avg_final_num_dets - avg_initial_num_dets:+.2f}</td></tr>
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
        <title>Class-Detect Task Evaluation Log</title>
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
    parser = argparse.ArgumentParser(description="Evaluate VLM on Habitat Class-Detect tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--log_filename",
        type=str,
        default="/data1/tct_data/habitat/eval_logs/class_detect_eval_log.html",
        help="Path to save the HTML log file"
    )
    
    args = parser.parse_args()
    
    # 初始化 ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    main(args)

