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
from typing import Dict, List, Optional

import requests
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)
import ray
from pycocotools import mask as mask_utils

# --- Custom Module Imports ---
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.utils.third_party import call_grounding_from_pil_raw
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_UNIFIED_COT_TEMPLATE
from agent_system.environments.env_package.habitat_sim.projection import habitat_projection as unified_habitat_projection
from agent_system.multi_turn_rollout.utils import process_image

# Note: habitat_projection is now imported from projection.py as unified_habitat_projection
# It returns (pred_task_types, pred_task_prompts, action_indices, valids)

class NumpyEncoder(json.JSONEncoder):
    """自定义 JSON Encoder，处理 numpy 数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

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

def reshape_bbox(bbox, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
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

def reshape_bbox_xxyy(bbox, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
    """
    (针对GroundingDINO的输出) 根据图像尺寸变化对bbox进行转换。

    参数:
        bbox (tuple): 原始bbox，格式为 (x1, y1, x2, y2)。
        original_size (tuple): 原始图像尺寸 (宽, 高)。
        new_size (tuple): 新图像尺寸 (宽, 高)。

    返回:
        tuple: 转换后的bbox，格式为 (x, x, y, y)。
    """
    if bbox is None:
        return (0, 0, 0, 0)
    # 计算宽度和高度的缩放比例
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    # 对bbox的坐标进行缩放
    x1, y1, x2, y2 = bbox
    x1_new = x1 * scale_x
    y1_new = y1 * scale_y
    x2_new = x2 * scale_x
    y2_new = y2 * scale_y

    return (x1_new, x2_new, y1_new, y2_new)

def calculate_iou(box2_gt, box1_pred):
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

def compute_ap(recalls, precisions):
    """
    计算 Average Precision (AP)
    使用所有点插值方法（VOC 2010+ 方式）
    
    参数:
        recalls: numpy array, 召回率序列
        precisions: numpy array, 精确率序列
    
    返回:
        ap: float, Average Precision
    """
    # 在开头和结尾添加哨兵值
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # 确保 precision 单调递减（从右向左取最大值）
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 找到 recall 变化的点
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    
    # 计算曲线下面积 (AP = sum of rectangular areas)
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap

def calculate_mAP(all_predictions, iou_threshold=0.5):
    """
    计算 mAP (mean Average Precision) for phrase grounding
    
    参数:
        all_predictions: list of dict, 每个 dict 包含:
            - 'gt_bbox': ground truth bbox in format (x, x, y, y) or similar
            - 'pred_bboxes': list of predicted bboxes, each in format [x1, y1, x2, y2]
            - 'pred_scores': list of scores corresponding to pred_bboxes
            - 'phrase': 短语文本（可选，用于调试）
        iou_threshold: IoU 阈值，默认 0.5
    
    返回:
        mAP: float, mean Average Precision
        详细统计: dict, 包含每个短语的 AP 和其他统计信息
    """
    total_ap = 0.0
    num_phrases = len(all_predictions)
    ap_list = []
    
    for idx, pred_data in enumerate(all_predictions):
        gt_bbox = pred_data['gt_bbox']
        pred_bboxes = pred_data['pred_bboxes']
        pred_scores = pred_data['pred_scores']
        
        if len(pred_bboxes) == 0:
            # 如果没有预测，AP = 0
            ap = 0.0
        else:
            # 按置信度降序排序
            sorted_indices = np.argsort(pred_scores)[::-1]
            sorted_bboxes = [pred_bboxes[i] for i in sorted_indices]
            sorted_scores = [pred_scores[i] for i in sorted_indices]
            
            # 计算每个预测框与 GT 的 IoU，判断 TP/FP
            tp = []
            fp = []
            matched = False  # 确保每个 GT 只匹配一次
            
            for bbox in sorted_bboxes:
                iou = calculate_iou(gt_bbox, bbox)
                if iou >= iou_threshold and not matched:
                    tp.append(1)
                    fp.append(0)
                    matched = True  # 第一个匹配后，后续都算 FP
                else:
                    tp.append(0)
                    fp.append(1)
            
            # 累积 TP 和 FP
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # 计算 Precision 和 Recall
            # 注意：每个短语只有1个 GT
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / 1.0  # N_GT = 1 for each phrase in grounding task
            
            # 计算 AP
            ap = compute_ap(recalls, precisions)
        
        ap_list.append(ap)
        total_ap += ap
    
    mAP = total_ap / num_phrases if num_phrases > 0 else 0.0
    
    stats = {
        'mAP': mAP,
        'ap_list': ap_list,
        'num_phrases': num_phrases,
        'iou_threshold': iou_threshold
    }
    
    return mAP, stats


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Rule-based policy functions (imported from collect_sft_dataset_rule.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import (
    POLICY_THRESHOLDS,
    get_rule_based_action,
    extract_bbox_from_mask,
    extract_bbox_from_3dbox
)

def draw_image_with_bboxes(image: Image.Image, bbox_gt=None, bbox_pred=None) -> Image.Image:
    """
    在图像上绘制 gt 和 pred 的 bbox。
    
    参数:
        image: PIL 图像
        bbox_gt: ground truth bbox，格式为 (xmin, xmax, ymin, ymax)
        bbox_pred: predicted bbox，格式为 (x1, y1, x2, y2)，会被转换为 (xmin, xmax, ymin, ymax)
    
    返回:
        标注后的 PIL 图像
    """
    result_image = image.copy()
    
    # 绘制 ground truth bbox (绿色)
    if bbox_gt is not None and bbox_gt != (0, 0, 0, 0):
        result_image = draw_bbox_with_text(result_image, bbox_gt, text="gt", color="red", width=3)
    
    # 绘制 predicted bbox (红色)
    if bbox_pred is not None and bbox_pred != (0, 0, 0, 0):
        result_image = draw_bbox_with_text(result_image, reshape_bbox_xxyy(bbox_pred), text="pred", color="green", width=3)
    
    return result_image


def main(args):
    """主评估逻辑函数"""
    # --- Configuration ---
    mode = args.mode
    model_path = args.model_path
    exp_name = args.exp_name
    
    # 创建实验输出目录
    exp_dir = os.path.join("/data/tct/ActivePerception/exp", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 日志文件路径
    txt_log_filename = os.path.join(exp_dir, "log.txt")
    
    # 创建记录目录
    record_dir = os.path.join(exp_dir, "task_records")
    os.makedirs(record_dir, exist_ok=True)
    
    print(f"Experiment output directory: {exp_dir}")

    dataset_name = "HM3D" if "HM3D" in exp_name else "ReplicaCAD"
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
    txt_log_entries = []
    all_initial_ious = []
    all_final_ious = []
    all_initial_confs = []
    all_final_confs = []
    all_predictions_initial = []  # 用于计算初始 mAP
    all_predictions_final = []    # 用于计算最终 mAP
    task_type_correct_indices = []  # 记录 task_type 预测正确的任务索引

    input_filename = args.input_filename
    input_file_path = f'/data/tct/habitat/eval_data/{input_filename}/task_infos.json'
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_task_infos = json.load(f)
        print(f"成功从 {input_file_path} 中加载了 {len(all_task_infos)} 条任务信息。\n")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}。")
        all_task_infos = []

    # ----------------- 主循环开始 -----------------
    total_tasks = len(all_task_infos)
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Tasks")):
        task_info["task_type"] = "grounding"
        obs, info = env.reset_eval(sync_info=task_info)

        response = call_grounding_from_pil_raw(obs, info.get("task_prompt"))
        pred_bboxes = response["bboxes"]
        print(f"pred_bboxes: {pred_bboxes}")
        pred_scores = response["scores"]
        
        # 收集初始预测数据用于 mAP 计算
        gt_bbox = info.get("gt")["bbox_gt"]
        all_predictions_initial.append({
            'gt_bbox': gt_bbox,
            'pred_bboxes': pred_bboxes,
            'pred_scores': pred_scores,
            'phrase': info.get("task_prompt")
        })
        
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        # 创建任务文件夹用于保存记录
        task_record_dir = os.path.join(record_dir, f"task_{idx:04d}")
        os.makedirs(task_record_dir, exist_ok=True)
        
        # 初始化任务步骤记录列表
        task_step_records = []
        
        # 保存初始状态图像（step_0）
        init_pred_bbox = pred_bboxes[int(np.argmax(pred_scores))] if pred_scores else None
        init_image_with_bbox = draw_image_with_bboxes(obs, bbox_gt=gt_bbox, bbox_pred=init_pred_bbox)
        init_image_path = os.path.join(task_record_dir, "step_000_init.png")
        init_image_with_bbox.save(init_image_path)
        
        is_task_type_correct = True
        done = False
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            print(f"Task: {info['task_prompt']}")
            
            prompt = build_text_obs([info])[0]
            
            # --- 根据模式选择动作 ---
            # Get ground truth task_type and task_prompt for non-model modes
            gt_task_type = info.get("task_type", "grounding")
            gt_task_prompt = info.get("task_prompt", "")
            
            if mode == 'model':
                # 模型推理模式：模型预测 task_type, task_prompt, action
                if "qwen" in model_type:
                    prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
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
            
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Pred Task Type: '{pred_task_types[0]}', Pred Task Prompt: '{pred_task_prompts[0][:50]}...'\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)

            # env.step now requires (pred_task_type, pred_task_prompt, action_index, is_valid_action)
            # NOTE: must env.step before calculate initial_iou and initial_conf
            obs, reward, done, info = env.step(
                pred_task_types[0], 
                pred_task_prompts[0], 
                actions[0], 
                valids[0]
            )

            # 记录第一次预测结果（仅在 k==0 时）
            if k == 0:
                is_task_type_correct = (pred_task_types[0] == gt_task_type)

                if is_task_type_correct:
                    task_type_correct_indices.append(idx)

                initial_iou = calculate_iou(gt_bbox, env.init_pred) if is_task_type_correct else 0.0
                initial_conf = env.init_conf_score if is_task_type_correct else 0.0
                all_initial_ious.append(initial_iou)
                all_initial_confs.append(initial_conf)

            if not is_task_type_correct:
                print("Task type incorrect, skipping step")
                # 记录该步信息（即使 task_type 错误）
                step_record = {
                    "step": k + 1,
                    "model_output": text_actions[0],
                    "pred_task_type": pred_task_types[0],
                    "pred_task_prompt": pred_task_prompts[0],
                    "gt_task_type": gt_task_type,
                    "gt_task_prompt": gt_task_prompt,
                    "action": action_name if 'action_name' in dir() else "N/A",
                    "is_valid": bool(valids[0]),
                    "is_task_type_correct": False,
                    "skipped": True
                }
                task_step_records.append(step_record)
                break
            
            # 获取当前步的 pred bbox
            current_pred_bbox = info.get("pred")
            gt_info = info.get("gt", None)
            if gt_info is not None:
                current_gt_bbox = gt_info.get("bbox_gt")
            else:
                current_gt_bbox = None
            
            # 保存当前步的图像（标注 bbox）
            step_image_with_bbox = draw_image_with_bboxes(obs, bbox_gt=current_gt_bbox, bbox_pred=current_pred_bbox)
            step_image_path = os.path.join(task_record_dir, f"step_{k+1:03d}.png")
            step_image_with_bbox.save(step_image_path)
            
            # 记录该步的详细信息
            step_record = {
                "step": k + 1,
                "model_output": text_actions[0],
                "pred_task_type": pred_task_types[0],
                "pred_task_prompt": pred_task_prompts[0],
                "gt_task_type": gt_task_type,
                "gt_task_prompt": gt_task_prompt,
                "action": action_name,
                "is_valid": bool(valids[0]),
                "is_task_type_correct": is_task_type_correct,
                "reward": reward if done else None,
                "done": done,
                "pred_bbox": current_pred_bbox,
                "gt_bbox": current_gt_bbox,
                "conf_score": info.get("conf_score", 0.0),
                "image_path": f"step_{k+1:03d}.png"
            }
            task_step_records.append(step_record)

            if done:
                final_log_str = (
                    f"\n---------- FINAL STEP (Episode End) ----------\n"
                    f"Task: {info['task_prompt']}\n"
                    f"Model Output: {text_actions[0]}\n"
                    f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})\n"
                    f"Reward: {reward}"
                )
                print(final_log_str)
                txt_log_entries.append(final_log_str)
                break
        
        response = call_grounding_from_pil_raw(obs, info.get("task_prompt"))
        pred_bboxes = response["bboxes"]
        pred_scores = response["scores"]
        
        # 如果 task_type 预测错：final AP 直接置 0（通过“无预测框”实现）
        if not is_task_type_correct:
            pred_bboxes = []
            pred_scores = []
        # 收集最终预测数据用于 mAP 计算
        all_predictions_final.append({
            'gt_bbox': info.get("gt")["bbox_gt"],
            'pred_bboxes': pred_bboxes,
            'pred_scores': pred_scores,
            'phrase': info.get("task_prompt")
        })
        
        final_iou = calculate_iou(info.get("gt")["bbox_gt"], info.get("pred")) if is_task_type_correct else 0.0
        final_conf = info.get("conf_score", 0.0) if is_task_type_correct else 0.0
        all_final_ious.append(final_iou)
        all_final_confs.append(final_conf)

        # 计算指标变化
        iou_change = final_iou - initial_iou
        conf_change = final_conf - initial_conf
        
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"IoU: {initial_iou:.4f} → {final_iou:.4f} ({iou_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"IoU: {initial_iou:.4f} → {final_iou:.4f} ({iou_change:+.4f})\n"
                f"Conf: {initial_conf:.4f} → {final_conf:.4f} ({conf_change:+.4f})"
            )
        
        print(episode_finish_str)
        txt_log_entries.append(episode_finish_str + "\n")
        
        # 保存该任务的完整记录到 JSON 文件
        task_record = {
            "task_id": task_info.get("task_id", idx),
            "task_idx": idx,
            "scene_id": info.get("scene_id", "N/A"),
            "task_prompt": info.get("task_prompt", "N/A"),
            "gt_task_type": gt_task_type,
            "is_task_type_correct": is_task_type_correct,
            "initial_iou": initial_iou,
            "final_iou": final_iou,
            "iou_change": iou_change,
            "initial_conf": initial_conf,
            "final_conf": final_conf,
            "conf_change": conf_change,
            "total_steps": k + 1,
            "done": done,
            "steps": task_step_records
        }
        task_record_json_path = os.path.join(task_record_dir, "task_record.json")
        with open(task_record_json_path, 'w', encoding='utf-8') as f:
            json.dump(task_record, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    print("\nEvaluation loop completed.")

    # --- 计算平均值并生成报告 ---
    avg_initial_iou = np.mean(all_initial_ious) if all_initial_ious else 0.0
    avg_final_iou = np.mean(all_final_ious) if all_final_ious else 0.0
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0

    # --- 计算 task_type 预测正确率 ---
    num_task_type_correct = len(task_type_correct_indices)
    task_type_accuracy = num_task_type_correct / total_tasks if total_tasks > 0 else 0.0
    print(f"\nTask Type Accuracy: {num_task_type_correct}/{total_tasks} = {task_type_accuracy:.4f}")
    
    # --- 计算 mAP ---
    print("\nCalculating mAP at multiple IoU thresholds...")
    iou_thresholds = [0.5, 0.75, 0.90, 0.95]
    initial_mAPs = {}
    final_mAPs = {}
    
    for threshold in iou_thresholds:
        print(f"  Computing mAP@{threshold}...")
        initial_mAP, _ = calculate_mAP(all_predictions_initial, iou_threshold=threshold)
        final_mAP, _ = calculate_mAP(all_predictions_final, iou_threshold=threshold)
        initial_mAPs[threshold] = initial_mAP
        final_mAPs[threshold] = final_mAP
        print(f"    Initial mAP@{threshold}: {initial_mAP:.4f}")
        print(f"    Final mAP@{threshold}:   {final_mAP:.4f}")
    
    # 计算 mAP@[0.5:0.95] (平均值)
    avg_initial_mAP = np.mean(list(initial_mAPs.values()))
    avg_final_mAP = np.mean(list(final_mAPs.values()))

    def calculate_improvement(initial, final):
        if initial > 0:
            percent = ((final - initial) / initial) * 100
            return f"{percent:+.2f}%"
        return "N/A (初始值为0)"

    iou_improvement_str = calculate_improvement(avg_initial_iou, avg_final_iou)
    conf_improvement_str = calculate_improvement(avg_initial_conf, avg_final_conf)
    
    summary_header = "\n--- Evaluation Summary ---"
    summary_body = (
        f"Average Initial IoU:    {avg_initial_iou:.4f}\n"
        f"Average Final IoU:      {avg_final_iou:.4f} ({iou_improvement_str})\n"
        f"Average Initial Conf:   {avg_initial_conf:.4f}\n"
        f"Average Final Conf:     {avg_final_conf:.4f} ({conf_improvement_str})\n"
        f"\n"
        f"Task Type Prediction:\n"
        f"  Accuracy: {num_task_type_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)\n"
        f"\n"
        f"mAP Results:\n"
    )
    
    for threshold in iou_thresholds:
        improvement_str = calculate_improvement(initial_mAPs[threshold], final_mAPs[threshold])
        summary_body += (
            f"  Initial mAP@{threshold}:      {initial_mAPs[threshold]:.4f}\n"
            f"  Final mAP@{threshold}:        {final_mAPs[threshold]:.4f} ({improvement_str})\n"
        )
    
    avg_mAP_improvement_str = calculate_improvement(avg_initial_mAP, avg_final_mAP)
    summary_body += (
        f"\n"
        f"  Initial mAP@[0.5:0.95]: {avg_initial_mAP:.4f}\n"
        f"  Final mAP@[0.5:0.95]:   {avg_final_mAP:.4f} ({avg_mAP_improvement_str})"
    )
    
    print(summary_header)
    print(summary_body)
    txt_log_entries.append(summary_header + "\n" + summary_body)

    
    # --- 保存 all_predictions_initial 和 all_predictions_final 到 JSON 文件 ---
    predictions_initial_path = os.path.join(record_dir, "all_predictions_initial.json")
    with open(predictions_initial_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_initial, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ Initial predictions saved to '{predictions_initial_path}'")
    
    predictions_final_path = os.path.join(record_dir, "all_predictions_final.json")
    with open(predictions_final_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_final, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ Final predictions saved to '{predictions_final_path}'")

    # --- 写入 TXT 文件 ---
    full_txt_content = "\n".join(txt_log_entries)
    with open(txt_log_filename, 'w', encoding='utf-8') as f:
        f.write(full_txt_content)
    print(f"\n✅ Text log successfully saved to '{txt_log_filename}'")


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run Habitat environment evaluation with different action selection modes.")
    parser.add_argument('--mode', type=str, choices=['model', 'random', 'rule', 'forward'], default='model',
                        help='Action selection mode: "model" for VL model inference, "random" for random actions, "rule" for rule-based policy.')
    parser.add_argument('--model_path', type=str, default='/data/tct/models/RL/InternVL3_5-2B-unified-tasks-rl-HM3D', 
                        help='Path to the VL model checkpoint directory. Required when --mode=model.')
    parser.add_argument('--input_filename', type=str, 
                        default='hm3d_10-any-500-seen',
                        help='Path to the input_filename.')
    parser.add_argument('--exp_name', type=str, default='HM3D-grounding-SFT_GRPO',
                        help='Experiment name. All outputs will be saved to /data/tct/ActivePerception/exp/{exp_name}/')
    
    # 解析参数并运行主函数
    args = parser.parse_args()

    if not ray.is_initialized():
        ray.init(_temp_dir="/data/tct/ActivePerception/tmp",)

    main(args)

