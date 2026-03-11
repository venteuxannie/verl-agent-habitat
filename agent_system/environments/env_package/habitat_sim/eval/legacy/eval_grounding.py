import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import random
import time
import base64
import io
import json
import argparse  # 1.【新增】导入 argparse 模块
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

# For visualization in the notebook
# from IPython.display import display, clear_output

# --- Custom Module Imports ---
# Note: Ensure these modules are in your Python path or the same directory.
# sys.path.append("/data1/tct_data/verl-agent/agent_system/environments/env_package/habitat_sim")
# from utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_VISUAL_GROUNDING_COT_TEMPLATE
from agent_system.multi_turn_rollout.utils import process_image

# (此处所有辅助函数 habitat_projection, load_model_and_processor, inference, get_scene_path, build_text_obs,
# image_to_base64_str, reshape_bbox, calculate_iou 保持不变，为了简洁在此省略)

def habitat_projection(text_actions: List[str], env_name: str) -> (List[int], List[int]):
    """Projects the model's free-form text output to a discrete action index."""
    output_indices = []
    valids = []
    if env_name == 'habitat':
        action_list = ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "stop"]
    elif env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError(f"Action list not implemented for this env: {env_name}!")
    
    for string in text_actions:
        if not isinstance(string, str):
            output_indices.append(random.randint(0, len(action_list) - 1))
            valids.append(0)
            continue
        
        string = string.lower()
        action_index = string.find('"action":')
        string = string[action_index:]
        contained_actions = []
        
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')
            
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
    data_path = "/data1/tct_data/habitat/data" # <-- Note: Hardcoded path
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
    """Builds the text observation (prompt) for the agent."""
    postprocess_text_obs = []
    for i in range(len(infos)):
        prompt = HABITAT_VISUAL_GROUNDING_COT_TEMPLATE.format(
            task_caption=infos[i]['task_prompt'],
            conf_score=infos[i]['conf_score']
        )
        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def image_to_base64_str(obs, info=None):
    """将图像（NumPy数组或PIL图像）转换为Base64编码的字符串，可选择绘制bbox信息"""
    if not isinstance(obs, Image.Image):
        img = Image.fromarray(obs)
    else:
        img = obs
    
    # 如果提供了info信息，绘制bbox
    if info is not None:
        # 绘制GT bbox (红色)
        if info.get("bbox_gt") is not None:
            img = draw_bbox_with_text(img, info.get("bbox_gt"), text="GT", color="red", title=info.get("task_prompt"))
        
        # 绘制VG bbox (绿色) - 需要转换格式
        if info.get("pred") is not None:
            img = draw_bbox_with_text(img, reshape_bbox_xxyy(info.get("pred")), text="VG", color="green")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Rule-based policy functions (imported from collect_sft_dataset_rule.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import (
    POLICY_THRESHOLDS,
    get_rule_based_action,
    extract_bbox_from_mask,
    extract_bbox_from_3dbox
)

# 2.【修改】将主逻辑封装到 main 函数中
def main(args):
    """主评估逻辑函数"""
    # --- Configuration ---
    # 3.【修改】使用 argparse 传入的参数
    mode = args.mode
    model_path = args.model_path
    log_filename = args.log_filename
    
    # 4.【新增】自动生成 txt 日志文件名
    txt_log_filename = os.path.splitext(log_filename)[0] + ".txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True) # 确保目录存在

    # (其余配置保持不变)
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
    projection_f = partial(habitat_projection, env_name='habitat')
    print("Habitat environment created.")

    # --- 初始化列表用于存储评估指标和日志 ---
    html_log_entries = []
    txt_log_entries = [] # 5.【新增】用于存储文本日志
    all_initial_ious = []
    all_final_ious = []
    all_initial_confs = []
    all_final_confs = []

    input_filename = args.input_filename
    input_file_path = f'/data1/tct_data/habitat/eval_data/{input_filename}/task_infos.json'
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_task_infos = json.load(f)
        print(f"成功从 {input_file_path} 中加载了 {len(all_task_infos)} 条任务信息。\n")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}。")
        all_task_infos = []

    # ----------------- 主循环开始 -----------------
    total_tasks = len(all_task_infos)
    # # debug
    # eval_task_id = 18
    # all_task_infos = all_task_infos[eval_task_id:eval_task_id+3]
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Tasks")):
        task_info["task_type"] = "grounding"
        obs, info = env.reset_eval(sync_info=task_info)
        
        initial_iou = calculate_iou(info.get("gt")["bbox_gt"], info.get("pred"))
        initial_conf = info.get("conf_score", 0.0)
        all_initial_ious.append(initial_iou)
        all_initial_confs.append(initial_conf)
        
        # 6.【修改】同时准备打印内容和文本日志内容 - 暂时只记录开始信息
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        episode_header_html = f"""
        <h2>--- Episode Start: Task {idx+1}/{total_tasks} (Scene: {info.get('scene_id', 'N/A')}) ---</h2>
        <p><b>Task Prompt:</b> {info.get('task_prompt', 'N/A')}</p>
        """
        html_log_entries.append(episode_header_html)
        
        done = False
        for k in range(max_step_length):
            # clear_output(wait=True)

            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            print(f"Task: {info['task_prompt']}")
            # display(obs)
            
            prompt = build_text_obs([info])[0]
            
            # --- 根据模式选择动作 ---
            if mode == 'model':
                # 模型推理模式
                # For Qwen models, replace <image> with special tokens
                if "qwen" in model_type:
                    prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                # For InternVL, keep <image> as is (it will be handled in inference function)
                text_actions = inference(model, processor, process_image(obs), prompt, model_type)
                actions, valids = projection_f(text_actions, env_name="habitat")
                action_name = env.action_space[actions[0]]
            elif mode == 'rule':
                # 基于规则的策略
                image_width, image_height = obs.size
                task_type = info.get("task_type", "grounding")
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
                text_actions = [f'{{"action": "{action_name}"}}']
                actions = [env.action_space.index(action_name)]
                valids = [1]
            elif mode == 'forward':
                # 向前动作模式
                text_actions = ["forward"]
                actions = [0]
                valids = [1]
                action_name = env.action_space[actions[0]]
            else:  # mode == 'random'
                # 随机动作模式
                text_actions = ["random"]
                actions, valids = projection_f(text_actions, env_name="habitat")
                action_name = env.action_space[actions[0]]
            
            # 7.【修改】同时准备打印内容和文本日志内容
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)

            base64_image = image_to_base64_str(obs, info)
            step_html = f"""
            <div class="step-container">
                <h3>STEP {k+1}/{max_step_length}</h3>
                <p><b>Task:</b> {info['task_prompt']}</p>
                <img src="{base64_image}" alt="Agent view at step {k+1}" class="agent-view">
                <div class="model-output">
                    <p><b>Model Output:</b> <code>{text_actions[0]}</code></p>
                    <p><b>Predicted Action:</b> '{action_name}' (Valid: {bool(valids[0])})</p>
                </div>
            </div>
            """
            html_log_entries.append(step_html)
            
            obs, reward, done, info = env.step(actions[0], valids[0])

            if done:
                # INSERT_YOUR_CODE
                # 在done后再保存一次当前图像到HTML日志
                final_base64_image = image_to_base64_str(obs, info)
                final_step_html = f"""
                <div class="step-container">
                    <h3>FINAL STEP (Episode End)</h3>
                    <p><b>Task:</b> {info['task_prompt']}</p>
                    <img src="{final_base64_image}" alt="Final agent view" class="agent-view">
                    <div class="model-output">
                        <p><b>Model Output:</b> <code>{text_actions[0]}</code></p>
                        <p><b>Predicted Action:</b> '{action_name}' (Valid: {bool(valids[0])})</p>
                        <p><b>Reward:</b> {reward}</p>
                    </div>
                </div>
                """
                html_log_entries.append(final_step_html)
                break
        
        final_iou = calculate_iou(info.get("gt")["bbox_gt"], info.get("pred"))
        final_conf = info.get("conf_score", 0.0)
        all_final_ious.append(final_iou)
        all_final_confs.append(final_conf)

        # 8.【修改】同时准备打印内容和文本日志内容 - 将初始和最终指标记录在一起
        # 计算指标变化
        iou_change = final_iou - initial_iou
        conf_change = final_conf - initial_conf
        
        # 文本日志格式
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
        
        # HTML日志格式 - 带颜色编码
        def get_colored_change(change):
            if change > 0:
                return f'<span style="color: green; font-weight: bold;">{change:+.4f}</span>'
            elif change < 0:
                return f'<span style="color: red; font-weight: bold;">{change:+.4f}</span>'
            else:
                return f'<span style="color: gray;">{change:+.4f}</span>'
        
        iou_change_html = get_colored_change(iou_change)
        conf_change_html = get_colored_change(conf_change)
        
        if done:
            finish_html = f"""
            <div class="episode-summary">
                <h3>Episode finished at step {k+1} with reward {reward}</h3>
                <div class="metrics-comparison">
                    <p><b>IoU:</b> {initial_iou:.4f} → <span style="font-weight: bold;">{final_iou:.4f}</span> ({iou_change_html})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({conf_change_html})</p>
                </div>
            </div>
            """
        else:
            finish_html = f"""
            <div class="episode-summary timeout">
                <h3>Episode timed out at step {k+1}</h3>
                <div class="metrics-comparison">
                    <p><b>IoU:</b> {initial_iou:.4f} → <span style="font-weight: bold;">{final_iou:.4f}</span> ({iou_change_html})</p>
                    <p><b>Conf:</b> {initial_conf:.4f} → <span style="font-weight: bold;">{final_conf:.4f}</span> ({conf_change_html})</p>
                </div>
            </div>
            """
        
        html_log_entries.append(finish_html)

    print("\nEvaluation loop completed.")

    # --- 计算平均值并生成报告 ---
    avg_initial_iou = np.mean(all_initial_ious) if all_initial_ious else 0.0
    avg_final_iou = np.mean(all_final_ious) if all_final_ious else 0.0
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0

    def calculate_improvement(initial, final):
        if initial > 0:
            percent = ((final - initial) / initial) * 100
            return f"{percent:+.2f}%"
        return "N/A (初始值为0)"

    iou_improvement_str = calculate_improvement(avg_initial_iou, avg_final_iou)
    conf_improvement_str = calculate_improvement(avg_initial_conf, avg_final_conf)
    
    # 9.【修改】同时准备打印内容和文本日志内容
    summary_header = "\n--- Evaluation Summary ---"
    summary_body = (
        f"Average Initial IoU:    {avg_initial_iou:.4f}\n"
        f"Average Final IoU:      {avg_final_iou:.4f} ({iou_improvement_str})\n"
        f"Average Initial Conf:   {avg_initial_conf:.4f}\n"
        f"Average Final Conf:     {avg_final_conf:.4f} ({conf_improvement_str})"
    )
    print(summary_header)
    print(summary_body)
    txt_log_entries.append(summary_header + "\n" + summary_body)

    # (HTML 生成部分保持不变, 在此省略)
    def get_colored_html_percent(percent_str):
        if percent_str.startswith('+'):
            return f'<span style="color: green; font-weight: bold;">{percent_str}</span>'
        elif percent_str.startswith('-'): 
            return f'<span style="color: red; font-weight: bold;">{percent_str}</span>'
        else: 
            return f'<span>{percent_str}</span>'
    iou_improvement_html = get_colored_html_percent(iou_improvement_str)
    conf_improvement_html = get_colored_html_percent(conf_improvement_str)
    summary_html = f"""<div class="summary-container"><h2>Evaluation Summary</h2><p><b>Total Tasks Evaluated:</b> {total_tasks}</p><p><b>Average Initial IoU:</b> {avg_initial_iou:.4f}</p><p><b>Average Final IoU:</b> {avg_final_iou:.4f}</p><p><b>IoU Improvement:</b> {iou_improvement_html}</p><hr style="border-top: 1px dashed #ccc; margin: 10px 0;"><p><b>Average Initial Conf:</b> {avg_initial_conf:.4f}</p><p><b>Average Final Conf:</b> {avg_final_conf:.4f}</p><p><b>Conf Improvement:</b> {conf_improvement_html}</p></div><hr>"""
    html_header = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>Evaluation Log</title><style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }} h1 {{ color: #1a237e; }} h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 40px; }} h3 {{ color: #34495e; }} .summary-container {{ background-color: #e3f2fd; border: 2px solid #90caf9; padding: 20px; margin-bottom: 20px; border-radius: 8px; }} .step-container {{ border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 8px; background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }} .agent-view {{ max-width: 480px; height: auto; border: 1px solid #ddd; border-radius: 4px; display: block; margin-top: 10px; }} .model-output {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; margin-top: 10px; }} code {{ background-color: #e0e0e0; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', Courier, monospace; }} .episode-summary {{ background-color: #f8f9fa; border: 2px solid #dee2e6; padding: 15px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }} .episode-summary.timeout {{ border-color: #ffc107; background-color: #fff3cd; }} .metrics-comparison {{ margin-top: 10px; }} .metrics-comparison p {{ margin: 8px 0; font-size: 16px; }}</style></head><body><h1>Evaluation Log - {time.strftime('%Y-%m-%d %H:%M:%S')}</h1>"""
    html_footer = """</body></html>"""

    # --- 写入文件 ---
    # 写入 HTML 文件
    full_html_content = html_header + summary_html + "\n".join(html_log_entries) + html_footer
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(full_html_content)
    print(f"\n✅ HTML log successfully saved to '{log_filename}'")
    
    # 10.【新增】写入 TXT 文件
    full_txt_content = "\n".join(txt_log_entries)
    with open(txt_log_filename, 'w', encoding='utf-8') as f:
        f.write(full_txt_content)
    print(f"✅ Text log successfully saved to '{txt_log_filename}'")


# 11.【新增】脚本入口点
if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Run Habitat environment evaluation with different action selection modes.")
    parser.add_argument('--mode', type=str, choices=['model', 'random', 'rule', 'forward'], default='random',
                        help='Action selection mode: "model" for VL model inference, "random" for random actions, "rule" for rule-based policy.')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the VL model checkpoint directory. Required when --mode=model.')
    parser.add_argument('--input_filename', type=str, 
                        default='replicacad_10-any-500',
                        help='Path to the input_filename.')
    parser.add_argument('--log_filename', type=str, default='evaluation_logs/evaluation_log.html',
                        help='Path to save the output HTML log file. A .txt file with the same base name will also be created.')
    
    # 解析参数并运行主函数
    args = parser.parse_args()

    if not ray.is_initialized():
        ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp",)

    main(args)