"""
统一评估脚本：支持 grounding、segment 和 3d-box 任务
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
from functools import partial
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import ray

# --- Custom Module Imports ---
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.third_party import (
    call_grounding_from_pil_raw, 
    call_grounding_segment_pipeline
)
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import predict_3d_bbox_from_mask
from agent_system.environments.env_package.habitat_sim.projection import habitat_projection as unified_habitat_projection
from agent_system.multi_turn_rollout.utils import process_image

# --- Rule-based policy functions ---
from agent_system.environments.env_package.habitat_sim.dataset.collect_sft_dataset_rule import (
    POLICY_THRESHOLDS,
    get_rule_based_action,
    extract_bbox_from_mask,
    extract_bbox_from_3dbox
)

# --- Local imports ---
from agent_system.environments.env_package.habitat_sim.eval.models import (
    load_model_and_processor, 
    inference,
    create_vlm_client,
    unified_inference,
    VLMClient
)
from agent_system.environments.env_package.habitat_sim.eval.utils import (
    NumpyEncoder,
    build_text_obs,
    GroundingMetrics,
    SegmentMetrics,
    Box3DMetrics,
    GroundingVisualizer,
    SegmentVisualizer,
    Box3DVisualizer
)
from agent_system.environments.env_package.habitat_sim.utils.constants import EVAL_DATA_PATH, EXPERIMENT_DIR, TEMP_DIR

# =============================================================================
# Checkpoint 管理：增量保存与断点续传
# =============================================================================

def save_checkpoint(record_dir: str, checkpoint_data: dict, checkpoint_name: str = "checkpoint.json"):
    """保存 checkpoint 文件"""
    checkpoint_path = os.path.join(record_dir, checkpoint_name)
    # 先写入临时文件，再原子性重命名，避免写入中断导致文件损坏
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    os.replace(tmp_path, checkpoint_path)  # 原子性操作


def load_checkpoint(record_dir: str, checkpoint_name: str = "checkpoint.json") -> Optional[dict]:
    """加载 checkpoint 文件，如果不存在返回 None"""
    checkpoint_path = os.path.join(record_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 加载 checkpoint 失败: {e}，将从头开始")
            return None
    return None


def incremental_save_predictions(record_dir: str, predictions: list, filename: str):
    """增量保存预测结果"""
    filepath = os.path.join(record_dir, filename)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    os.replace(tmp_path, filepath)


def detect_task_type(exp_name: str) -> str:
    """根据 exp_name 关键字判断任务类型"""
    exp_name_lower = exp_name.lower()
    if "grounding" in exp_name_lower:
        return "grounding"
    elif "segment" in exp_name_lower:
        return "segment"
    elif "3d-box" in exp_name_lower or "3dbox" in exp_name_lower:
        return "3d-box"
    else:
        raise ValueError(f"无法从 exp_name '{exp_name}' 推断任务类型。请确保 exp_name 包含 'grounding'、'segment' 或 '3d-box'")


def get_input_filename(task_type: str, base_filename: str) -> str:
    """根据任务类型获取输入文件名"""
    if task_type == "grounding":
        return os.path.join(EVAL_DATA_PATH, base_filename, "task_infos.json")
    elif task_type == "segment":
        return os.path.join(EVAL_DATA_PATH, base_filename, "task_infos_segment.json")
    elif task_type == "3d-box":
        return os.path.join(EVAL_DATA_PATH, base_filename, "task_infos_3d-box.json")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_action_for_mode(mode: str, obs, info, model, processor, model_type, 
                       projection_f, env, gt_task_type: str, gt_task_prompt: str,
                       backend: str = "local"):
    """根据模式选择动作"""
    prompt = build_text_obs([info])[0]
    
    if mode == 'model':
        # 模型推理模式
        if backend == "api":
            # API 模式：model 实际上是 VLMClient 实例
            text_actions = model.inference(process_image(obs), prompt)
        else:
            # Local 模式：直接使用模型推理
            if model_type and ("qwen2" in model_type or "qwen3" in model_type):
                prompt = prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            text_actions = inference(model, processor, process_image(obs), prompt, model_type)
        pred_task_types, pred_task_prompts, actions, valids = projection_f(text_actions)
        action_name = env.action_space[actions[0]]
    elif mode == 'rule':
        # 基于规则的策略
        image_width, image_height = obs.size
        pred = info.get("pred")
        
        bbox_for_action = None
        if gt_task_type == "grounding":
            bbox_for_action = pred
        elif gt_task_type == "class-detect":
            if pred and isinstance(pred, list) and len(pred) > 0 and pred[0].get("category") != "unknown":
                bbox_for_action = pred[0]["bbox"]
        elif gt_task_type == "segment":
            bbox_for_action = extract_bbox_from_mask(pred.get("segment_mask") if pred else None)
        elif gt_task_type == "detect":
            bbox_for_action = None
        elif gt_task_type == "3d-box":
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
        # 向前动作模式
        action_name = "move_forward"
        text_actions = [f'{{"task_type": "{gt_task_type}", "task_prompt": "{gt_task_prompt}", "action": "{action_name}"}}']
        pred_task_types = [gt_task_type]
        pred_task_prompts = [gt_task_prompt]
        actions = [0]
        valids = [1]
    else:  # mode == 'random'
        # 随机动作模式
        text_actions = ["random"]
        pred_task_types, pred_task_prompts, actions, valids = projection_f(text_actions)
        pred_task_types = [gt_task_type]
        pred_task_prompts = [gt_task_prompt]
        action_name = env.action_space[actions[0]]
    
    return text_actions, pred_task_types, pred_task_prompts, actions, valids, action_name


def eval_grounding(args, env, projection_f, model, processor, model_type, 
                  all_task_infos, exp_dir, record_dir, txt_log_filename):
    """Grounding 任务评估"""
    metrics = GroundingMetrics()
    visualizer = GroundingVisualizer(record_dir)
    
    max_step_length = 10
    total_tasks = len(all_task_infos)
    
    # 尝试加载 checkpoint（断点续传）
    checkpoint = load_checkpoint(record_dir)
    if checkpoint:
        print(f"✅ 发现 checkpoint，从任务 {checkpoint['last_completed_idx'] + 1} 继续")
        start_idx = checkpoint['last_completed_idx'] + 1
        txt_log_entries = checkpoint.get('txt_log_entries', [])
        all_initial_ious = checkpoint.get('all_initial_ious', [])
        all_final_ious = checkpoint.get('all_final_ious', [])
        all_initial_confs = checkpoint.get('all_initial_confs', [])
        all_final_confs = checkpoint.get('all_final_confs', [])
        all_predictions_initial = checkpoint.get('all_predictions_initial', [])
        all_predictions_final = checkpoint.get('all_predictions_final', [])
        task_type_correct_indices = checkpoint.get('task_type_correct_indices', [])
    else:
        print("📝 未发现 checkpoint，从头开始评估")
        start_idx = 0
        txt_log_entries = []
        all_initial_ious = []
        all_final_ious = []
        all_initial_confs = []
        all_final_confs = []
        all_predictions_initial = []
        all_predictions_final = []
        task_type_correct_indices = []
    
    # #NOTE: 仅评估前100个任务
    # all_task_infos = all_task_infos[:100]
    
    # 增量保存间隔（每 N 个任务保存一次）
    save_interval = 5
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Grounding Tasks")):
        # 跳过已完成的任务
        if idx < start_idx:
            continue
        task_info["task_type"] = "grounding"
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测
        response = call_grounding_from_pil_raw(obs, info.get("task_prompt"))
        pred_bboxes = response["bboxes"]
        pred_scores = response["scores"]
        
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
        
        # 创建任务目录并保存初始图像
        task_dir = visualizer.create_task_dir(idx)
        task_step_records = []
        
        initial_conf = max(pred_scores) if pred_scores else 0
        init_pred_bbox = pred_bboxes[int(np.argmax(pred_scores))] if pred_scores else None
        initial_iou = metrics.calculate_iou(gt_bbox, init_pred_bbox)
        all_initial_ious.append(initial_iou)
        all_initial_confs.append(initial_conf)
        visualizer.save_step_image(task_dir, 0, obs, init_pred_bbox, gt_bbox, is_init=True)
        
        all_steps_task_type_correct = True
        done = False
        
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            
            gt_task_type = info.get("task_type", "grounding")
            gt_task_prompt = info.get("task_prompt", "")
            
            text_actions, pred_task_types, pred_task_prompts, actions, valids, action_name = get_action_for_mode(
                args.mode, obs, info, model, processor, model_type,
                projection_f, env, gt_task_type, gt_task_prompt, backend=args.backend
            )
            
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Pred Task Type: '{pred_task_types[0]}', Pred Task Prompt: '{pred_task_prompts[0][:50]}...'\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)
            

            is_task_type_correct = (pred_task_types[0] == gt_task_type)                                     
            if not is_task_type_correct:
                print("Task type incorrect, skipping step")
                all_steps_task_type_correct = False
                step_record = {
                    "step": k + 1,
                    "model_output": text_actions[0],
                    "pred_task_type": pred_task_types[0],
                    "pred_task_prompt": pred_task_prompts[0],
                    "gt_task_type": gt_task_type,
                    "gt_task_prompt": gt_task_prompt,
                    "action": action_name,
                    "is_valid": bool(valids[0]),
                    "is_task_type_correct": False,
                    "skipped": True
                }
                _, info["gt"] = env.get_gt()
                task_step_records.append(step_record)
                break
            
            obs, reward, done, info = env.step(
                pred_task_types[0], pred_task_prompts[0], actions[0], valids[0]
            )
            
            current_pred_bbox = info.get("pred")
            gt_info = info.get("gt", {})
            current_gt_bbox = gt_info.get("bbox_gt") if gt_info else None
            
            visualizer.save_step_image(task_dir, k + 1, obs, current_pred_bbox, current_gt_bbox)
            
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
        
        # 获取最终预测      
        if all_steps_task_type_correct:
            task_type_correct_indices.append(idx)

            response = call_grounding_from_pil_raw(obs, info.get("task_prompt"))
            pred_bboxes = response["bboxes"]
            pred_scores = response["scores"]        
        else:
            pred_bboxes = []
            pred_scores = []


        all_predictions_final.append({
            'gt_bbox': info.get("gt")["bbox_gt"],
            'pred_bboxes': pred_bboxes,
            'pred_scores': pred_scores,
            'phrase': info.get("task_prompt")
        })
        
        final_iou = metrics.calculate_iou(info.get("gt")["bbox_gt"], info.get("pred")) if is_task_type_correct else 0.0
        final_conf = info.get("conf_score", 0.0) if is_task_type_correct else 0.0
        all_final_ious.append(final_iou)
        all_final_confs.append(final_conf)
        
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
        
        # 保存任务记录
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
        visualizer.save_task_record(task_dir, task_record)
        
        # 增量保存：每完成 save_interval 个任务或最后一个任务时保存
        if (idx + 1) % save_interval == 0 or idx == len(all_task_infos) - 1:
            # 保存 checkpoint
            checkpoint_data = {
                'last_completed_idx': idx,
                'txt_log_entries': txt_log_entries,
                'all_initial_ious': all_initial_ious,
                'all_final_ious': all_final_ious,
                'all_initial_confs': all_initial_confs,
                'all_final_confs': all_final_confs,
                'all_predictions_initial': all_predictions_initial,
                'all_predictions_final': all_predictions_final,
                'task_type_correct_indices': task_type_correct_indices
            }
            save_checkpoint(record_dir, checkpoint_data)
            
            # 增量保存预测结果
            incremental_save_predictions(record_dir, all_predictions_initial, "all_predictions_initial.json")
            incremental_save_predictions(record_dir, all_predictions_final, "all_predictions_final.json")
            
            print(f"💾 Checkpoint 已保存 (任务 {idx + 1}/{len(all_task_infos)})")
    
    print("\nEvaluation loop completed.")
    
    # 计算并打印摘要
    avg_initial_iou = np.mean(all_initial_ious) if all_initial_ious else 0.0
    avg_final_iou = np.mean(all_final_ious) if all_final_ious else 0.0
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0
    num_task_type_correct = len(task_type_correct_indices)
    
    # 计算 mAP
    print("\nCalculating mAP at multiple IoU thresholds...")
    initial_mAPs = {}
    final_mAPs = {}
    
    for threshold in metrics.iou_thresholds:
        print(f"  Computing mAP@{threshold}...")
        initial_mAP, _ = metrics.calculate_mAP(all_predictions_initial, iou_threshold=threshold)
        final_mAP, _ = metrics.calculate_mAP(all_predictions_final, iou_threshold=threshold)
        initial_mAPs[threshold] = initial_mAP
        final_mAPs[threshold] = final_mAP
        print(f"    Initial mAP@{threshold}: {initial_mAP:.4f}")
        print(f"    Final mAP@{threshold}:   {final_mAP:.4f}")
    
    aggregated = {
        'avg_initial_iou': avg_initial_iou,
        'avg_final_iou': avg_final_iou
    }
    
    summary = metrics.generate_summary(
        aggregated, total_tasks, num_task_type_correct,
        initial_mAPs, final_mAPs,
        avg_initial_conf, avg_final_conf
    )
    
    print(summary)
    txt_log_entries.append(summary)
    
    # 最终保存预测结果（带格式化）
    predictions_initial_path = os.path.join(record_dir, "all_predictions_initial.json")
    with open(predictions_initial_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_initial, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ Initial predictions saved to '{predictions_initial_path}'")
    
    predictions_final_path = os.path.join(record_dir, "all_predictions_final.json")
    with open(predictions_final_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_final, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ Final predictions saved to '{predictions_final_path}'")
    
    # # 清理 checkpoint（评估完成后删除）
    # checkpoint_path = os.path.join(record_dir, "checkpoint.json")
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     print("🗑️ Checkpoint 已清理（评估完成）")
    
    # 保存文本日志
    full_txt_content = "\n".join(txt_log_entries)
    with open(txt_log_filename, 'w', encoding='utf-8') as f:
        f.write(full_txt_content)
    print(f"\n✅ Text log successfully saved to '{txt_log_filename}'")


def eval_segment(args, env, projection_f, model, processor, model_type,
                all_task_infos, exp_dir, record_dir, txt_log_filename):
    """Segment 任务评估"""
    metrics = SegmentMetrics()
    visualizer = SegmentVisualizer(record_dir)
    
    max_step_length = 10
    total_tasks = len(all_task_infos)
    
    # 尝试加载 checkpoint（断点续传）
    checkpoint = load_checkpoint(record_dir)
    if checkpoint:
        print(f"✅ 发现 checkpoint，从任务 {checkpoint['last_completed_idx'] + 1} 继续")
        start_idx = checkpoint['last_completed_idx'] + 1
        txt_log_entries = checkpoint.get('txt_log_entries', [])
        all_initial_metrics = checkpoint.get('all_initial_metrics', [])
        all_final_metrics = checkpoint.get('all_final_metrics', [])
        all_initial_confs = checkpoint.get('all_initial_confs', [])
        all_final_confs = checkpoint.get('all_final_confs', [])
        task_type_correct_indices = checkpoint.get('task_type_correct_indices', [])
    else:
        print("📝 未发现 checkpoint，从头开始评估")
        start_idx = 0
        txt_log_entries = []
        all_initial_metrics = []
        all_final_metrics = []
        all_initial_confs = []
        all_final_confs = []
        task_type_correct_indices = []
    
    # 增量保存间隔
    save_interval = 5
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing Segment Tasks")):
        # 跳过已完成的任务
        if idx < start_idx:
            continue
        task_info["task_type"] = "segment"
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测
        response = call_grounding_segment_pipeline(obs, info.get("task_prompt"))
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        initial_conf = (grounding_score + segment_score) / 2.0
        initial_pred_mask = response.get("segment_mask")
        
        gt_info = info.get("gt", {})
        gt_bbox = gt_info.get("bbox_gt")
        gt_mask = gt_info.get("mask_gt")
        
        if gt_mask is None and gt_bbox is not None:
            gt_mask = metrics.create_gt_mask_from_bbox(gt_bbox)
        
        initial_metrics_dict = metrics.calculate_metrics(initial_pred_mask, gt_mask)
        
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}\n"
            f"Initial Metrics - mIoU: {initial_metrics_dict['mask_iou']:.4f}, "
            f"Dice: {initial_metrics_dict['dice']:.4f}, PixelAcc: {initial_metrics_dict['pixel_acc']:.4f}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        # 创建任务目录并保存初始图像
        task_dir = visualizer.create_task_dir(idx)
        task_step_records = []
        
        visualizer.save_step_image(task_dir, 0, obs, initial_pred_mask, gt_bbox, is_init=True)
        
        all_initial_metrics.append(initial_metrics_dict)
        all_initial_confs.append(initial_conf)

        all_steps_task_type_correct = True
        done = False
        
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            
            gt_task_type = info.get("task_type", "segment")
            gt_task_prompt = info.get("task_prompt", "")
            
            text_actions, pred_task_types, pred_task_prompts, actions, valids, action_name = get_action_for_mode(
                args.mode, obs, info, model, processor, model_type,
                projection_f, env, gt_task_type, gt_task_prompt, backend=args.backend
            )
            
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Pred Task Type: '{pred_task_types[0]}', Pred Task Prompt: '{pred_task_prompts[0][:50]}...'\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)
            

            is_task_type_correct = (pred_task_types[0] == gt_task_type)            
            if not is_task_type_correct:
                print("Task type incorrect, skipping step")
                all_steps_task_type_correct = False
                step_record = {
                    "step": k + 1,
                    "model_output": text_actions[0],
                    "pred_task_type": pred_task_types[0],
                    "pred_task_prompt": pred_task_prompts[0],
                    "gt_task_type": gt_task_type,
                    "gt_task_prompt": gt_task_prompt,
                    "action": action_name,
                    "is_valid": bool(valids[0]),
                    "is_task_type_correct": False,
                    "skipped": True
                }
                task_step_records.append(step_record)
                _, info["gt"] = env.get_gt()
                break
            
            obs, reward, done, info = env.step(
                pred_task_types[0], pred_task_prompts[0], actions[0], valids[0]
            )
            
            current_pred = info.get("pred", {})
            current_pred_mask = current_pred.get("segment_mask") if current_pred else None
            gt_info = info.get("gt", {})
            current_gt_bbox = gt_info.get("bbox_gt") if gt_info else None
            
            visualizer.save_step_image(task_dir, k + 1, obs, current_pred_mask, current_gt_bbox)
            
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
        
        # 计算最终指标
        final_pred = info.get("pred", {})
        final_pred_mask = final_pred.get("segment_mask") if final_pred else None
        final_conf = info.get("conf_score", 0.0)
        final_gt_mask = info.get("gt", {}).get("mask_gt")
        
        if all_steps_task_type_correct:
            task_type_correct_indices.append(idx)
            
            final_metrics_dict = metrics.calculate_metrics(final_pred_mask, final_gt_mask)
            all_final_metrics.append(final_metrics_dict)
            all_final_confs.append(final_conf)
        else:
            all_final_metrics.append({
                'mask_iou': 0.0, 'dice': 0.0, 'pixel_acc': 0.0,
                'precision': 0.0, 'recall': 0.0
            })
            all_final_confs.append(0.0)
            final_metrics_dict = all_final_metrics[-1]
        
        initial_m = all_initial_metrics[-1]
        final_m = final_metrics_dict
        
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"mIoU: {initial_m['mask_iou']:.4f} → {final_m['mask_iou']:.4f}\n"
                f"Dice: {initial_m['dice']:.4f} → {final_m['dice']:.4f}\n"
                f"Conf: {all_initial_confs[-1]:.4f} → {all_final_confs[-1]:.4f}"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"mIoU: {initial_m['mask_iou']:.4f} → {final_m['mask_iou']:.4f}\n"
                f"Dice: {initial_m['dice']:.4f} → {final_m['dice']:.4f}\n"
                f"Conf: {all_initial_confs[-1]:.4f} → {all_final_confs[-1]:.4f}"
            )
        
        print(episode_finish_str)
        txt_log_entries.append(episode_finish_str + "\n")
        
        # 保存任务记录
        task_record = {
            "task_id": task_info.get("task_id", idx),
            "task_idx": idx,
            "scene_id": info.get("scene_id", "N/A"),
            "task_prompt": info.get("task_prompt", "N/A"),
            "gt_task_type": gt_task_type,
            "is_task_type_correct": is_task_type_correct,
            "initial_metrics": initial_m,
            "final_metrics": final_m,
            "initial_conf": all_initial_confs[-1],
            "final_conf": all_final_confs[-1],
            "total_steps": k + 1,
            "done": done,
            "steps": task_step_records
        }
        visualizer.save_task_record(task_dir, task_record)
        
        # 增量保存：每完成 save_interval 个任务或最后一个任务时保存
        if (idx + 1) % save_interval == 0 or idx == len(all_task_infos) - 1:
            checkpoint_data = {
                'last_completed_idx': idx,
                'txt_log_entries': txt_log_entries,
                'all_initial_metrics': all_initial_metrics,
                'all_final_metrics': all_final_metrics,
                'all_initial_confs': all_initial_confs,
                'all_final_confs': all_final_confs,
                'task_type_correct_indices': task_type_correct_indices
            }
            save_checkpoint(record_dir, checkpoint_data)
            print(f"💾 Checkpoint 已保存 (任务 {idx + 1}/{len(all_task_infos)})")
    
    print("\nEvaluation loop completed.")
    
    # 计算并打印摘要
    num_task_type_correct = len(task_type_correct_indices)
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0
    
    aggregated = metrics.aggregate_metrics(all_initial_metrics, all_final_metrics, task_type_correct_indices)
    summary = metrics.generate_summary(aggregated, total_tasks, num_task_type_correct, avg_initial_conf, avg_final_conf)
    
    print(summary)
    txt_log_entries.append(summary)
    
    # 保存文本日志
    full_txt_content = "\n".join(txt_log_entries)
    with open(txt_log_filename, 'w', encoding='utf-8') as f:
        f.write(full_txt_content)
    print(f"\n✅ Text log successfully saved to '{txt_log_filename}'")
    
    # # 清理 checkpoint
    # checkpoint_path = os.path.join(record_dir, "checkpoint.json")
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     print("🗑️ Checkpoint 已清理（评估完成）")


def eval_3dbox(args, env, projection_f, model, processor, model_type,
               all_task_infos, exp_dir, record_dir, txt_log_filename):
    """3D-Box 任务评估"""
    metrics = Box3DMetrics()
    visualizer = Box3DVisualizer(record_dir)
    
    max_step_length = 10
    total_tasks = len(all_task_infos)
    
    # 尝试加载 checkpoint（断点续传）
    checkpoint = load_checkpoint(record_dir)
    if checkpoint:
        print(f"✅ 发现 checkpoint，从任务 {checkpoint['last_completed_idx'] + 1} 继续")
        start_idx = checkpoint['last_completed_idx'] + 1
        txt_log_entries = checkpoint.get('txt_log_entries', [])
        all_initial_metrics = checkpoint.get('all_initial_metrics', [])
        all_final_metrics = checkpoint.get('all_final_metrics', [])
        all_initial_confs = checkpoint.get('all_initial_confs', [])
        all_final_confs = checkpoint.get('all_final_confs', [])
        task_type_correct_indices = checkpoint.get('task_type_correct_indices', [])
    else:
        print("📝 未发现 checkpoint，从头开始评估")
        start_idx = 0
        txt_log_entries = []
        all_initial_metrics = []
        all_final_metrics = []
        all_initial_confs = []
        all_final_confs = []
        task_type_correct_indices = []
    
    # 增量保存间隔
    save_interval = 5
    
    for idx, task_info in enumerate(tqdm(all_task_infos, desc="Processing 3D-Box Tasks")):
        # 跳过已完成的任务
        if idx < start_idx:
            continue
        task_info["task_type"] = "3d-box"
        obs, info = env.reset_eval(sync_info=task_info)
        
        # 获取初始预测 3D bbox
        response = call_grounding_segment_pipeline(obs, info.get("task_prompt"))
        segment_mask = response.get("segment_mask")
        bbox_3d = predict_3d_bbox_from_mask(
            mask_rle=segment_mask,
            depth_obs=env.sim.get_sensor_observations()["depth"],
            agent_state=env.sim.get_agent(0).get_state(),
            hfov=90.0,
            denoise=True,
            align_to_ground=True
        )
        
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        geometric_confidence = bbox_3d.get("geometric_confidence", 0.0) if bbox_3d else 0.0
        initial_conf = (grounding_score + segment_score + geometric_confidence) / 3.0
        initial_pred_bbox_3d = bbox_3d
        
        gt_info = info.get("gt", {})
        gt_bbox_3d = gt_info.get("bbox_3d_gt")
        
        initial_metrics_dict = metrics.calculate_metrics(initial_pred_bbox_3d, gt_bbox_3d)
        
        episode_start_str = (
            f"\n--- Starting new episode: Task {idx+1}/{total_tasks} ---\n"
            f"Scene Path: {info.get('scene_id', 'N/A')}\n"
            f"Task Prompt: {info.get('task_prompt', 'N/A')}\n"
            f"Initial Metrics - 3D IoU: {initial_metrics_dict['iou_3d']:.4f}, "
            f"Center Score: {initial_metrics_dict['center_score']:.4f}"
        )
        print(episode_start_str)
        txt_log_entries.append(episode_start_str)
        
        # 创建任务目录并保存初始图像
        task_dir = visualizer.create_task_dir(idx)
        task_step_records = []
        
        visualizer.save_step_image(task_dir, 0, obs, initial_pred_bbox_3d, gt_bbox_3d,
                                   info.get('task_prompt'), idx, is_init=True)
        
        all_initial_metrics.append(initial_metrics_dict)
        all_initial_confs.append(initial_conf)

        all_steps_task_type_correct = True
        done = False
        
        for k in range(max_step_length):
            print(f"TASK {idx+1}/{total_tasks} | STEP {k+1}/{max_step_length}")
            
            gt_task_type = info.get("task_type", "3d-box")
            gt_task_prompt = info.get("task_prompt", "")
            
            text_actions, pred_task_types, pred_task_prompts, actions, valids, action_name = get_action_for_mode(
                args.mode, obs, info, model, processor, model_type,
                projection_f, env, gt_task_type, gt_task_prompt, backend=args.backend
            )
            
            step_log_str = (
                f"\n---------- STEP {k+1}/{max_step_length} ----------\n"
                f"Model Output: {text_actions[0]}\n"
                f"Pred Task Type: '{pred_task_types[0]}', Pred Task Prompt: '{pred_task_prompts[0][:50]}...'\n"
                f"Predicted Action: '{action_name}' (Valid: {bool(valids[0])})"
            )
            print(step_log_str)
            txt_log_entries.append(step_log_str)
            

            is_task_type_correct = (pred_task_types[0] == gt_task_type)
            if not is_task_type_correct:
                print("Task type incorrect, skipping step")
                all_steps_task_type_correct = False
                step_record = {
                    "step": k + 1,
                    "model_output": text_actions[0],
                    "pred_task_type": pred_task_types[0],
                    "pred_task_prompt": pred_task_prompts[0],
                    "gt_task_type": gt_task_type,
                    "gt_task_prompt": gt_task_prompt,
                    "action": action_name,
                    "is_valid": bool(valids[0]),
                    "is_task_type_correct": False,
                    "skipped": True
                }
                task_step_records.append(step_record)
                _, info["gt"] = env.get_gt()
                break
            
            obs, reward, done, info = env.step(
                pred_task_types[0], pred_task_prompts[0], actions[0], valids[0]
            )
            
            current_pred_bbox_3d = info.get("pred")
            gt_info = info.get("gt", {})
            current_gt_bbox_3d = gt_info.get("bbox_3d_gt") if gt_info else None
            
            visualizer.save_step_image(task_dir, k + 1, obs, current_pred_bbox_3d, current_gt_bbox_3d,
                                       info.get('task_prompt'), idx)
            
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
        
        # 计算最终指标
        final_pred_bbox_3d = info.get("pred")
        final_conf = info.get("conf_score", 0.0)
        final_gt_bbox_3d = info.get("gt", {}).get("bbox_3d_gt")
        
        if all_steps_task_type_correct:
            task_type_correct_indices.append(idx)
            
            final_metrics_dict = metrics.calculate_metrics(final_pred_bbox_3d, final_gt_bbox_3d)
            all_final_metrics.append(final_metrics_dict)
            all_final_confs.append(final_conf)
        else:
            all_final_metrics.append({'iou_3d': 0.0, 'center_score': 0.0})
            all_final_confs.append(0.0)
            final_metrics_dict = all_final_metrics[-1]
        
        initial_m = all_initial_metrics[-1]
        final_m = final_metrics_dict
        
        if done:
            episode_finish_str = (
                f"\n--- Episode finished at step {k+1} with reward {reward} ---\n"
                f"3D IoU: {initial_m['iou_3d']:.4f} → {final_m['iou_3d']:.4f}\n"
                f"Center Score: {initial_m['center_score']:.4f} → {final_m['center_score']:.4f}\n"
                f"Conf: {all_initial_confs[-1]:.4f} → {all_final_confs[-1]:.4f}"
            )
        else:
            episode_finish_str = (
                f"\n--- Episode timed out at step {k+1} ---\n"
                f"3D IoU: {initial_m['iou_3d']:.4f} → {final_m['iou_3d']:.4f}\n"
                f"Center Score: {initial_m['center_score']:.4f} → {final_m['center_score']:.4f}\n"
                f"Conf: {all_initial_confs[-1]:.4f} → {all_final_confs[-1]:.4f}"
            )
        
        print(episode_finish_str)
        txt_log_entries.append(episode_finish_str + "\n")
        
        # 保存任务记录
        task_record = {
            "task_id": task_info.get("task_id", idx),
            "task_idx": idx,
            "scene_id": info.get("scene_id", "N/A"),
            "task_prompt": info.get("task_prompt", "N/A"),
            "gt_task_type": gt_task_type,
            "is_task_type_correct": is_task_type_correct,
            "initial_metrics": initial_m,
            "final_metrics": final_m,
            "initial_conf": all_initial_confs[-1],
            "final_conf": all_final_confs[-1],
            "total_steps": k + 1,
            "done": done,
            "steps": task_step_records
        }
        visualizer.save_task_record(task_dir, task_record)
        
        # 增量保存：每完成 save_interval 个任务或最后一个任务时保存
        if (idx + 1) % save_interval == 0 or idx == len(all_task_infos) - 1:
            checkpoint_data = {
                'last_completed_idx': idx,
                'txt_log_entries': txt_log_entries,
                'all_initial_metrics': all_initial_metrics,
                'all_final_metrics': all_final_metrics,
                'all_initial_confs': all_initial_confs,
                'all_final_confs': all_final_confs,
                'task_type_correct_indices': task_type_correct_indices
            }
            save_checkpoint(record_dir, checkpoint_data)
            print(f"💾 Checkpoint 已保存 (任务 {idx + 1}/{len(all_task_infos)})")
    
    print("\nEvaluation loop completed.")
    
    # 计算并打印摘要
    num_task_type_correct = len(task_type_correct_indices)
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0
    
    aggregated = metrics.aggregate_metrics(all_initial_metrics, all_final_metrics, task_type_correct_indices)
    summary = metrics.generate_summary(aggregated, total_tasks, num_task_type_correct, avg_initial_conf, avg_final_conf)
    
    print(summary)
    txt_log_entries.append(summary)
    
    # 保存文本日志
    full_txt_content = "\n".join(txt_log_entries)
    with open(txt_log_filename, 'w', encoding='utf-8') as f:
        f.write(full_txt_content)
    print(f"\n✅ Text log successfully saved to '{txt_log_filename}'")
    
    # # 清理 checkpoint
    # checkpoint_path = os.path.join(record_dir, "checkpoint.json")
    # if os.path.exists(checkpoint_path):
    #     os.remove(checkpoint_path)
    #     print("🗑️ Checkpoint 已清理（评估完成）")


def main(args):
    """主评估逻辑函数"""
    mode = args.mode
    model_path = args.model_path
    exp_name = args.exp_name
    
    # 检测任务类型
    task_type = detect_task_type(exp_name)
    print(f"Detected task type: {task_type}")
    
    # 根据 model_path 确定模型类型子目录
    model_subdir = ""
    if model_path:
        model_path_lower = model_path.lower()
        if "internvl" in model_path_lower:
            model_subdir = "internvl"
        elif "qwen" in model_path_lower:
            model_subdir = "qwen"
    
    # 创建实验输出目录
    if model_subdir:
        exp_dir = os.path.join(EXPERIMENT_DIR, model_subdir, exp_name)
    else:
        exp_dir = os.path.join(EXPERIMENT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    txt_log_filename = os.path.join(exp_dir, "log.txt")
    record_dir = os.path.join(exp_dir, "task_records")
    os.makedirs(record_dir, exist_ok=True)
    
    print(f"Experiment output directory: {exp_dir}")
    
    # 确定数据集名称
    dataset_name = "HM3D" if "HM3D" in exp_name else "ReplicaCAD"
    seed = 0
    scenes_size = 10
    max_scene_instance = 10
    max_step_length = 10
    
    # 加载模型（如果需要）
    model, processor, model_type = None, None, None
    backend = args.backend
    
    if mode == 'model':
        if backend == 'api':
            # API 模式：使用 VLMClient
            if not args.model_name:
                raise ValueError("--model_name is required when --backend=api")
            model = create_vlm_client(
                backend='api',
                api_url=args.api_url,
                api_key=args.api_key,
                model_name=args.model_name
            )
            processor, model_type = None, None
            print(f"VLMClient created: api_url={args.api_url}, model_name={args.model_name}")
        else:
            # Local 模式：直接加载模型
            if not model_path:
                raise ValueError("--model_path is required when --mode=model and --backend=local")
            model, processor, model_type = load_model_and_processor(model_path)
            print(f"Model and processor loaded successfully. Model type: {model_type}")
    else:
        print(f"Running in {mode.upper()} mode, skipping model loading.")
    
    # 创建环境
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    projection_f = partial(unified_habitat_projection, env_name='habitat')
    print("Habitat environment created.")
    
    # 加载任务信息
    input_file_path = get_input_filename(task_type, args.input_filename)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_task_infos = json.load(f)
        print(f"成功从 {input_file_path} 中加载了 {len(all_task_infos)} 条任务信息。\n")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}。")
        return
    
    # 根据任务类型调用对应的评估函数
    if task_type == "grounding":
        eval_grounding(args, env, projection_f, model, processor, model_type,
                      all_task_infos, exp_dir, record_dir, txt_log_filename)
    elif task_type == "segment":
        eval_segment(args, env, projection_f, model, processor, model_type,
                    all_task_infos, exp_dir, record_dir, txt_log_filename)
    elif task_type == "3d-box":
        eval_3dbox(args, env, projection_f, model, processor, model_type,
                  all_task_infos, exp_dir, record_dir, txt_log_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Habitat environment evaluation with different action selection modes.")
    parser.add_argument('--mode', type=str, choices=['model', 'random', 'rule', 'forward'], default='model',
                        help='Action selection mode: "model" for VL model inference, "random" for random actions, "rule" for rule-based policy.')
    parser.add_argument('--model_path', type=str, default='/data/tct/models/RL/InternVL3_5-2B-unified-tasks-rl', 
                        help='Path to the VL model checkpoint directory. Required when --mode=model and --backend=local.')
    parser.add_argument('--input_filename', type=str, 
                        default='replicacad_10-any-500-seen',
                        help='Path to the input_filename.')
    parser.add_argument('--exp_name', type=str, default='ReplicaCAD-grounding-alpha0.5',
                        help='Experiment name. All outputs will be saved to /data/tct/ActivePerception/exp/{exp_name}/')
    
    # vLLM API 相关参数
    parser.add_argument('--backend', type=str, choices=['api', 'local'], default='local',
                        help='Backend mode: "api" for vLLM/OpenAI API, "local" for direct HuggingFace loading.')
    parser.add_argument('--api_url', type=str, default='http://localhost:8045/v1',
                        help='vLLM/OpenAI API base URL. Used when --backend=api.')
    parser.add_argument('--api_key', type=str, default='sk-54e8acb6671340e59cb00eab1f5b447c',
                        help='API key for authentication. Used when --backend=api.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name for API calls. Required when --backend=api.')
    
    args = parser.parse_args()
    
    if not ray.is_initialized():
        ray.init(_temp_dir=TEMP_DIR,)
    
    main(args)
