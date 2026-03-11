"""
从checkpoint计算前N个任务的指标
使用现有的checkpoint和预测结果文件,计算指定数量任务的评估指标
"""
import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List

# 导入工具类
from agent_system.environments.env_package.habitat_sim.eval.utils import (
    NumpyEncoder,
    GroundingMetrics,
)


def load_checkpoint(checkpoint_path: str) -> dict:
    """加载checkpoint文件"""
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics_for_n_tasks(checkpoint_data: dict, n_tasks: int, output_dir: str):
    """计算前N个任务的指标"""
    
    # 提取前N个任务的数据
    all_initial_ious = checkpoint_data.get('all_initial_ious', [])[:n_tasks]
    all_final_ious = checkpoint_data.get('all_final_ious', [])[:n_tasks]
    all_initial_confs = checkpoint_data.get('all_initial_confs', [])[:n_tasks]
    all_final_confs = checkpoint_data.get('all_final_confs', [])[:n_tasks]
    all_predictions_initial = checkpoint_data.get('all_predictions_initial', [])[:n_tasks]
    all_predictions_final = checkpoint_data.get('all_predictions_final', [])[:n_tasks]
    task_type_correct_indices = [idx for idx in checkpoint_data.get('task_type_correct_indices', []) if idx < n_tasks]
    
    print(f"\n{'='*60}")
    print(f"计算前 {n_tasks} 个任务的指标")
    print(f"{'='*60}\n")
    
    # 基本统计
    print(f"✅ 成功加载 {len(all_initial_ious)} 个任务的数据")
    print(f"✅ 任务类型正确的任务数: {len(task_type_correct_indices)}")
    print(f"✅ 任务类型正确率: {len(task_type_correct_indices)/n_tasks*100:.2f}%\n")
    
    # 计算平均IoU和置信度
    avg_initial_iou = np.mean(all_initial_ious) if all_initial_ious else 0.0
    avg_final_iou = np.mean(all_final_ious) if all_final_ious else 0.0
    avg_initial_conf = np.mean(all_initial_confs) if all_initial_confs else 0.0
    avg_final_conf = np.mean(all_final_confs) if all_final_confs else 0.0
    
    # 只计算任务类型正确的任务的指标
    if task_type_correct_indices:
        correct_initial_ious = [all_initial_ious[i] for i in task_type_correct_indices]
        correct_final_ious = [all_final_ious[i] for i in task_type_correct_indices]
        correct_initial_confs = [all_initial_confs[i] for i in task_type_correct_indices]
        correct_final_confs = [all_final_confs[i] for i in task_type_correct_indices]
        
        avg_correct_initial_iou = np.mean(correct_initial_ious)
        avg_correct_final_iou = np.mean(correct_final_ious)
        avg_correct_initial_conf = np.mean(correct_initial_confs)
        avg_correct_final_conf = np.mean(correct_final_confs)
    else:
        avg_correct_initial_iou = 0.0
        avg_correct_final_iou = 0.0
        avg_correct_initial_conf = 0.0
        avg_correct_final_conf = 0.0
    
    # 创建metrics对象用于计算mAP
    metrics = GroundingMetrics()
    
    # 计算mAP (所有任务)
    print("计算所有任务的 mAP...")
    initial_mAPs = {}
    final_mAPs = {}
    
    for threshold in metrics.iou_thresholds:
        initial_mAP, _ = metrics.calculate_mAP(all_predictions_initial, iou_threshold=threshold)
        final_mAP, _ = metrics.calculate_mAP(all_predictions_final, iou_threshold=threshold)
        initial_mAPs[threshold] = initial_mAP
        final_mAPs[threshold] = final_mAP
    
    # 计算mAP (仅任务类型正确的任务)
    print("计算任务类型正确任务的 mAP...")
    correct_predictions_initial = [all_predictions_initial[i] for i in task_type_correct_indices]
    correct_predictions_final = [all_predictions_final[i] for i in task_type_correct_indices]
    
    correct_initial_mAPs = {}
    correct_final_mAPs = {}
    
    for threshold in metrics.iou_thresholds:
        correct_initial_mAP, _ = metrics.calculate_mAP(correct_predictions_initial, iou_threshold=threshold)
        correct_final_mAP, _ = metrics.calculate_mAP(correct_predictions_final, iou_threshold=threshold)
        correct_initial_mAPs[threshold] = correct_initial_mAP
        correct_final_mAPs[threshold] = correct_final_mAP
    
    # 生成报告
    summary = f"""
{'='*80}
前 {n_tasks} 个任务的评估结果
{'='*80}

【所有任务统计】
--------------------------------------------------------------------------------
总任务数: {n_tasks}
任务类型正确数: {len(task_type_correct_indices)}
任务类型正确率: {len(task_type_correct_indices)/n_tasks*100:.2f}%

【IoU 指标 (所有任务)】
--------------------------------------------------------------------------------
初始平均 IoU: {avg_initial_iou:.4f}
最终平均 IoU: {avg_final_iou:.4f}
IoU 变化: {avg_final_iou - avg_initial_iou:+.4f}

【置信度指标 (所有任务)】
--------------------------------------------------------------------------------
初始平均置信度: {avg_initial_conf:.4f}
最终平均置信度: {avg_final_conf:.4f}
置信度变化: {avg_final_conf - avg_initial_conf:+.4f}

【mAP 指标 (所有任务)】
--------------------------------------------------------------------------------
"""
    
    for threshold in sorted(initial_mAPs.keys()):
        summary += f"mAP@{threshold:.2f}:\n"
        summary += f"  初始: {initial_mAPs[threshold]:.4f}\n"
        summary += f"  最终: {final_mAPs[threshold]:.4f}\n"
        summary += f"  变化: {final_mAPs[threshold] - initial_mAPs[threshold]:+.4f}\n"
    
    summary += f"""
{'='*80}
【仅任务类型正确的任务统计】({len(task_type_correct_indices)} 个任务)
{'='*80}

【IoU 指标】
--------------------------------------------------------------------------------
初始平均 IoU: {avg_correct_initial_iou:.4f}
最终平均 IoU: {avg_correct_final_iou:.4f}
IoU 变化: {avg_correct_final_iou - avg_correct_initial_iou:+.4f}

【置信度指标】
--------------------------------------------------------------------------------
初始平均置信度: {avg_correct_initial_conf:.4f}
最终平均置信度: {avg_correct_final_conf:.4f}
置信度变化: {avg_correct_final_conf - avg_correct_initial_conf:+.4f}

【mAP 指标】
--------------------------------------------------------------------------------
"""
    
    for threshold in sorted(correct_initial_mAPs.keys()):
        summary += f"mAP@{threshold:.2f}:\n"
        summary += f"  初始: {correct_initial_mAPs[threshold]:.4f}\n"
        summary += f"  最终: {correct_final_mAPs[threshold]:.4f}\n"
        summary += f"  变化: {correct_final_mAPs[threshold] - correct_initial_mAPs[threshold]:+.4f}\n"
    
    summary += f"\n{'='*80}\n"
    
    print(summary)
    
    # 保存结果
    results = {
        'n_tasks': n_tasks,
        'num_task_type_correct': len(task_type_correct_indices),
        'task_type_correct_rate': len(task_type_correct_indices) / n_tasks,
        'all_tasks': {
            'avg_initial_iou': float(avg_initial_iou),
            'avg_final_iou': float(avg_final_iou),
            'iou_change': float(avg_final_iou - avg_initial_iou),
            'avg_initial_conf': float(avg_initial_conf),
            'avg_final_conf': float(avg_final_conf),
            'conf_change': float(avg_final_conf - avg_initial_conf),
            'initial_mAPs': {str(k): float(v) for k, v in initial_mAPs.items()},
            'final_mAPs': {str(k): float(v) for k, v in final_mAPs.items()},
        },
        'correct_tasks_only': {
            'avg_initial_iou': float(avg_correct_initial_iou),
            'avg_final_iou': float(avg_correct_final_iou),
            'iou_change': float(avg_correct_final_iou - avg_correct_initial_iou),
            'avg_initial_conf': float(avg_correct_initial_conf),
            'avg_final_conf': float(avg_correct_final_conf),
            'conf_change': float(avg_correct_final_conf - avg_correct_initial_conf),
            'initial_mAPs': {str(k): float(v) for k, v in correct_initial_mAPs.items()},
            'final_mAPs': {str(k): float(v) for k, v in correct_final_mAPs.items()},
        },
        'task_type_correct_indices': task_type_correct_indices
    }
    
    # 保存JSON结果
    results_path = os.path.join(output_dir, f"metrics_top_{n_tasks}_tasks.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ 结果已保存到: {results_path}")
    
    # 保存文本报告
    summary_path = os.path.join(output_dir, f"metrics_top_{n_tasks}_tasks.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✅ 文本报告已保存到: {summary_path}")
    
    # 保存前N个任务的预测结果
    predictions_initial_path = os.path.join(output_dir, f"predictions_top_{n_tasks}_initial.json")
    with open(predictions_initial_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_initial, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ 初始预测结果已保存到: {predictions_initial_path}")
    
    predictions_final_path = os.path.join(output_dir, f"predictions_top_{n_tasks}_final.json")
    with open(predictions_final_path, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_final, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"✅ 最终预测结果已保存到: {predictions_final_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="从checkpoint计算前N个任务的指标")
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='/data/tct/ActivePerception/exp/HM3D-grounding-gemini3flash/task_records',
                       help='checkpoint所在目录路径')
    parser.add_argument('--n_tasks', type=int, default=300,
                       help='计算前N个任务的指标')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录,默认与checkpoint_dir相同')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir
    
    # 加载checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到checkpoint文件: {checkpoint_path}")
        return
    
    print(f"📂 从 {checkpoint_path} 加载checkpoint...")
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    # 检查可用任务数
    total_available_tasks = checkpoint_data['last_completed_idx'] + 1
    print(f"✅ Checkpoint中共有 {total_available_tasks} 个已完成的任务")
    
    if args.n_tasks > total_available_tasks:
        print(f"⚠️  警告: 请求的任务数({args.n_tasks})超过可用任务数({total_available_tasks})")
        print(f"将计算前 {total_available_tasks} 个任务的指标")
        args.n_tasks = total_available_tasks
    
    # 计算指标
    calculate_metrics_for_n_tasks(checkpoint_data, args.n_tasks, args.output_dir)
    
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()

