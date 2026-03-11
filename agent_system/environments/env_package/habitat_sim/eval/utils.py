"""
评估工具函数：指标计算与可视化类
"""
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as mask_utils

from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import draw_bbox_with_text
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import visualize_bbox_on_image
from agent_system.environments.env_package.habitat_sim.utils.constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE
from agent_system.environments.prompts import HABITAT_UNIFIED_COT_TEMPLATE


# ========================================
# Common Utilities
# ========================================

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


def calculate_improvement(initial: float, final: float) -> str:
    """计算百分比变化"""
    if initial > 0:
        percent = ((final - initial) / initial) * 100
        return f"{percent:+.2f}%"
    return "N/A (初始值为0)"


# ========================================
# Metrics Base Class
# ========================================

class BaseMetrics(ABC):
    """指标计算基类"""
    
    @abstractmethod
    def calculate_metrics(self, pred, gt) -> dict:
        """计算预测与GT之间的指标"""
        pass
    
    @abstractmethod
    def aggregate_metrics(self, all_initial_metrics: List[dict], all_final_metrics: List[dict], 
                         task_type_correct_indices: List[int]) -> dict:
        """汇总所有任务的指标"""
        pass
    
    @abstractmethod
    def generate_summary(self, aggregated: dict, total_tasks: int, num_correct: int) -> str:
        """生成评估摘要文本"""
        pass


# ========================================
# Grounding Metrics
# ========================================

class GroundingMetrics(BaseMetrics):
    """Grounding 任务指标计算"""
    
    def __init__(self):
        self.iou_thresholds = [0.5, 0.75, 0.90, 0.95]
    
    @staticmethod
    def reshape_bbox(bbox, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
        """将bbox从原始尺寸缩放到新尺寸"""
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
    
    @staticmethod
    def reshape_bbox_xxyy(bbox, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
        """将bbox从(x1,y1,x2,y2)格式转换为(xmin,xmax,ymin,ymax)格式并缩放"""
        if bbox is None:
            return (0, 0, 0, 0)
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        x1, y1, x2, y2 = bbox
        x1_new = x1 * scale_x
        y1_new = y1 * scale_y
        x2_new = x2 * scale_x
        y2_new = y2 * scale_y
        return (x1_new, x2_new, y1_new, y2_new)
    
    def calculate_iou(self, box_gt, box_pred) -> float:
        """计算IoU"""
        if box_pred is None or box_gt is None:
            return 0.0
        box1 = self.reshape_bbox(box_pred)
        box2 = (box_gt[0], box_gt[2], box_gt[1], box_gt[3])
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
    
    @staticmethod
    def compute_ap(recalls, precisions) -> float:
        """计算 Average Precision (AP)"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        
        return ap
    
    def calculate_mAP(self, all_predictions: List[dict], iou_threshold: float = 0.5) -> Tuple[float, dict]:
        """计算 mAP (mean Average Precision)"""
        total_ap = 0.0
        num_phrases = len(all_predictions)
        ap_list = []
        
        for pred_data in all_predictions:
            gt_bbox = pred_data['gt_bbox']
            pred_bboxes = pred_data['pred_bboxes']
            pred_scores = pred_data['pred_scores']
            
            if len(pred_bboxes) == 0:
                ap = 0.0
            else:
                sorted_indices = np.argsort(pred_scores)[::-1]
                sorted_bboxes = [pred_bboxes[i] for i in sorted_indices]
                
                tp = []
                fp = []
                matched = False
                
                for bbox in sorted_bboxes:
                    iou = self.calculate_iou(gt_bbox, bbox)
                    if iou >= iou_threshold and not matched:
                        tp.append(1)
                        fp.append(0)
                        matched = True
                    else:
                        tp.append(0)
                        fp.append(1)
                
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                recalls = tp_cumsum / 1.0
                
                ap = self.compute_ap(recalls, precisions)
            
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
    
    def calculate_metrics(self, pred, gt) -> dict:
        """计算单个样本的指标"""
        iou = self.calculate_iou(gt, pred)
        return {'iou': iou}
    
    def aggregate_metrics(self, all_initial_metrics: List[dict], all_final_metrics: List[dict],
                         task_type_correct_indices: List[int]) -> dict:
        """汇总所有任务的指标"""
        all_initial_ious = [m['iou'] for m in all_initial_metrics]
        all_final_ious = [m['iou'] for m in all_final_metrics]
        
        avg_initial_iou = np.mean(all_initial_ious) if all_initial_ious else 0.0
        avg_final_iou = np.mean(all_final_ious) if all_final_ious else 0.0
        
        return {
            'avg_initial_iou': avg_initial_iou,
            'avg_final_iou': avg_final_iou,
            'all_initial_ious': all_initial_ious,
            'all_final_ious': all_final_ious
        }
    
    def generate_summary(self, aggregated: dict, total_tasks: int, num_correct: int,
                        initial_mAPs: dict = None, final_mAPs: dict = None,
                        avg_initial_conf: float = 0.0, avg_final_conf: float = 0.0) -> str:
        """生成评估摘要"""
        task_type_accuracy = num_correct / total_tasks if total_tasks > 0 else 0.0
        
        iou_improvement_str = calculate_improvement(aggregated['avg_initial_iou'], aggregated['avg_final_iou'])
        conf_improvement_str = calculate_improvement(avg_initial_conf, avg_final_conf)
        
        summary = (
            f"--- Evaluation Summary ---\n"
            f"Average Initial IoU:    {aggregated['avg_initial_iou']:.4f}\n"
            f"Average Final IoU:      {aggregated['avg_final_iou']:.4f} ({iou_improvement_str})\n"
            f"Average Initial Conf:   {avg_initial_conf:.4f}\n"
            f"Average Final Conf:     {avg_final_conf:.4f} ({conf_improvement_str})\n"
            f"\n"
            f"Task Type Prediction:\n"
            f"  Accuracy: {num_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)\n"
        )
        
        if initial_mAPs and final_mAPs:
            summary += f"\nmAP Results:\n"
            for threshold in self.iou_thresholds:
                improvement_str = calculate_improvement(initial_mAPs[threshold], final_mAPs[threshold])
                summary += (
                    f"  Initial mAP@{threshold}:      {initial_mAPs[threshold]:.4f}\n"
                    f"  Final mAP@{threshold}:        {final_mAPs[threshold]:.4f} ({improvement_str})\n"
                )
            
            avg_initial_mAP = np.mean(list(initial_mAPs.values()))
            avg_final_mAP = np.mean(list(final_mAPs.values()))
            avg_mAP_improvement_str = calculate_improvement(avg_initial_mAP, avg_final_mAP)
            summary += (
                f"\n"
                f"  Initial mAP@[0.5:0.95]: {avg_initial_mAP:.4f}\n"
                f"  Final mAP@[0.5:0.95]:   {avg_final_mAP:.4f} ({avg_mAP_improvement_str})"
            )
        
        return summary


# ========================================
# Segment Metrics
# ========================================

class SegmentMetrics(BaseMetrics):
    """Segment 任务指标计算"""
    
    @staticmethod
    def calculate_mask_iou(pred_mask_rle, gt_mask_rle) -> float:
        """计算两个 RLE 格式 mask 的 IoU"""
        if pred_mask_rle is None or gt_mask_rle is None:
            return 0.0
        
        try:
            pred_mask = mask_utils.decode(pred_mask_rle).astype(np.bool_)
            gt_mask = mask_utils.decode(gt_mask_rle).astype(np.bool_)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union == 0:
                return 0.0
            
            iou = intersection / union
            return float(iou)
        except Exception as e:
            print(f"Error calculating mask IoU: {e}")
            return 0.0
    
    @staticmethod
    def calculate_dice_score(pred_mask_rle, gt_mask_rle) -> float:
        """计算 Dice Score"""
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
    
    @staticmethod
    def calculate_pixel_accuracy(pred_mask_rle, gt_mask_rle) -> float:
        """计算像素级准确率"""
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
    
    @staticmethod
    def calculate_precision_recall(pred_mask_rle, gt_mask_rle) -> Tuple[float, float]:
        """计算 Precision 和 Recall"""
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
    
    @staticmethod
    def create_gt_mask_from_bbox(bbox_gt, image_size=(800, 640)):
        """从 bbox_gt 创建一个简单的矩形 mask"""
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
    
    def calculate_metrics(self, pred_mask_rle, gt_mask_rle) -> dict:
        """计算单个样本的所有指标"""
        mask_iou = self.calculate_mask_iou(pred_mask_rle, gt_mask_rle)
        dice = self.calculate_dice_score(pred_mask_rle, gt_mask_rle)
        pixel_acc = self.calculate_pixel_accuracy(pred_mask_rle, gt_mask_rle)
        precision, recall = self.calculate_precision_recall(pred_mask_rle, gt_mask_rle)
        
        return {
            'mask_iou': mask_iou,
            'dice': dice,
            'pixel_acc': pixel_acc,
            'precision': precision,
            'recall': recall
        }
    
    def aggregate_metrics(self, all_initial_metrics: List[dict], all_final_metrics: List[dict],
                         task_type_correct_indices: List[int]) -> dict:
        """汇总所有任务的指标"""
        def extract_metric(metrics_list, key):
            return [m[key] for m in metrics_list]
        
        result = {}
        for key in ['mask_iou', 'dice', 'pixel_acc', 'precision', 'recall']:
            initial_values = extract_metric(all_initial_metrics, key)
            final_values = extract_metric(all_final_metrics, key)
            
            result[f'avg_initial_{key}'] = np.mean(initial_values) if initial_values else 0.0
            result[f'avg_final_{key}'] = np.mean(final_values) if final_values else 0.0
            result[f'all_initial_{key}'] = initial_values
            result[f'all_final_{key}'] = final_values
        
        return result
    
    def generate_summary(self, aggregated: dict, total_tasks: int, num_correct: int,
                        avg_initial_conf: float = 0.0, avg_final_conf: float = 0.0) -> str:
        """生成评估摘要"""
        task_type_accuracy = num_correct / total_tasks if total_tasks > 0 else 0.0
        
        def calc_pct(key):
            initial = aggregated[f'avg_initial_{key}']
            final = aggregated[f'avg_final_{key}']
            return ((final - initial) / (initial + 1e-8)) * 100
        
        summary = f"""
========================================
    Segment Task Evaluation Summary
========================================
Total Tasks: {total_tasks}

Task Type Prediction:
  Accuracy: {num_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)

--- All Tasks ---
Mask IoU:
  Initial: {aggregated['avg_initial_mask_iou']:.4f}
  Final:   {aggregated['avg_final_mask_iou']:.4f}
  Change:  {aggregated['avg_final_mask_iou'] - aggregated['avg_initial_mask_iou']:+.4f} ({calc_pct('mask_iou'):+.2f}%)

Dice Score:
  Initial: {aggregated['avg_initial_dice']:.4f}
  Final:   {aggregated['avg_final_dice']:.4f}
  Change:  {aggregated['avg_final_dice'] - aggregated['avg_initial_dice']:+.4f} ({calc_pct('dice'):+.2f}%)

Pixel Accuracy:
  Initial: {aggregated['avg_initial_pixel_acc']:.4f}
  Final:   {aggregated['avg_final_pixel_acc']:.4f}
  Change:  {aggregated['avg_final_pixel_acc'] - aggregated['avg_initial_pixel_acc']:+.4f} ({calc_pct('pixel_acc'):+.2f}%)

Precision:
  Initial: {aggregated['avg_initial_precision']:.4f}
  Final:   {aggregated['avg_final_precision']:.4f}
  Change:  {aggregated['avg_final_precision'] - aggregated['avg_initial_precision']:+.4f} ({calc_pct('precision'):+.2f}%)

Recall:
  Initial: {aggregated['avg_initial_recall']:.4f}
  Final:   {aggregated['avg_final_recall']:.4f}
  Change:  {aggregated['avg_final_recall'] - aggregated['avg_initial_recall']:+.4f} ({calc_pct('recall'):+.2f}%)

Confidence Score:
  Initial: {avg_initial_conf:.4f}
  Final:   {avg_final_conf:.4f}
  Change:  {avg_final_conf - avg_initial_conf:+.4f} ({calculate_improvement(avg_initial_conf, avg_final_conf)})
========================================
"""
        return summary


# ========================================
# Visualizer Base Class
# ========================================

class BaseVisualizer(ABC):
    """可视化基类"""
    
    def __init__(self, record_dir: str):
        self.record_dir = record_dir
        os.makedirs(record_dir, exist_ok=True)
    
    def create_task_dir(self, task_idx: int) -> str:
        """创建任务目录"""
        task_dir = os.path.join(self.record_dir, f"task_{task_idx:04d}")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir
    
    @abstractmethod
    def draw_image(self, image: Image.Image, pred, gt, task_prompt: str = None) -> Image.Image:
        """绘制带标注的图像"""
        pass
    
    def save_step_image(self, task_dir: str, step: int, image: Image.Image, 
                       pred, gt, task_prompt: str = None, is_init: bool = False):
        """保存步骤图像"""
        annotated_image = self.draw_image(image, pred, gt, task_prompt)
        if is_init:
            filename = f"step_{step:03d}_init.png"
        else:
            filename = f"step_{step:03d}.png"
        filepath = os.path.join(task_dir, filename)
        annotated_image.save(filepath)
        return filename
    
    def save_task_record(self, task_dir: str, task_record: dict):
        """保存任务记录到JSON"""
        filepath = os.path.join(task_dir, "task_record.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task_record, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


# ========================================
# Grounding Visualizer
# ========================================

class GroundingVisualizer(BaseVisualizer):
    """Grounding 任务可视化 - 绘制 bbox"""
    
    def __init__(self, record_dir: str):
        super().__init__(record_dir)
        self.metrics = GroundingMetrics()
    
    def draw_image(self, image: Image.Image, bbox_pred=None, bbox_gt=None, 
                   task_prompt: str = None) -> Image.Image:
        """在图像上绘制 gt 和 pred 的 bbox"""
        result_image = image.copy()
        
        # 绘制 ground truth bbox (红色)
        if bbox_gt is not None and bbox_gt != (0, 0, 0, 0):
            result_image = draw_bbox_with_text(result_image, bbox_gt, text="gt", color="red", width=3)
        
        # 绘制 predicted bbox (绿色)
        if bbox_pred is not None and bbox_pred != (0, 0, 0, 0):
            bbox_xxyy = self.metrics.reshape_bbox_xxyy(bbox_pred)
            result_image = draw_bbox_with_text(result_image, bbox_xxyy, text="pred", color="green", width=3)
        
        return result_image


# ========================================
# Segment Visualizer
# ========================================

class SegmentVisualizer(BaseVisualizer):
    """Segment 任务可视化 - 绘制 mask overlay"""
    
    def __init__(self, record_dir: str):
        super().__init__(record_dir)
    
    def overlay_mask_on_image(self, image: Image.Image, mask_rle, 
                              color=(0, 255, 0), alpha=0.3) -> Image.Image:
        """在图像上叠加分割 mask"""
        if mask_rle is None:
            return image
        
        try:
            mask = mask_utils.decode(mask_rle)
            image_np = np.array(image)
            overlay = image_np.copy()
            overlay[mask > 0] = color
            blended = cv2.addWeighted(image_np, 1-alpha, overlay, alpha, 0)
            return Image.fromarray(blended)
        except Exception as e:
            print(f"Error overlaying mask: {e}")
            return image
    
    def draw_image(self, image: Image.Image, pred_mask_rle=None, bbox_gt=None,
                   task_prompt: str = None) -> Image.Image:
        """绘制分割结果：显示预测 mask 和 GT bbox"""
        img = image.copy()
        
        # 1. 绘制 GT bbox (红色)
        if bbox_gt is not None and sum(bbox_gt) > 0:
            img = draw_bbox_with_text(img, bbox_gt, text="gt", color="red", width=3)
        
        # 2. 叠加预测 mask (绿色半透明)
        if pred_mask_rle is not None:
            try:
                mask = mask_utils.decode(pred_mask_rle)
                image_np = np.array(img)
                overlay = image_np.copy()
                overlay[mask > 0] = [0, 255, 0]  # 绿色
                blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
                img = Image.fromarray(blended)
            except Exception as e:
                print(f"Error in visualization: {e}")
        
        return img


# ========================================
# 3D-Box Metrics
# ========================================

class Box3DMetrics(BaseMetrics):
    """3D-Box 任务指标计算"""
    
    def __init__(self):
        # 延迟导入，避免循环依赖
        pass
    
    @staticmethod
    def calculate_3d_iou(pred_bbox_3d, gt_bbox_3d) -> float:
        """计算两个3D bounding box的IoU (使用OBB模式)"""
        if pred_bbox_3d is None or gt_bbox_3d is None:
            return 0.0
        
        try:
            from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import compute_bbox_iou_3d
            iou = compute_bbox_iou_3d(pred_bbox_3d, gt_bbox_3d, use_obb=True)
            return float(iou)
        except Exception as e:
            print(f"Error calculating 3D IoU: {e}")
            return 0.0
    
    @staticmethod
    def calculate_center_distance_score(pred_bbox_3d, gt_bbox_3d) -> float:
        """
        计算3D bbox中心点的距离得分
        使用截断处理: max(0, 1 - d)，让指标处于 0~1 的范围
        距离越近得分越高（1表示完美匹配，0表示距离>=1米）
        
        Args:
            pred_bbox_3d: 预测的3D bbox字典，包含'obb_center'字段
            gt_bbox_3d: Ground truth 3D bbox字典
            
        Returns:
            center distance score (float), 范围 [0, 1]
        """
        if pred_bbox_3d is None or gt_bbox_3d is None:
            return 0.0
        
        try:
            pred_center = pred_bbox_3d.get('obb_center')
            gt_center = gt_bbox_3d.get('obb_center')
            
            if pred_center is None or gt_center is None:
                return 0.0
            
            # 计算欧氏距离
            pred_center = np.array(pred_center)
            gt_center = np.array(gt_center)
            distance = np.linalg.norm(pred_center - gt_center)
            
            # 截断处理: max(0, 1 - d)
            score = max(0.0, 1.0 - distance)
            return float(score)
        except Exception as e:
            print(f"Error calculating center distance score: {e}")
            return 0.0
    
    def calculate_metrics(self, pred_bbox_3d, gt_bbox_3d) -> dict:
        """计算单个样本的所有指标"""
        iou_3d = self.calculate_3d_iou(pred_bbox_3d, gt_bbox_3d)
        center_score = self.calculate_center_distance_score(pred_bbox_3d, gt_bbox_3d)
        
        return {
            'iou_3d': iou_3d,
            'center_score': center_score
        }
    
    def aggregate_metrics(self, all_initial_metrics: List[dict], all_final_metrics: List[dict],
                         task_type_correct_indices: List[int]) -> dict:
        """汇总所有任务的指标"""
        def extract_metric(metrics_list, key):
            return [m[key] for m in metrics_list]
        
        result = {}
        for key in ['iou_3d', 'center_score']:
            initial_values = extract_metric(all_initial_metrics, key)
            final_values = extract_metric(all_final_metrics, key)
            
            result[f'avg_initial_{key}'] = np.mean(initial_values) if initial_values else 0.0
            result[f'avg_final_{key}'] = np.mean(final_values) if final_values else 0.0
            result[f'all_initial_{key}'] = initial_values
            result[f'all_final_{key}'] = final_values
        
        return result
    
    def generate_summary(self, aggregated: dict, total_tasks: int, num_correct: int,
                        avg_initial_conf: float = 0.0, avg_final_conf: float = 0.0) -> str:
        """生成评估摘要"""
        task_type_accuracy = num_correct / total_tasks if total_tasks > 0 else 0.0
        
        def calc_pct(key):
            initial = aggregated[f'avg_initial_{key}']
            final = aggregated[f'avg_final_{key}']
            return ((final - initial) / (initial + 1e-8)) * 100
        
        summary = f"""
========================================
    3D-Box Task Evaluation Summary
========================================
Total Tasks: {total_tasks}

Task Type Prediction:
  Accuracy: {num_correct}/{total_tasks} = {task_type_accuracy:.4f} ({task_type_accuracy*100:.2f}%)

--- All Tasks ---
3D IoU:
  Initial: {aggregated['avg_initial_iou_3d']:.4f}
  Final:   {aggregated['avg_final_iou_3d']:.4f}
  Change:  {aggregated['avg_final_iou_3d'] - aggregated['avg_initial_iou_3d']:+.4f} ({calc_pct('iou_3d'):+.2f}%)

Center Distance Score (max(0, 1-d)):
  Initial: {aggregated['avg_initial_center_score']:.4f}
  Final:   {aggregated['avg_final_center_score']:.4f}
  Change:  {aggregated['avg_final_center_score'] - aggregated['avg_initial_center_score']:+.4f} ({calc_pct('center_score'):+.2f}%)

Confidence Score:
  Initial: {avg_initial_conf:.4f}
  Final:   {avg_final_conf:.4f}
  Change:  {avg_final_conf - avg_initial_conf:+.4f} ({calculate_improvement(avg_initial_conf, avg_final_conf)})
========================================
"""
        return summary


# ========================================
# 3D-Box Visualizer
# ========================================

class Box3DVisualizer(BaseVisualizer):
    """3D-Box 任务可视化"""
    
    def __init__(self, record_dir: str):
        super().__init__(record_dir)
        # 延迟导入可视化函数
        self._visualize_task = None
    
    def visualize_task(self, obs_pil, task_prompt, gt_bbox_3d, pred_bbox_3d, task_id, hfov=90.0):
        # 转换为 numpy 数组
        img_np = np.array(obs_pil)
        
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
        return result_img

    def draw_image(self, image: Image.Image, pred_bbox_3d=None, gt_bbox_3d=None,
                   task_prompt: str = None, task_id: int = None) -> Image.Image:
        """绘制3D bbox可视化结果"""
        try:
            return self.visualize_task(image, task_prompt, gt_bbox_3d, pred_bbox_3d, task_id)
        except Exception as e:
            print(f"Error in 3D-Box visualization: {e}")
            return image
    
    def save_step_image(self, task_dir: str, step: int, image: Image.Image,
                       pred_bbox_3d=None, gt_bbox_3d=None, task_prompt: str = None,
                       task_id: int = None, is_init: bool = False):
        """保存步骤图像（重写以支持额外参数）"""
        annotated_image = self.draw_image(image, pred_bbox_3d, gt_bbox_3d, task_prompt, task_id)
        if is_init:
            filename = f"step_{step:03d}_init.png"
        else:
            filename = f"step_{step:03d}.png"
        filepath = os.path.join(task_dir, filename)
        annotated_image.save(filepath)
        return filename
