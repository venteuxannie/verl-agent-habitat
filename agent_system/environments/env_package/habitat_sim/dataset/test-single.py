import os
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"

import sys
import argparse
from typing import Dict, List, Optional, Tuple

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.prompts import HABITAT_UNIFIED_COT_TEMPLATE
from agent_system.multi_turn_rollout.utils import process_image
from agent_system.environments.env_package.habitat_sim.utils.constants import SCENE_DATA_PATH, TEMP_DIR, HM3D_TRAINING_SCENES
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import visualize_bbox_on_image
from agent_system.environments.env_package.habitat_sim.utils.third_party import reshape_bbox_xxyy
from agent_system.environments.env_package.habitat_sim.utils.reward_function import reward_function

data_path = SCENE_DATA_PATH

from tqdm import tqdm
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pycocotools import mask as mask_utils
import cv2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 图像标注功能函数
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_default_font(size: int = 20):
    """获取默认字体，优先使用系统字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    # 使用默认字体
    return ImageFont.load_default()

def draw_text_on_image(
    image: Image.Image,
    texts: List[str],
    position: str = "top-left",
    font_size: int = 16,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 180),
    padding: int = 8,
    line_spacing: int = 4
) -> Image.Image:
    """
    在图像上绘制多行文本标注。

    Args:
        image: PIL图像
        texts: 要绘制的文本列表，每个元素一行
        position: 文本位置 ("top-left", "top-right", "bottom-left", "bottom-right")
        font_size: 字体大小
        text_color: 文本颜色 (R, G, B)
        bg_color: 背景颜色 (R, G, B, A)
        padding: 内边距
        line_spacing: 行间距

    Returns:
        标注后的图像
    """
    # 转换为RGBA以支持透明度
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = get_default_font(font_size)

    # 计算文本区域大小
    text_heights = []
    text_widths = []
    for text in texts:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_widths.append(bbox[2] - bbox[0])
        text_heights.append(bbox[3] - bbox[1])

    total_height = sum(text_heights) + line_spacing * (len(texts) - 1) + 2 * padding
    max_width = max(text_widths) + 2 * padding

    # 确定位置
    img_width, img_height = image.size
    if "left" in position:
        x = padding
    else:
        x = img_width - max_width - padding

    if "top" in position:
        y = padding
    else:
        y = img_height - total_height - padding

    # 绘制半透明背景
    draw.rectangle(
        [x - padding, y - padding, x + max_width, y + total_height - padding],
        fill=bg_color
    )

    # 绘制文本
    current_y = y
    for i, text in enumerate(texts):
        draw.text((x, current_y), text, font=font, fill=text_color + (255,))
        current_y += text_heights[i] + line_spacing

    # 合并图层
    result = Image.alpha_composite(image_rgba, overlay)
    return result.convert("RGB")

def draw_2d_bbox_on_image(
    image: Image.Image,
    bbox: List[int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    label: Optional[str] = None,
    font_size: int = 14
) -> Image.Image:
    """
    在图像上绘制2D边界框。

    Args:
        image: PIL图像
        bbox: 边界框 [x1, y1, x2, y2]
        color: 边框颜色 (R, G, B)
        thickness: 线条粗细
        label: 可选的标签文本
        font_size: 标签字体大小

    Returns:
        绘制边界框后的图像
    """
    if bbox is None or len(bbox) != 4 or sum(bbox) == 0:
        return image

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    x1, y1, x2, y2 = bbox

    # 绘制边界框
    for i in range(thickness):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

    # 如果有标签，绘制标签
    if label:
        font = get_default_font(font_size)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 标签背景
        label_bg = [x1, y1 - text_height - 6, x1 + text_width + 8, y1]
        if label_bg[1] < 0:  # 如果超出顶部，放到框内
            label_bg = [x1, y1, x1 + text_width + 8, y1 + text_height + 6]

        draw.rectangle(label_bg, fill=color)
        draw.text((label_bg[0] + 4, label_bg[1] + 2), label, font=font, fill=(255, 255, 255))

    return image_copy

def draw_mask_on_image(
    image: Image.Image,
    mask_rle: dict,
    color: Tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.4,
    draw_contour: bool = True,
    contour_thickness: int = 2
) -> Image.Image:
    """
    在图像上绘制分割掩码。

    Args:
        image: PIL图像
        mask_rle: RLE格式的掩码
        color: 掩码颜色 (R, G, B)
        alpha: 掩码透明度
        draw_contour: 是否绘制轮廓线
        contour_thickness: 轮廓线粗细

    Returns:
        绘制掩码后的图像
    """
    if mask_rle is None:
        return image

    try:
        # 解码RLE掩码
        mask_binary = mask_utils.decode(mask_rle)

        # 转换为numpy数组
        image_np = np.array(image)

        # 创建彩色掩码层
        mask_layer = np.zeros_like(image_np)
        mask_layer[mask_binary > 0] = color

        # 混合掩码和原图
        result = image_np.copy()
        mask_indices = mask_binary > 0
        result[mask_indices] = (
            image_np[mask_indices] * (1 - alpha) + mask_layer[mask_indices] * alpha
        ).astype(np.uint8)

        # 绘制轮廓
        if draw_contour:
            contours, _ = cv2.findContours(
                mask_binary.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, contour_thickness)

        return Image.fromarray(result)
    except Exception as e:
        print(f"Error drawing mask: {e}")
        return image

def draw_3d_bbox_on_image(
    image: Image.Image,
    bbox_3d: dict,
    hfov: float = 90.0,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> Image.Image:
    """
    在图像上绘制3D边界框（复用habitat_3dbox_utils的函数）。

    Args:
        image: PIL图像
        bbox_3d: 3D边界框字典（相机坐标系），包含 corners 等信息
        hfov: 水平视场角（degrees）
        color: 边框颜色 (R, G, B)
        thickness: 线条粗细

    Returns:
        绘制3D边界框后的图像
    """
    if bbox_3d is None or 'corners' not in bbox_3d:
        return image

    try:
        # 转换为numpy数组
        image_np = np.array(image)

        # 使用habitat_3dbox_utils中的函数绘制3D bbox
        result_np = visualize_bbox_on_image(
            image=image_np,
            bbox_3d=bbox_3d,
            hfov=hfov,
            color=color,
            thickness=thickness,
            use_obb=True
        )

        return Image.fromarray(result_np)
    except Exception as e:
        print(f"Error drawing 3D bbox: {e}")
        return image

def annotate_image(
    image: Image.Image,
    task_type: str,
    target_category: str,
    task_prompt: str,
    pred: Optional[any] = None,
    conf_score: float = 0.0,
    step_id: int = 0,
    image_size: Tuple[int, int] = (512, 512),
    hfov: float = 90.0,
    # 额外的标注数据（用于绘制所有类型的标注）
    bbox_2d: Optional[List[int]] = None,      # 2D边界框 [x1, y1, x2, y2]
    mask_rle: Optional[dict] = None,           # RLE格式的分割掩码
    bbox_3d: Optional[dict] = None,            # 3D边界框字典
    # 标注选项
    draw_text: bool = True,
    draw_bbox: bool = True,
    draw_mask: bool = True,
    draw_3dbox: bool = True,
    draw_all_annotations: bool = True,         # 是否绘制所有类型的标注（不管任务类型）
    # 颜色配置
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    mask_color: Tuple[int, int, int] = (255, 0, 255),
    box3d_color: Tuple[int, int, int] = (0, 0, 255),
) -> Image.Image:
    """
    根据任务类型对图像进行标注。支持绘制所有三种标注类型。

    Args:
        image: 原始PIL图像
        task_type: 任务类型 ("grounding", "segment", "3d-box")
        target_category: 目标类别
        task_prompt: 任务提示
        pred: 预测结果（根据任务类型不同：bbox、mask或3d_bbox）
        conf_score: 置信度分数
        step_id: 当前步骤ID
        image_size: 图像尺寸 (W, H)
        hfov: 水平视场角
        bbox_2d: 额外提供的2D边界框（用于绘制所有标注）
        mask_rle: 额外提供的分割掩码（用于绘制所有标注）
        bbox_3d: 额外提供的3D边界框（用于绘制所有标注）
        draw_text: 是否绘制文本标注
        draw_bbox: 是否绘制2D边界框
        draw_mask: 是否绘制分割掩码
        draw_3dbox: 是否绘制3D边界框
        draw_all_annotations: 是否绘制所有类型的标注（不管任务类型）
        bbox_color: 2D边界框颜色
        mask_color: 分割掩码颜色
        box3d_color: 3D边界框颜色

    Returns:
        标注后的图像
    """
    annotated_image = image.copy()
    if bbox_2d is not None:
        bbox_2d_xyxy = [bbox_2d[0], bbox_2d[2], bbox_2d[1], bbox_2d[3]]
    else:
        bbox_2d_xyxy = None

    # 从pred中提取对应任务类型的数据
    pred_bbox_2d = None
    pred_mask_rle = None
    pred_bbox_3d = None

    if task_type == "grounding" and pred is not None:
        print(f"DEBUG: pred={pred}")
        pred_bbox_2d = reshape_bbox_xxyy(pred, reverse_order=True)
    elif task_type == "segment" and pred and isinstance(pred, dict) and "segment_mask" in pred:
        pred_mask_rle = pred["segment_mask"]
    elif task_type == "3d-box" and pred and isinstance(pred, dict) and 'corners' in pred:
        pred_bbox_3d = pred

    # 使用额外提供的数据或pred中的数据
    final_bbox_2d = bbox_2d_xyxy if bbox_2d_xyxy is not None else pred_bbox_2d
    print(f"DEBUG: bbox-gt: {bbox_2d is not None} ")
    final_mask_rle = mask_rle if mask_rle is not None else pred_mask_rle
    final_bbox_3d = bbox_3d if bbox_3d is not None else pred_bbox_3d

    if draw_all_annotations:
        # === 绘制所有三种标注（不管任务类型） ===

        # 1. 绘制分割掩码（先画，这样不会覆盖其他标注）
        if draw_mask and final_mask_rle is not None:
            annotated_image = draw_mask_on_image(
                annotated_image,
                mask_rle=final_mask_rle,
                color=mask_color,
                alpha=0.4,
                draw_contour=True
            )

        # 2. 绘制3D边界框
        if draw_3dbox and final_bbox_3d is not None and 'corners' in final_bbox_3d:
            annotated_image = draw_3d_bbox_on_image(
                annotated_image,
                bbox_3d=final_bbox_3d,
                hfov=hfov,
                color=box3d_color,
                thickness=2
            )

        # 3. 绘制2D边界框（最后画，这样在最上层）
        if draw_bbox:
            # 尝试从不同来源获取2D bbox
            bbox_to_draw = final_bbox_2d

            if bbox_to_draw is not None:
                print(f"DEBUG: bbox_to_draw={bbox_to_draw}")
                annotated_image = draw_2d_bbox_on_image(
                    annotated_image,
                    bbox=bbox_to_draw,
                    color=bbox_color,
                    thickness=3,
                    label=target_category
                )
    else:
        # === 原有逻辑：只绘制对应任务类型的标注 ===
        if task_type == "grounding" and draw_bbox:
            annotated_image = draw_2d_bbox_on_image(
                annotated_image,
                bbox=pred,
                color=bbox_color,
                thickness=3,
                label=target_category
            )

        elif task_type == "segment" and draw_mask:
            if pred and isinstance(pred, dict) and "segment_mask" in pred:
                annotated_image = draw_mask_on_image(
                    annotated_image,
                    mask_rle=pred["segment_mask"],
                    color=mask_color,
                    alpha=0.4,
                    draw_contour=True
                )
                if draw_bbox:
                    bbox = extract_bbox_from_mask(pred["segment_mask"])
                    if bbox:
                        annotated_image = draw_2d_bbox_on_image(
                            annotated_image,
                            bbox=bbox,
                            color=bbox_color,
                            thickness=2,
                            label=target_category
                        )

        elif task_type == "3d-box" and draw_3dbox:
            if pred and isinstance(pred, dict) and 'corners' in pred:
                annotated_image = draw_3d_bbox_on_image(
                    annotated_image,
                    bbox_3d=pred,
                    hfov=hfov,
                    color=box3d_color,
                    thickness=2
                )
                if draw_bbox:
                    bbox = extract_bbox_from_3dbox(pred, image_size=image_size, hfov=hfov)
                    if bbox:
                        annotated_image = draw_2d_bbox_on_image(
                            annotated_image,
                            bbox=bbox,
                            color=bbox_color,
                            thickness=2,
                            label=target_category
                        )

    # 绘制文本标注
    if draw_text:
        text_lines = [
            f"Task: {task_type}",
            f"Target: {target_category}",
            f"Score: {conf_score:.3f}",
            f"Step: {step_id}",
        ]
        # 如果task_prompt不太长，也显示
        if len(task_prompt) <= 50:
            text_lines.append(f"Prompt: {task_prompt}")
        else:
            text_lines.append(f"Prompt: {task_prompt[:47]}...")

        annotated_image = draw_text_on_image(
            annotated_image,
            texts=text_lines,
            position="top-left",
            font_size=14,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0, 180)
        )

    return annotated_image

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ (ADD) 1. 定义新的基于规则的策略函数和其超参数
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 策略的超参数，将它们放在这里方便统一调整
POLICY_THRESHOLDS = {
    # 水平中心区域的容忍度比例 (占图像宽度)
    "HORIZONTAL_CENTER_RATIO": 0.15,
    # 垂直中心区域的容忍度比例 (占图像高度)
    "VERTICAL_CENTER_RATIO": 0.15,
    # 目标边界框面积占总图像面积的理想最小比例
    "TARGET_BBOX_AREA_RATIO": 0.1,
    # 边界框面积的最大比例，超过则认为太近
    "MAX_BBOX_AREA_RATIO": 0.5,
}

def get_rule_based_action(
    bbox: Optional[List[int]],
    image_width: int,
    image_height: int,
    action_space: List[str],
    thresholds: Dict[str, float]
) -> str:
    """
    根据目标边界框的位置和大小，决定下一步最优动作。
    这是一个分层、基于优先级的策略：先对准，再靠近。

    Args:
        bbox: 目标物体的边界框 [x1, y1, x2, y2]。如果为 None 或无效，则执行搜索。
        image_width: 图像宽度。
        image_height: 图像高度。
        action_space: 可用动作列表。
        thresholds: 包含所有阈值的字典。

    Returns:
        action_name (str): 决策出的最佳动作名称。
    """
    # --- 优先级 0: 搜索 ---
    # 如果没有检测到边界框 (bbox是None或全0)，则执行搜索
    if not bbox or sum(bbox) == 0:
        return "turn_right"  # 原地右转进行搜索

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    bbox_area = (x2 - x1) * (y2 - y1)

    # 从字典中获取阈值
    horizontal_tolerance = image_width * thresholds["HORIZONTAL_CENTER_RATIO"]
    vertical_tolerance = image_height * thresholds["VERTICAL_CENTER_RATIO"]
    min_target_area = image_width * image_height * thresholds["TARGET_BBOX_AREA_RATIO"]
    max_target_area = image_width * image_height * thresholds["MAX_BBOX_AREA_RATIO"]

    # --- 优先级 1: 水平对准 ---
    if bbox_center_x < image_center_x - horizontal_tolerance:
        # 物体在视野左侧，智能体需向左转
        return "turn_left"
    elif bbox_center_x > image_center_x + horizontal_tolerance:
        # 物体在视野右侧，智能体需向右转
        return "turn_right"

    # --- 优先级 2: 垂直对准 ---
    elif bbox_center_y < image_center_y - vertical_tolerance:
        # 物体在视野上方 (图像y坐标小)，智能体需抬头
        return "look_up"
    elif bbox_center_y > image_center_y + vertical_tolerance:
        # 物体在视野下方 (图像y坐标大)，智能体需低头
        return "look_down"

    # --- 优先级 3: 调整距离 ---
    elif bbox_area < min_target_area:
        # 物体已居中但太小，需要前进
        return "move_forward"

    # --- 满足所有条件: 理想状态 ---
    else:
        # 物体已居中，大小也合适 (介于min和max之间，或大于max)
        # 此时认为已达到最佳观测位置，可以停止。
        return "stop"

def extract_bbox_from_mask(mask_rle):
    """
    从 RLE 格式的 mask 中提取边界框。

    Args:
        mask_rle: RLE 格式的 mask 字典

    Returns:
        bbox [xmin, ymin, xmax, ymax] 或 None
    """
    if mask_rle is None:
        return None
    try:
        # 解码 RLE 格式的 mask
        mask_binary = mask_utils.decode(mask_rle)

        # 找到 mask 的边界
        rows = np.any(mask_binary, axis=1)
        cols = np.any(mask_binary, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return [int(xmin), int(ymin), int(xmax), int(ymax)]
    except Exception as e:
        print(f"Error extracting bbox from mask: {e}")
        return None

def extract_bbox_from_3dbox(bbox_3d, image_size=(512, 512), hfov=90.0):
    """
    从 3D bbox 投影提取 2D 边界框。

    Args:
        bbox_3d: 3D bbox 字典，包含 corners 等信息（相机坐标系）
        image_size: 图像尺寸 (W, H)
        hfov: 水平视场角（degrees）

    Returns:
        bbox [xmin, ymin, xmax, ymax] 或 None
    """
    if bbox_3d is None:
        return None

    try:
        W, H = image_size

        # 计算相机内参
        focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cx = W / 2.0
        cy = H / 2.0

        # 获取8个角点（相机坐标系）
        corners = bbox_3d.get('corners')
        if corners is None:
            return None

        # 投影到2D
        projected_points = []
        for corner in corners:
            x, y, z = corner
            if z >= 0:  # 在相机后面，跳过
                continue
            # 投影公式
            u = x * focal_length / (-z) + cx
            v = -y * focal_length / (-z) + cy
            # 不限制在图像内，允许部分在外
            projected_points.append((u, v))

        if len(projected_points) == 0:
            return None

        # 计算所有投影点的边界框
        projected_array = np.array(projected_points)
        xmin = int(np.floor(np.min(projected_array[:, 0])))
        ymin = int(np.floor(np.min(projected_array[:, 1])))
        xmax = int(np.ceil(np.max(projected_array[:, 0])))
        ymax = int(np.ceil(np.max(projected_array[:, 1])))

        # 裁剪到图像范围内
        xmin = max(0, min(W - 1, xmin))
        ymin = max(0, min(H - 1, ymin))
        xmax = max(0, min(W - 1, xmax))
        ymax = max(0, min(H - 1, ymax))

        # 验证 bbox 有效性
        if xmax <= xmin or ymax <= ymin:
            return None

        return [xmin, ymin, xmax, ymax]

    except Exception as e:
        print(f"Error extracting bbox from 3D bbox: {e}")
        return None

def get_scene_path(subfolders, dataset_name, eval_id=0):
    """Constructs the full path to a habitat scene file."""
    if dataset_name == "HM3D":
        dataset_path = os.path.join(data_path, "hm3d/train")
        scene = subfolders[eval_id % len(subfolders)]
        id = scene.split('-')[1]
        scene_path = f"{scene}/{id}.basis.glb"
        scene_id = os.path.join(dataset_path, scene_path)
    elif dataset_name == "ReplicaCAD":
        scene_name = subfolders[eval_id % len(subfolders)]
        scene_id = os.path.join(data_path, "replica_cad/configs/scenes", scene_name)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return scene_id

def build_text_obs(infos: List[Dict]) -> List[str]:
    """
    Builds the text observation (prompt) for the agent.
    Uses the unified prompt template with VLM-generated task_description.
    """
    postprocess_text_obs = []
    for i in range(len(infos)):
        # Get task_description from info (generated by VLM service in habitat_envs.py)
        task_description = infos[i].get('task_description', '')
        conf_score = infos[i].get('conf_score') or 0.0

        # Use the unified template
        prompt = HABITAT_UNIFIED_COT_TEMPLATE.format(
            task_description=task_description,
            conf_score=conf_score
        )
        postprocess_text_obs.append(prompt)
    return postprocess_text_obs

def json_entry(id, prompt, thoughts, task_type, task_prompt, action):
    """
    构造单条SFT数据记录。

    输出格式符合新的统一模板要求，包含 thoughts, task_type, task_prompt, action。
    """
    id_str = format(id, "06d")
    image_name = f"{id_str}.png"

    # 新的输出格式包含 task_type 和 task_prompt
    gpt_response = f'''{{"thoughts": "{thoughts}",
"task_type": "{task_type}",
"task_prompt": "{task_prompt}",
"action": "{action}"
}}'''

    conversations = [
        {"from": "human", "value": f"{prompt}"},
        {"from": "gpt", "value": gpt_response}
    ]

    return {"image": image_name, "conversations": conversations}, image_name

def get_object_location_description(
    image_width: int,
    image_height: int,
    bbox: List[int]
) -> str:
    """
    Generates a sentence describing an object's approximate location within an
    image, based on its bounding box.
    """
    if not bbox or not all(isinstance(val, (int, float)) for val in bbox) or len(bbox) != 4 or sum(bbox) == 0:
        return "The target object is not currently visible."

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if center_y < image_height / 3: vertical_pos = "top"
    elif center_y <= image_height * 2 / 3: vertical_pos = "middle"
    else: vertical_pos = "bottom"

    if center_x < image_width / 3: horizontal_pos = "left"
    elif center_x <= image_width * 2 / 3: horizontal_pos = "center"
    else: horizontal_pos = "right"

    location_map = {
        ('top', 'left'): "The object is in the top-left corner of the image.",
        ('top', 'center'): "The object is at the top-center of the image.",
        ('top', 'right'): "The object is in the top-right corner of the image.",
        ('middle', 'left'): "The object is on the left side of the image.",
        ('middle', 'center'): "The object is in the center of the image.",
        ('middle', 'right'): "The object is on the right side of the image.",
        ('bottom', 'left'): "The object is in the bottom-left corner of the image.",
        ('bottom', 'center'): "The object is at the bottom-center of the image.",
        ('bottom', 'right'): "The object is in the bottom-right corner of the image.",
    }
    return location_map.get((vertical_pos, horizontal_pos), "The location of the object could not be determined.")

def get_CoT_label(
    task_type: str,
    task_description: str,
    image_width: int,
    image_height: int,
    bbox: Optional[List[int]],
    conf_score: float,
    action: str,
    pred: Optional[any] = None
) -> str:
    """
    Generates a Chain-of-Thought (CoT) label based on task type, bounding box, and confidence score.

    The CoT now includes reasoning about task type inference from task_description.

    Args:
        task_type: Type of task ("grounding", "segment", "3d-box")
        task_description: Natural language task description (from VLM)
        image_width: Image width
        image_height: Image height
        bbox: Bounding box for the target object
        conf_score: Confidence score
        action: Action to take
        pred: Prediction result (varies by task type)

    Returns:
        CoT string
    """
    # Task type reasoning based on task_description
    task_type_reasoning = {
        "grounding": "The task description asks to find and locate an object, so this is a grounding task.",
        "segment": "The task description asks to segment an object, so this is a segmentation task.",
        "3d-box": "The task description asks to predict the 3D bounding box, so this is a 3d-box task.",
    }

    # Get location description from bbox
    location_description = get_object_location_description(image_width, image_height, bbox)

    # Add distance description based on action
    if action == "move_forward":
        location_description += " The object appears small, indicating that I am far from it."
    elif action == "stop":
        location_description += " The object appears to be at an appropriate distance."

    # Build CoT with task type reasoning
    CoT = task_type_reasoning.get(task_type, f"Based on the task description, this appears to be a {task_type} task.")
    CoT += f" {location_description}"
    CoT += f" The current score is {conf_score:.3f}."
    CoT += f" To improve the score, I should choose action '{action}'."

    return CoT

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 单场景处理函数（无 Ray）
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def collect_single_scene(
    dataset_name: str,
    scene_idx: int,
    scene_subfolder: List[str],
    image_folder: str,
    annotated_folder: str,
    enable_annotation: bool,
    save_annotated_only: bool,
    annotation_options: Dict,
    max_scene_instance: int,
    max_step_length: int,
    seed: int,
    scenes_size: int
) -> Dict:
    """
    收集单个场景的 SFT 数据（无 Ray 版本）。

    Args:
        dataset_name: 数据集名称
        scene_idx: 场景索引
        scene_subfolder: 场景子文件夹列表
        image_folder: 图像保存路径
        annotated_folder: 标注图像保存路径
        enable_annotation: 是否启用图像标注
        save_annotated_only: 是否只保存标注图像
        annotation_options: 标注配置选项
        max_scene_instance: 每个场景的最大实例数
        max_step_length: 每个实例的最大步数
        seed: 随机种子
        scenes_size: 场景总数

    Returns:
        包含场景结果和数据集条目的字典
    """
    # 创建环境
    env = CreateHabitatEnv(
        seed,
        dataset_name,
        scenes_size,
        max_scene_instance,
        max_step_length
    )

    scene_name = scene_subfolder[scene_idx]
    special_scene_id = get_scene_path(scene_subfolder, dataset_name, eval_id=scene_idx)

    dataset_entries = []
    step_id = 0
    collected_steps = 0

    print(f"Processing scene: {scene_name}")
    print(f"Scene path: {special_scene_id}")
    print(f"Max instances: {max_scene_instance}, Max steps per instance: {max_step_length}")
    print()

    for j in tqdm(range(max_scene_instance), desc=f"Instances"):
        try:
            obs, info = env.reset(
                seed=seed,
                is_unique=True,
                sync_info=None,
                special_scene_id=special_scene_id,
                task_type="grounding",
            )
            image_width, image_height = obs.size
            action_space = env.action_space

            # Prepare prompt for the model
            prompt = build_text_obs([info])[0]

            # Get pred after reset
            env_obs = env.sim.get_sensor_observations()
            _, _, info["conf_score"], _, _, info["pred"], _ = reward_function(
                True, info["task_type"], env_obs, info["task_prompt"], 0, True, "move_forward", 0,
                env.sim.get_agent(0).get_state()
            )

            for k in range(max_step_length):
                # Extract bbox based on task type
                bbox_for_action = None
                task_type = info["task_type"]
                task_description = info.get("task_description", "")
                pred = info.get("pred")

                if task_type == "grounding":
                    bbox_for_action = reshape_bbox_xxyy(pred, reverse_order=True)
                elif task_type == "segment":
                    if pred and isinstance(pred, dict) and "segment_mask" in pred:
                        bbox_for_action = extract_bbox_from_mask(pred["segment_mask"])
                elif task_type == "3d-box":
                    bbox_for_action = extract_bbox_from_3dbox(pred, image_size=(image_width, image_height), hfov=90.0)
                else:
                    continue  # Skip unsupported task types

                action = get_rule_based_action(
                    bbox=bbox_for_action,
                    image_width=image_width,
                    image_height=image_height,
                    action_space=action_space,
                    thresholds=POLICY_THRESHOLDS
                )

                # Generate CoT label
                thoughts = get_CoT_label(
                    task_type=task_type,
                    task_description=task_description,
                    image_width=image_width,
                    image_height=image_height,
                    bbox=bbox_for_action,
                    conf_score=info.get("conf_score") or 0.0,
                    action=action,
                )

                task_prompt = info.get("task_prompt", "")
                target_category = info.get("target_category", "unknown")

                # Create dataset entry
                entry, image_name = json_entry(step_id, prompt, thoughts, task_type, task_prompt, action)
                dataset_entries.append(entry)

                # Save original image
                if not enable_annotation or not save_annotated_only:
                    obs.save(os.path.join(image_folder, image_name))

                # Save annotated image
                if enable_annotation:
                    gt_data = info.get("gt", {}) or {}

                    bbox_2d_gt = gt_data.get("bbox_gt", None)
                    mask_rle_gt = gt_data.get("mask_gt", None)
                    bbox_3d_gt = gt_data.get("bbox_3d_gt", None)

                    annotated_obs = annotate_image(
                        image=obs,
                        task_type=task_type,
                        target_category=target_category,
                        task_prompt=task_prompt,
                        pred=pred,
                        conf_score=info.get("conf_score") or 0.0,
                        step_id=step_id,
                        image_size=(image_width, image_height),
                        hfov=90.0,
                        bbox_2d=bbox_2d_gt,
                        mask_rle=mask_rle_gt,
                        bbox_3d=bbox_3d_gt,
                        draw_text=annotation_options.get("draw_text", True),
                        draw_bbox=annotation_options.get("draw_bbox", True),
                        draw_mask=annotation_options.get("draw_mask", True),
                        draw_3dbox=annotation_options.get("draw_3dbox", True),
                        draw_all_annotations=annotation_options.get("draw_all_annotations", True),
                        bbox_color=annotation_options.get("bbox_color", (0, 255, 0)),
                        mask_color=annotation_options.get("mask_color", (255, 0, 255)),
                        box3d_color=annotation_options.get("box3d_color", (0, 0, 255)),
                    )

                    if save_annotated_only:
                        annotated_obs.save(os.path.join(image_folder, image_name))
                    else:
                        annotated_obs.save(os.path.join(annotated_folder, image_name))

                step_id += 1
                collected_steps += 1

                # Execute action
                action_index = action_space.index(action)
                obs, reward, done, info = env.step(task_type, task_prompt, action_index, True)

                if done:
                    break

                prompt = build_text_obs([info])[0]

        except Exception as e:
            import traceback
            print(f"ERROR in scene {scene_name}, instance {j}: {e}")
            traceback.print_exc()
            continue

    # 关闭环境
    env.close()

    return {
        "scene_name": scene_name,
        "scene_idx": scene_idx,
        "status": "success",
        "dataset_entries": dataset_entries,
        "collected_steps": collected_steps,
        "final_step_id": step_id
    }


def collect_sft_data_single(
    dataset_name: str = "HM3D",
    output_name: str = "unified_tasks_hm3d_CoT_single",
    save_path: str = "/data/tct/habitat/",
    scene_idx: int = 0,
    max_scene_instance: int = 300,
    max_step_length: int = 10,
    seed: int = 0,
    enable_annotation: bool = True,
    save_annotated_only: bool = False
):
    """
    收集单个场景的 SFT 数据（无 Ray 版本）。

    Args:
        dataset_name: 数据集名称
        output_name: 输出名称
        save_path: 保存路径
        scene_idx: 要处理的场景索引
        max_scene_instance: 每个场景的最大实例数
        max_step_length: 每个实例的最大步数
        seed: 随机种子
        enable_annotation: 是否启用图像标注
        save_annotated_only: 是否只保存标注图像
    """
    # Annotation options
    annotation_options = {
        "draw_text": True,
        "draw_bbox": True,
        "draw_mask": True,
        "draw_3dbox": True,
        "draw_all_annotations": True,
        "bbox_color": (0, 255, 0),
        "mask_color": (255, 0, 255),
        "box3d_color": (0, 0, 255),
    }

    # Setup paths
    image_folder = f"{save_path}/sft_data/{output_name}"
    json_file = f"{save_path}/sft_data/{output_name}.json"
    annotated_folder = f"{save_path}/sft_data/{output_name}_annotated"

    os.makedirs(image_folder, exist_ok=True)
    if enable_annotation and not save_annotated_only:
        os.makedirs(annotated_folder, exist_ok=True)

    # Get scene list
    if dataset_name == "HM3D":
        scene_subfolder = HM3D_TRAINING_SCENES
    else:
        raise ValueError("Only HM3D is supported for now")

    # 确保 scene_idx 有效
    if scene_idx >= len(scene_subfolder):
        raise ValueError(f"scene_idx {scene_idx} is out of range. Max: {len(scene_subfolder) - 1}")

    scenes_size = len(scene_subfolder)

    print(f"{'='*60}")
    print("SFT DATA COLLECTION (Single Scene, No Ray)")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Scene index: {scene_idx}")
    print(f"Scene name: {scene_subfolder[scene_idx]}")
    print(f"Output: {json_file}")
    print(f"{'='*60}")
    print()

    # 收集单个场景的数据
    result = collect_single_scene(
        dataset_name=dataset_name,
        scene_idx=scene_idx,
        scene_subfolder=scene_subfolder,
        image_folder=image_folder,
        annotated_folder=annotated_folder,
        enable_annotation=enable_annotation,
        save_annotated_only=save_annotated_only,
        annotation_options=annotation_options,
        max_scene_instance=max_scene_instance,
        max_step_length=max_step_length,
        seed=seed,
        scenes_size=scenes_size
    )

    # 获取数据集条目
    final_dataset = result["dataset_entries"]

    # Print summary
    print()
    print(f"{'='*60}")
    print("SFT DATA COLLECTION COMPLETE")
    print(f"{'='*60}")

    if result["status"] == "success":
        print(f"  {result['scene_name']}: {result['collected_steps']} samples ✓")
    else:
        print(f"  {result['scene_name']}: ERROR")

    print(f"\nTotal samples: {len(final_dataset)}")

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"\nJSON saved to: {json_file}")
    print(f"Images folder: {image_folder}")
    if enable_annotation and not save_annotated_only:
        print(f"Annotated folder: {annotated_folder}")

    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect SFT dataset for a single scene (No Ray)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HM3D",
        help="Dataset name (HM3D)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="unified_tasks_hm3d_CoT_single",
        help="Output name for the SFT dataset"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="/data/tct/habitat/",
        help="Base path for saving data"
    )
    parser.add_argument(
        "--scene-idx",
        type=int,
        default=0,
        help="Scene index to process (default: 0)"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=300,
        help="Number of instances per scene"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Maximum steps per instance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--no-annotation",
        action="store_true",
        help="Disable image annotation"
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Save only annotated images (replace original)"
    )

    args = parser.parse_args()

    collect_sft_data_single(
        dataset_name=args.dataset,
        output_name=args.name,
        save_path=args.save_path,
        scene_idx=args.scene_idx,
        max_scene_instance=args.instances,
        max_step_length=args.steps,
        seed=args.seed,
        enable_annotation=not args.no_annotation,
        save_annotated_only=args.annotated_only
    )

'''
用法:
python test-single.py [options]

Options:
  --dataset HM3D         数据集名称
  --name NAME            输出名称
  --save-path PATH       保存路径
  --scene-idx N          要处理的场景索引 (default: 0)
  --instances N          每个场景的实例数 (default: 300)
  --steps N              每个实例的最大步数 (default: 10)
  --seed N               随机种子
  --no-annotation        禁用图像标注
  --annotated-only       只保存标注图像

示例:
  # 处理第一个场景
  python test-single.py --scene-idx 0

  # 处理第5个场景，每个场景100个实例
  python test-single.py --scene-idx 5 --instances 100

  # 处理场景并禁用标注
  python test-single.py --scene-idx 0 --no-annotation
'''
