from PIL import Image
import requests, base64, io
import numpy as np
from pycocotools import mask as mask_utils
from habitat_sim.utils import viz_utils as vut
from .third_party import call_grounding_from_pil, call_detect_from_pil, call_class_detect_from_pil, call_grounding_segment_pipeline
from .habitat_3dbox_utils import predict_3d_bbox_from_mask
from .constants import SENSOR_RESOLUTION

def action_format_reward(is_valid_action):
    SCORE = 0.05 # 0.8
    wrong_score = -SCORE
    right_score = SCORE

    if is_valid_action == 1:
        return right_score
    else:
        return wrong_score

# 辅助函数：计算bbox面积占比
def _bbox_area_ratio(bbox, image_size):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    W, H = image_size
    img_area = max(1.0, float(W) * float(H))
    return max(0.0, min(1.0, (w * h) / img_area))

# 辅助函数：计算bbox中心与图像中心的对齐度（越靠近中心越接近1）
def _bbox_center_alignment(bbox, image_size):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    W, H = image_size
    # 归一化到[0,1]的距离，再取1-距离作为对齐度
    dx = abs(cx - (W / 2.0)) / max(1.0, (W / 2.0))
    dy = abs(cy - (H / 2.0)) / max(1.0, (H / 2.0))
    dist01 = min(1.0, (dx + dy) * 0.5)
    return 1.0 - dist01

# 辅助函数：从RLE格式mask计算面积占比
def _mask_area_ratio(mask_rle, image_size):
    """
    计算mask的面积占比
    
    Args:
        mask_rle: RLE格式的mask字典
        image_size: 图像尺寸 (W, H)
        
    Returns:
        面积占比 [0, 1]
    """
    if mask_rle is None:
        return 0.0
    try:
        # 解码RLE格式的mask
        mask_binary = mask_utils.decode(mask_rle)
        mask_area = np.sum(mask_binary)
        W, H = image_size
        img_area = max(1.0, float(W) * float(H))
        return max(0.0, min(1.0, mask_area / img_area))
    except Exception as e:
        print(f"Error calculating mask area ratio: {e}")
        return 0.0

# 辅助函数：从RLE格式mask计算中心对齐度
def _mask_center_alignment(mask_rle, image_size):
    """
    计算mask中心与图像中心的对齐度（越靠近中心越接近1）
    
    Args:
        mask_rle: RLE格式的mask字典
        image_size: 图像尺寸 (W, H)
        
    Returns:
        中心对齐度 [0, 1]
    """
    if mask_rle is None:
        return 0.0
    try:
        # 解码RLE格式的mask
        mask_binary = mask_utils.decode(mask_rle)
        
        # 计算mask的质心
        y_coords, x_coords = np.where(mask_binary > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return 0.0
        
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
        
        W, H = image_size
        # 归一化到[0,1]的距离，再取1-距离作为对齐度
        dx = abs(cx - (W / 2.0)) / max(1.0, (W / 2.0))
        dy = abs(cy - (H / 2.0)) / max(1.0, (H / 2.0))
        dist01 = min(1.0, (dx + dy) * 0.5)
        return 1.0 - dist01
    except Exception as e:
        print(f"Error calculating mask center alignment: {e}")
        return 0.0

# 辅助函数：从3D bbox投影计算面积占比
def _bbox3d_area_ratio(bbox_3d, image_size, hfov=90.0):
    """
    将3D bbox投影到2D平面，计算投影面积占比
    
    Args:
        bbox_3d: 3D bbox字典（相机坐标系）
        image_size: 图像尺寸 (W, H)
        hfov: 水平视场角（degrees）
        
    Returns:
        面积占比 [0, 1]
    """
    if bbox_3d is None:
        return 0.0
    
    try:
        W, H = image_size
        
        # 计算相机内参
        focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cx = W / 2.0
        cy = H / 2.0
        
        # 获取8个角点（相机坐标系）
        corners = bbox_3d.get('corners')
        if corners is None:
            return 0.0
        
        # 投影到2D
        projected_points = []
        for corner in corners:
            x, y, z = corner
            if z >= 0:  # 在相机后面，跳过
                continue
            # 投影公式
            u = x * focal_length / (-z) + cx
            v = -y * focal_length / (-z) + cy
            if 0 <= u < W and 0 <= v < H:
                projected_points.append((u, v))
        
        if len(projected_points) < 3:
            return 0.0
        
        # 计算投影点的凸包面积
        from scipy.spatial import ConvexHull
        points_array = np.array(projected_points)
        try:
            hull = ConvexHull(points_array)
            projected_area = hull.volume  # 2D中volume就是面积
        except:
            # 如果点共线，使用外接矩形面积
            min_u, min_v = np.min(points_array, axis=0)
            max_u, max_v = np.max(points_array, axis=0)
            projected_area = (max_u - min_u) * (max_v - min_v)
        
        img_area = float(W * H)
        area_ratio = max(0.0, min(1.0, projected_area / img_area))
        
        return area_ratio
        
    except Exception as e:
        print(f"Error calculating bbox3d area ratio: {e}")
        return 0.0

# 辅助函数：从3D bbox投影计算中心对齐度
def _bbox3d_center_alignment(bbox_3d, image_size, hfov=90.0):
    """
    将3D bbox中心投影到2D平面，计算与图像中心的对齐度
    
    Args:
        bbox_3d: 3D bbox字典（相机坐标系）
        image_size: 图像尺寸 (W, H)
        hfov: 水平视场角（degrees）
        
    Returns:
        中心对齐度 [0, 1]
    """
    if bbox_3d is None:
        return 0.0
    
    try:
        W, H = image_size
        
        # 计算相机内参
        focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cx = W / 2.0
        cy = H / 2.0
        
        # 获取3D bbox中心
        center_3d = bbox_3d.get('center')
        if center_3d is None:
            return 0.0
        
        x, y, z = center_3d
        
        # 如果中心在相机后面，返回0
        if z >= 0:
            return 0.0
        
        # 投影到2D
        u = x * focal_length / (-z) + cx
        v = -y * focal_length / (-z) + cy
        
        # 如果投影点在图像外，返回较低的对齐度
        if u < 0 or u >= W or v < 0 or v >= H:
            # 计算到图像边界的距离
            u_clamped = max(0, min(W - 1, u))
            v_clamped = max(0, min(H - 1, v))
            dx = abs(u_clamped - (W / 2.0)) / max(1.0, (W / 2.0))
            dy = abs(v_clamped - (H / 2.0)) / max(1.0, (H / 2.0))
        else:
            # 计算归一化距离
            dx = abs(u - (W / 2.0)) / max(1.0, (W / 2.0))
            dy = abs(v - (H / 2.0)) / max(1.0, (H / 2.0))
        
        dist01 = min(1.0, (dx + dy) * 0.5)
        return 1.0 - dist01
        
    except Exception as e:
        print(f"Error calculating bbox3d center alignment: {e}")
        return 0.0

def reward_function(is_correct_task_type, pred_task_type, obs, prompt, pre_phi, is_valid_action, action, step_counter, agent_state, alpha_conf=0.5):
    # 超参数
    clip_delta = 0.5   # 形状化差分裁剪

    # 语法有效性奖励（作为小幅辅助项）
    format_score = action_format_reward(is_valid_action)

    # 视觉评估
    # "grounding text" for "grounding"; "bbox_gt_visible" for "segment" & "detect" ("object category" for "class detect")
    obs_rgb = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
    img_vg = None
    if pred_task_type == "grounding":
        response = call_grounding_from_pil(obs_rgb, prompt)

        pred = bbox = response["bbox"]
        conf_raw = response["score"]
        conf_score = max(0.0, min(1.0, conf_raw))

        img_bytes = base64.b64decode(response["image_base64"]) if "image_base64" in response else None
        if img_bytes is not None:
            img_vg = Image.open(io.BytesIO(img_bytes))
            W, H = img_vg.size
        else:
            # 回退：若没有返回图像，构造占位
            img_vg = Image.new("RGB", SENSOR_RESOLUTION)
            W, H = img_vg.size
        # 稳定潜在函数：conf + 面积占比 + 中心对齐
        area_ratio = _bbox_area_ratio(bbox, (W, H))
        center_align = _bbox_center_alignment(bbox, (W, H))

        # phi = alpha_conf * conf_score + beta_area * area_ratio + gamma_center * center_align

    elif pred_task_type == "detect":
        prompt = (prompt[0], prompt[2], prompt[1], prompt[3]) # xxyy->xyxy
        response = call_detect_from_pil(obs_rgb, prompt)

        pred = response["category"]
        bbox = response["bbox"]
        conf_raw = response["score"]
        conf_score = max(0.0, min(1.0, conf_raw))

        area_ratio = _bbox_area_ratio(bbox, SENSOR_RESOLUTION)
        center_align = _bbox_center_alignment(bbox, SENSOR_RESOLUTION)

        # phi = alpha_conf * conf_score + beta_area * area_ratio + gamma_center * center_align

    elif pred_task_type == "class-detect":
        response = call_class_detect_from_pil(obs_rgb, prompt)

        pred = response
        # 计算所有检测结果的平均置信度、面积占比和中心对齐度
        if len(response) > 0 and response[0]["category"] != "unknown":
            conf_raw = sum(item["score"] for item in response) / len(response)
            # 计算所有bbox的平均面积占比和中心对齐度
            area_ratios = [_bbox_area_ratio(item["bbox"], SENSOR_RESOLUTION) for item in response]
            center_aligns = [_bbox_center_alignment(item["bbox"], SENSOR_RESOLUTION) for item in response]
            area_ratio = sum(area_ratios) / len(area_ratios)
            center_align = sum(center_aligns) / len(center_aligns)
        else:
            conf_raw = 0.0
            area_ratio = 0.0
            center_align = 0.0
        conf_score = max(0.0, min(1.0, conf_raw))

        # phi = alpha_conf * conf_score + beta_area * area_ratio + gamma_center * center_align

    elif pred_task_type == "segment":
        response = call_grounding_segment_pipeline(obs_rgb, prompt)

        # pred = response["segment_mask"]
        pred = response
        
        # conf_score 来自 grounding_score 和 segment_score 的平均
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        conf_raw = (grounding_score + segment_score) / 2.0
        conf_score = max(0.0, min(1.0, conf_raw))
        
        # 使用 segment_mask 计算 area_ratio 和 center_align
        segment_mask = response.get("segment_mask")
        area_ratio = _mask_area_ratio(segment_mask, SENSOR_RESOLUTION)
        center_align = _mask_center_alignment(segment_mask, SENSOR_RESOLUTION)
        
        # 计算综合 phi
        # phi = alpha_conf * conf_score + beta_area * area_ratio + gamma_center * center_align
        
    elif pred_task_type == "3d-box":
        response = call_grounding_segment_pipeline(obs_rgb, prompt)
        segment_mask = response.get("segment_mask")
        bbox_3d = predict_3d_bbox_from_mask(
            mask_rle=segment_mask,
            depth_obs=obs["depth"],
            agent_state=agent_state,
            hfov=90.0,
            denoise=True,
            align_to_ground=True
        )
        pred = bbox_3d
        
        # conf_score 来自 grounding_score 和 segment_score 的平均
        grounding_score = response.get("grounding_score", 0.0)
        segment_score = response.get("segment_score", 0.0)
        geometric_confidence = bbox_3d.get("geometric_confidence", 0.0)
        
        conf_raw = (grounding_score + segment_score + geometric_confidence) / 3.0
        conf_score = max(0.0, min(1.0, conf_raw))
        
        # 使用 bbox_3d 计算 area_ratio 和 center_align（基于3D bbox投影）
        area_ratio = _bbox3d_area_ratio(bbox_3d, SENSOR_RESOLUTION, hfov=90.0)
        center_align = _bbox3d_center_alignment(bbox_3d, SENSOR_RESOLUTION, hfov=90.0)
        
        # 计算综合 phi
        # phi = alpha_conf * conf_score + beta_area * area_ratio + gamma_center * center_align
    else:
        raise ValueError(f"Unsupported task type: {pred_task_type}")

    phi = alpha_conf * conf_score + (1 - alpha_conf) * (0.75 * area_ratio + 0.25 * center_align)
    
    # 这里将 pre_conf_score 视为上一时刻的 phi（保持接口不变）
    pre_phi = float(pre_phi)
    delta_phi = max(-clip_delta, min(clip_delta, phi - pre_phi))

    # 如果任务类型预测不正确，则惩罚极大
    if not is_correct_task_type:
        delta_phi = -1
    reward = format_score + delta_phi

    return reward, format_score, conf_raw, img_vg, None, pred, phi
