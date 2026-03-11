import io
import base64
import requests
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from .constants import GROUNDINGDINO_ORIGINAL_SIZE, GROUNDINGDINO_NEW_SIZE

# BASE_URL_CAPTION = "http://10.105.100.175:8001"
# BASE_URL_GROUNDING = "http://10.105.100.175:8000"

BASE_URL_CAPTION = "http://127.0.0.1:8002"
BASE_URL_GROUNDING = "http://127.0.0.1:8000"
BASE_URL_DETECT = "http://127.0.0.1:8003"
BASE_URL_SEGMENT = "http://127.0.0.1:8004"
# BASE_URL_CAPTION = "http://10.105.100.230:8002"
# BASE_URL_GROUNDING = "http://10.105.100.230:8000"
# BASE_URL_DETECT = "http://10.105.100.230:8003"
# BASE_URL_SEGMENT = "http://10.105.100.230:8004"

def create_empty_mask(height: int, width: int):
    """
    创建一个空的RLE格式mask
    
    Args:
        height: 图像高度
        width: 图像宽度
        
    Returns:
        RLE格式的空mask字典
    """
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    rle = mask_utils.encode(np.asfortranarray(empty_mask))
    # 将bytes转换为str以便序列化
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    计算两个bbox的IoU (Intersection over Union)
    
    Args:
        box1: (xmin, ymin, xmax, ymax)
        box2: (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU值 (0-1之间)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    # 如果没有交集
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    # 计算交集面积
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # 计算并集面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def call_detect_from_pil(img: Image.Image, prompt: Tuple[int, int, int, int],
                         iou_threshold: float = 0.4):
    """
    调用DETR检测服务，并找出与prompt bbox最匹配的检测结果
    
    Args:
        img: PIL图像对象
        prompt: 提示bbox坐标 (xmin, ymin, xmax, ymax)
        iou_threshold: IoU阈值，低于此值返回默认值
        
    Returns:
        dict: 包含category、score、bbox的字典
              如果没有匹配的结果，返回默认值
    """
    # 1. 将 PIL.Image 对象保存到内存字节流
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. 构造 multipart/form-data 上传
    files = {"img_file": ("image.png", buf, "image/png")}

    # 3. 发起请求
    try:
        resp = requests.post(f"{BASE_URL_DETECT}/detect", files=files)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"Error calling DETR service: {e}")
        return {
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }
    
    detections = result.get("detections", [])
    
    # 4. 如果没有检测结果，返回默认值
    if not detections:
        return {
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }
    
    # 5. 计算每个检测结果与prompt的IoU，找出最佳匹配
    best_match = None
    best_iou = 0.0
    
    for detection in detections:
        bbox = detection["bbox"]  # [xmin, ymin, xmax, ymax]
        iou = calculate_iou(prompt, tuple(bbox))
        
        if iou > best_iou:
            best_iou = iou
            best_match = detection
    
    # 6. 如果最佳匹配的IoU低于阈值，返回默认值
    if best_iou < iou_threshold:
        return {
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }
    
    # 7. 返回最佳匹配结果
    return {
        "category": best_match["category"],
        "score": best_match["score"],
        "bbox": tuple(best_match["bbox"])
    }

def call_class_detect_from_pil(img: Image.Image, prompt: str):
    """
    调用DETR检测服务，返回所有类别匹配prompt的检测结果
    
    Args:
        img: PIL图像对象
        prompt: 目标类别名称
        
    Returns:
        list: 包含所有匹配类别的检测结果列表
              每个元素是dict，包含category、score、bbox
              如果没有匹配的结果，返回包含默认值的列表
    """
    # 1. 将 PIL.Image 对象保存到内存字节流
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. 构造 multipart/form-data 上传
    files = {"img_file": ("image.png", buf, "image/png")}

    # 3. 发起请求
    try:
        resp = requests.post(f"{BASE_URL_DETECT}/detect", files=files)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"Error calling DETR service: {e}")
        return [{
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }]
    
    detections = result.get("detections", [])
    
    # 4. 如果没有检测结果，返回默认值
    if not detections:
        return [{
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }]
    
    # 5. 过滤出所有类别匹配的检测结果
    matched_detections = []
    for detection in detections:
        if detection["category"] == prompt:
            matched_detections.append({
                "category": detection["category"],
                "score": detection["score"],
                "bbox": tuple(detection["bbox"])
            })
    
    # 6. 如果没有匹配的结果，返回默认值
    if not matched_detections:
        return [{
            "category": "unknown",
            "score": 0.0,
            "bbox": (0, 0, 0, 0)
        }]
    
    # 7. 按照置信度从高到低排序
    matched_detections.sort(key=lambda x: x["score"], reverse=True)
    
    return matched_detections



def call_segment_from_pil(img: Image.Image, prompt: Tuple[float, float, float, float]):
    """
    使用SAM对图像进行分割
    
    Args:
        img: PIL图像对象
        prompt: 边界框坐标元组，格式为 (xmin, ymin, xmax, ymax)
        
    Returns:
        分割结果字典，包含：
        - mask: RLE格式的分割mask（失败时返回空掩码而非None）
        - score: 分割置信度（predicted_iou）
        - bbox: 分割后的边界框
    """
    try:
        # 解析prompt中的边界框坐标
        xmin, ymin, xmax, ymax = prompt
        
        # 验证边界框
        img_width, img_height = img.size
        if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
            raise ValueError(f"Bounding box out of image bounds. Image size: {img_width}x{img_height}, Box: {prompt}")
        
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid bounding box coordinates: {prompt}")
        
        # 将PIL图像转换为字节流
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # 调用SAM服务
        files = {'img_file': ('image.png', img_bytes, 'image/png')}
        data = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'multimask_output': False  # 只返回一个mask
        }
        
        response = requests.post(f"{BASE_URL_SEGMENT}/segment", files=files, data=data, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"SAM service error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if not result.get('segmentations'):
            img_width, img_height = img.size
            return {
                "mask": create_empty_mask(img_height, img_width),
                "score": 0.0,
                "bbox": [xmin, ymin, xmax, ymax]
            }
        
        # 返回第一个分割结果
        segmentation = result['segmentations'][0]
        
        return {
            "mask": segmentation['mask'],  # RLE格式的mask
            "score": segmentation['score'],  # predicted_iou
            "bbox": segmentation['bbox']
        }
        
    except Exception as e:
        print(f"SAM segmentation error: {e}")
        img_width, img_height = img.size
        return {
            "mask": create_empty_mask(img_height, img_width),
            "score": 0.0,
            "bbox": [0, 0, 0, 0]
        }

def call_grounding_from_pil(img: Image.Image, prompt: str,
                            box_threshold: float = 0.25,
                            text_threshold: float = 0.25):
    # 1. 将 PIL.Image 对象保存到内存字节流
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. 构造 multipart/form-data 上传
    files = {"img_file": ("image.png", buf, "image/png")}
    data = {
        "prompt": prompt,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
    }

    # 3. 发起请求
    resp = requests.post(f"{BASE_URL_GROUNDING}/ground", files=files, data=data)
    resp.raise_for_status()
    return resp.json()

# raw version, 用于计算mAP: 获取原始的grounding结果，不用置信度过滤
def call_grounding_from_pil_raw(img: Image.Image, prompt: str,
                            box_threshold: float = 0.25,
                            text_threshold: float = 0.25):
    # 1. 将 PIL.Image 对象保存到内存字节流
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. 构造 multipart/form-data 上传
    files = {"img_file": ("image.png", buf, "image/png")}
    data = {
        "prompt": prompt,
        "box_threshold": str(box_threshold),
        "text_threshold": str(text_threshold),
    }

    # 3. 发起请求
    resp = requests.post(f"{BASE_URL_GROUNDING}/ground_raw", files=files, data=data)
    resp.raise_for_status()
    return resp.json()

def call_grounding_segment_pipeline(img: Image.Image, prompt: str,
                                   box_threshold: float = 0.25,
                                   text_threshold: float = 0.25):
    """
    先使用GroundingDINO定位目标，再使用SAM进行分割的pipeline
    
    Args:
        img: PIL图像对象
        prompt: 目标描述文本
        box_threshold: grounding的box阈值
        text_threshold: grounding的文本阈值
        
    Returns:
        dict: 包含检测和分割结果的字典：
              - grounding_bbox: grounding检测的边界框 [xmin, ymin, xmax, ymax]
              - grounding_score: grounding检测的置信度
              - segment_score: 分割的置信度（predicted_iou）
              - segment_mask: RLE格式的分割mask（失败时返回空掩码，可直接解码）
              如果没有检测结果或分割失败，返回默认值
    """
    # 获取图像尺寸用于创建空掩码
    img_width, img_height = img.size
    
    default_result = {
        "grounding_bbox": [0, 0, 0, 0],
        "grounding_score": 0.0,
        "segment_score": 0.0,
        "segment_mask": create_empty_mask(img_height, img_width)
    }
    
    try:
        # 1. 调用grounding获取目标边界框
        grounding_result = call_grounding_from_pil(
            img, prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold
        )
        
        # 2. 检查是否有检测结果
        bbox = grounding_result.get("bbox", [])
        score = grounding_result.get("score", [])
        
        if not bbox or len(bbox) == 0:
            print(f"No grounding results found for prompt: {prompt}")
            return default_result

        # 3. reshape bbox
        bbox = reshape_bbox_xxyy(bbox)

        # 3. 调用SAM分割
        bbox_tuple = (bbox[0], bbox[2], bbox[1], bbox[3]) #xxyy->xyxy
        # bbox_tuple = (bbox[0], bbox[1], bbox[2], bbox[3]) #xyxy
        segment_result = call_segment_from_pil(img, bbox_tuple)
        
        # 4. 组合结果
        result = {
            "grounding_bbox": bbox_tuple,
            "grounding_score": float(score),
            "segment_score": segment_result.get("score", 0.0),
            "segment_mask": segment_result.get("mask")
        }
        
        return result
        
    except Exception as e:
        print(f"Grounding-Segment pipeline error: {e}")
        return default_result

# def reshape_bbox_xxyy(bbox, original_size=(1000, 800), new_size=(800, 640)):
def reshape_bbox_xxyy(bbox, reverse_order: bool = False, original_size=GROUNDINGDINO_ORIGINAL_SIZE, new_size=GROUNDINGDINO_NEW_SIZE):
    """
    (针对GroundingDINO的输出) 根据图像尺寸变化对bbox进行转换。

    参数:
        bbox (tuple): 原始bbox，格式为 (x1, y1, x2, y2)。
        original_size (tuple): 原始图像尺寸 (宽, 高)。
        new_size (tuple): 新图像尺寸 (宽, 高)。

    返回:
        tuple: 转换后的bbox，格式为 (x, x, y, y) 或 (x, y, x, y)。
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

    # 限制坐标范围，防止超出图像边界
    x1_new = max(0, min(x1_new, new_size[0]))
    x2_new = max(0, min(x2_new, new_size[0]))
    y1_new = max(0, min(y1_new, new_size[1]))
    y2_new = max(0, min(y2_new, new_size[1]))

    if reverse_order:
        return (x1_new, y1_new, x2_new, y2_new)
    else:
        return (x1_new, x2_new, y1_new, y2_new)
    
def draw_box_on_image(img: Image.Image, x1: float, y1: float, x2: float, y2: float, color: str = "blue"):
    """在图像上绘制矩形框。

    Args:
        img: PIL.Image 对象。
        x1: 左上角 x 坐标。
        y1: 左上角 y 坐标。
        x2: 右下角 x 坐标。
        y2: 右下角 y 坐标。
        color: 矩形框的颜色，默认为红色。
    """
    draw = ImageDraw.Draw(img)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)  # width 设置边框粗细
    return img

def call_caption_from_pil(img: Image.Image, x1: float, y1: float, x2: float, y2: float, category: str = None):
    """Qwen-VL"""
    # 1. 将 PIL.Image 对象保存到内存字节流
    buf = io.BytesIO()
    img = img.copy()  # 创建图像的副本，避免修改原图
    img = draw_box_on_image(img, x1, y1, x2, y2)  # (Qwen-VL) 在图像上绘制矩形框
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. 构造 multipart/form-data 上传
    files = {"img_file": ("image.png", buf, "image/png")}

    # 3. 发起请求
    if category is not None:
        data = {
            "category": category
        }
        resp = requests.post(f"{BASE_URL_CAPTION}/caption_w_category", files=files, data=data)
    else:
        resp = requests.post(f"{BASE_URL_CAPTION}/caption", files=files)
    resp.raise_for_status()
    return resp.json()


def call_generate_task_description(caption: str, task_type: str):
    """
    调用 Qwen-VL 服务生成自然语言形式的 task_description。
    
    Args:
        caption: 目标物体的描述（来自 /caption_w_category 接口）
        task_type: 任务类型（grounding, segment, 3d-box）
    
    Returns:
        dict: 包含 task_prompt 和 task_description 的字典
            - task_prompt: 原始 caption
            - task_description: 生成的自然语言任务描述
    """
    data = {
        "caption": caption,
        "task_type": task_type
    }
    resp = requests.post(f"{BASE_URL_CAPTION}/generate_task_description", json=data)
    resp.raise_for_status()
    return resp.json()


def call_caption_and_task_description(img: Image.Image, x1: float, y1: float, x2: float, y2: float, category: str, task_type: str):
    """
    一次性获取 caption 和 task_description。
    
    先调用 caption_w_category 获取 caption，再调用 generate_task_description 获取自然语言任务描述。
    
    Args:
        img: PIL Image 对象
        x1, y1, x2, y2: 边界框坐标
        category: 目标类别
        task_type: 任务类型（grounding, segment, 3d-box）
    
    Returns:
        dict: 包含 task_prompt 和 task_description 的字典
    """
    # Step 1: 获取 caption
    caption_response = call_caption_from_pil(img, x1, y1, x2, y2, category)
    caption = caption_response["caption"]
    
    # Step 2: 生成 task_description
    task_desc_response = call_generate_task_description(caption, task_type)
    
    return {
        "task_prompt": caption,
        "task_description": task_desc_response["task_description"]
    }

# def call_caption_from_pil(img: Image.Image, x1: float, y1: float, x2: float, y2: float):
#     """florence"""
#     # 1. 将 PIL.Image 对象保存到内存字节流
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)

#     # 2. 构造 multipart/form-data 上传
#     files = {"img_file": ("image.png", buf, "image/png")}
#     data = {
#         "x1": str(x1),
#         "y1": str(y1),
#         "x2": str(x2),
#         "y2": str(y2),
#     }

#     # 3. 发起请求
#     resp = requests.post(f"{BASE_URL_CAPTION}/caption", files=files, data=data)
#     resp.raise_for_status()
#     return resp.json()