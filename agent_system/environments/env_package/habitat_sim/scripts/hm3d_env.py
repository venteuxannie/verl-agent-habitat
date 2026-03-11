import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import os

from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import visualize_bbox_on_image
from habitat_sim.utils import viz_utils as vut

def decode_rle_mask(mask_rle, shape):
    """解码RLE格式的掩码"""
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    
    # 如果counts是字符串，需要转换回bytes
    if isinstance(mask_rle['counts'], str):
        mask_rle = mask_rle.copy()
        mask_rle['counts'] = mask_rle['counts'].encode('utf-8')
    
    # 解码RLE掩码
    mask = mask_utils.decode(mask_rle)
    return mask.astype(np.uint8)

def visualize_mask_on_image(rgb_image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    在RGB图像上可视化掩码
    
    Args:
        rgb_image: PIL Image 或 numpy array
        mask: numpy array, 二值掩码 (0或1)
        alpha: 掩码透明度
        color: 掩码颜色 (R, G, B)
    
    Returns:
        PIL Image
    """
    # 转换为numpy array
    if isinstance(rgb_image, Image.Image):
        rgb_np = np.array(rgb_image)
    else:
        rgb_np = rgb_image.copy()
    
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 创建彩色掩码
    mask_colored = np.zeros_like(rgb_np)
    mask_colored[mask_binary > 0] = color
    
    # 混合原图和掩码
    result = (rgb_np * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    
    return Image.fromarray(result)

def visualize_mask_on_blank(mask, image_size, color=(0, 255, 0)):
    """
    在空白图像上可视化掩码
    
    Args:
        mask: numpy array, 二值掩码 (0或1)
        image_size: (width, height) 图像尺寸
        color: 掩码颜色 (R, G, B)
    
    Returns:
        PIL Image
    """
    # 创建空白图像
    blank_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 在掩码区域填充颜色
    blank_image[mask_binary > 0] = color
    
    return Image.fromarray(blank_image)

def save_visualizations(rgb_image, original_mask, unoccluded_mask, bbox_3d=None, output_dir="visualizations", hfov=90.0):
    """保存所有可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像尺寸
    if isinstance(rgb_image, Image.Image):
        image_size = rgb_image.size  # (width, height)
        rgb_np = np.array(rgb_image)
    else:
        image_size = (rgb_image.shape[1], rgb_image.shape[0])
        rgb_np = rgb_image.copy()
    
    # 1. 原始语义掩码叠加到RGB图像
    vis_original_on_rgb = visualize_mask_on_image(rgb_image, original_mask, alpha=0.5, color=(255, 0, 0))
    vis_original_on_rgb.save(os.path.join(output_dir, "original_mask_on_rgb.png"))
    print(f"已保存: {output_dir}/original_mask_on_rgb.png")
    
    # 2. 无遮挡掩码叠加到RGB图像
    vis_unoccluded_on_rgb = visualize_mask_on_image(rgb_image, unoccluded_mask, alpha=0.5, color=(0, 255, 0))
    vis_unoccluded_on_rgb.save(os.path.join(output_dir, "unoccluded_mask_on_rgb.png"))
    print(f"已保存: {output_dir}/unoccluded_mask_on_rgb.png")
    
    # 3. 原始语义掩码叠加到空白图像
    vis_original_on_blank = visualize_mask_on_blank(original_mask, image_size, color=(255, 0, 0))
    vis_original_on_blank.save(os.path.join(output_dir, "original_mask_on_blank.png"))
    print(f"已保存: {output_dir}/original_mask_on_blank.png")
    
    # 4. 无遮挡掩码叠加到空白图像
    vis_unoccluded_on_blank = visualize_mask_on_blank(unoccluded_mask, image_size, color=(0, 255, 0))
    vis_unoccluded_on_blank.save(os.path.join(output_dir, "unoccluded_mask_on_blank.png"))
    print(f"已保存: {output_dir}/unoccluded_mask_on_blank.png")
    
    # 5. 3D box可视化（如果存在）
    vis_3dbox_on_rgb = None
    vis_3dbox_on_blank = None
    if bbox_3d is not None:
        # 3D box叠加到RGB图像
        rgb_np_copy = rgb_np.copy()
        vis_3dbox_on_rgb_np = visualize_bbox_on_image(
            rgb_np_copy, bbox_3d, hfov=hfov, color=(0, 0, 255), thickness=2, use_obb=True
        )
        vis_3dbox_on_rgb = Image.fromarray(vis_3dbox_on_rgb_np)
        vis_3dbox_on_rgb.save(os.path.join(output_dir, "3dbox_on_rgb.png"))
        print(f"已保存: {output_dir}/3dbox_on_rgb.png")
        
        # 3D box叠加到空白图像
        blank_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        vis_3dbox_on_blank_np = visualize_bbox_on_image(
            blank_image, bbox_3d, hfov=hfov, color=(0, 0, 255), thickness=2, use_obb=True
        )
        vis_3dbox_on_blank = Image.fromarray(vis_3dbox_on_blank_np)
        vis_3dbox_on_blank.save(os.path.join(output_dir, "3dbox_on_blank.png"))
        print(f"已保存: {output_dir}/3dbox_on_blank.png")
    
    # 6. 创建对比图
    if bbox_3d is not None:
        # 如果有3D box，创建3x3的布局
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        axes[0, 0].imshow(vis_original_on_rgb)
        axes[0, 0].set_title("Original Semantic Mask (Red) Overlaid on RGB Image", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(vis_unoccluded_on_rgb)
        axes[0, 1].set_title("Unoccluded Mask (Green) Overlaid on RGB Image", fontsize=12)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(vis_original_on_blank)
        axes[1, 0].set_title("Original Semantic Mask (Red) on Blank Image", fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(vis_unoccluded_on_blank)
        axes[1, 1].set_title("Unoccluded Mask (Green) on Blank Image", fontsize=12)
        axes[1, 1].axis('off')
        
        axes[2, 0].imshow(vis_3dbox_on_rgb)
        axes[2, 0].set_title("3D Bounding Box (Blue) Overlaid on RGB Image", fontsize=12)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(vis_3dbox_on_blank)
        axes[2, 1].set_title("3D Bounding Box (Blue) on Blank Image", fontsize=12)
        axes[2, 1].axis('off')
    else:
        # 如果没有3D box，创建2x2的布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(vis_original_on_rgb)
        axes[0, 0].set_title("Original Semantic Mask (Red) Overlaid on RGB Image", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(vis_unoccluded_on_rgb)
        axes[0, 1].set_title("Unoccluded Mask (Green) Overlaid on RGB Image", fontsize=12)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(vis_original_on_blank)
        axes[1, 0].set_title("Original Semantic Mask (Red) on Blank Image", fontsize=12)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(vis_unoccluded_on_blank)
        axes[1, 1].set_title("Unoccluded Mask (Green) on Blank Image", fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mask_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"已保存: {output_dir}/mask_comparison.png")
    plt.close()
    
    # 7. 保存原始RGB图像作为参考
    if isinstance(rgb_image, Image.Image):
        rgb_image.save(os.path.join(output_dir, "original_rgb.png"))
    else:
        Image.fromarray(rgb_image).save(os.path.join(output_dir, "original_rgb.png"))
    print(f"已保存: {output_dir}/original_rgb.png")

# ============== 配置选项 ==============
EXECUTE_ACTION = True  # 设置为 True 执行 look_down 动作，False 则不执行
ACTION_TO_EXECUTE = "look_down"  # 要执行的动作：look_down, look_up, move_forward, turn_left, turn_right
# ======================================

# 初始化环境
print("\n初始化环境...")
dataset_name = "HM3D"
seed = 42
scenes_size = 10
max_scene_instance = 100
max_step_length = 10

env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)

print("\n加载任务到环境...")
obs_pil, info = env.reset(task_type="segment")

print(f"\n任务信息:")
print(f"  任务类型: {info['task_type']}")
print(f"  目标类别: {info['target_category']}")
print(f"  语义ID: {info['semantic_id']}")
print(f"  实例ID: {info['instance_id']}")

# ============== 可选：执行动作 ==============
if EXECUTE_ACTION:
    print(f"\n执行 {ACTION_TO_EXECUTE} 动作...")
    obs = env.sim.step(ACTION_TO_EXECUTE)
    print(f"  {ACTION_TO_EXECUTE} 动作执行完成")
    
    # 打印 agent 和 sensor 状态
    agent_state = env.sim.get_agent(0).get_state()
    print(f"\n执行 {ACTION_TO_EXECUTE} 后的状态:")
    print(f"  Agent position: {agent_state.position}")
    print(f"  Agent rotation: {agent_state.rotation}")
    if hasattr(agent_state, 'sensor_states') and agent_state.sensor_states:
        for sensor_name, sensor_pose in agent_state.sensor_states.items():
            print(f"  Sensor '{sensor_name}' position: {sensor_pose.position}")
            print(f"  Sensor '{sensor_name}' rotation: {sensor_pose.rotation}")
else:
    print("\n不执行任何动作，使用初始视角...")
    obs = env.sim.get_sensor_observations()
# ================================================

# 获取当前观测数据
rgb_image = vut.observation_to_image(obs["rgb"], "color").convert("RGB")
semantic_obs = obs["semantic"]

# 获取原始语义掩码（从当前场景的语义分割）
print("\n提取掩码...")
semantic_np = np.array(semantic_obs)
original_mask = (semantic_np == info['semantic_id']).astype(np.uint8)

# ============== 重新获取 GT 信息 ==============
action_desc = f"（{ACTION_TO_EXECUTE} 后）" if EXECUTE_ACTION else "（初始视角）"
print(f"\n重新获取 GT 信息{action_desc}...")
_, gt_dict = env.get_gt()
print("  GT 信息获取完成")
# ============================================================

# 获取无遮挡掩码（从新获取的gt）
if gt_dict and gt_dict.get('mask_gt') is not None:
    mask_rle = gt_dict['mask_gt']
    image_size = rgb_image.size  # (width, height)
    unoccluded_mask = decode_rle_mask(mask_rle, (image_size[1], image_size[0]))  # RLE需要(height, width)
    print("  成功解码无遮挡掩码")
else:
    print("  警告: 未找到无遮挡掩码，使用空掩码")
    image_size = rgb_image.size
    unoccluded_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

# 获取3D box信息（从新获取的gt）
bbox_3d = None
if gt_dict and gt_dict.get('bbox_3d_gt') is not None:
    bbox_3d = gt_dict['bbox_3d_gt']
    print("  成功获取3D box信息")
else:
    print("  警告: 未找到3D box信息")

# 打印掩码统计信息
print(f"\n掩码统计:")
print(f"  原始语义掩码像素数: {np.sum(original_mask)}")
print(f"  无遮挡掩码像素数: {np.sum(unoccluded_mask)}")
print(f"  图像尺寸: {image_size}")

# 打印3D box信息
if bbox_3d is not None:
    print(f"\n3D Box信息:")
    if 'center' in bbox_3d:
        print(f"  中心点: {bbox_3d['center']}")
    if 'size' in bbox_3d:
        print(f"  尺寸: {bbox_3d['size']}")
    if 'obb_center' in bbox_3d:
        print(f"  OBB中心: {bbox_3d['obb_center']}")
    if 'obb_extents' in bbox_3d:
        print(f"  OBB半长度: {bbox_3d['obb_extents']}")

# 可视化并保存
action_info = f"（{ACTION_TO_EXECUTE} 后的视角）" if EXECUTE_ACTION else "（初始视角）"
print(f"\n生成可视化{action_info}...")
save_visualizations(rgb_image, original_mask, unoccluded_mask, bbox_3d=bbox_3d, output_dir="mask_visualizations", hfov=90.0)

print("\n可视化完成！所有结果已保存到 mask_visualizations/ 目录")
if EXECUTE_ACTION:
    print(f"（注意：这是执行 {ACTION_TO_EXECUTE} 动作后的视角）")
else:
    print("（注意：这是初始视角，未执行任何动作）")