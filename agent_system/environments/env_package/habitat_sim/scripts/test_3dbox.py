"""
测试脚本：测试3D Bounding Box预测效果

使用方法:
    # 测试指定任务ID（默认14）
    python test_3dbox.py --task_id 14
    
    # 测试不同的任务ID
    python test_3dbox.py --task_id 24
    
    # 只测试高精度模式
    python test_3dbox.py --task_id 14 --configs advanced
    
    # 测试所有配置
    python test_3dbox.py --task_id 14 --configs all
    
    # 启用旋转搜索调试可视化
    python test_3dbox.py --task_id 14 --configs advanced --debug_vis
    
    # 自定义输出目录
    python test_3dbox.py --task_id 14 --output_dir /path/to/output

参数说明:
    --task_id: 目标任务ID (默认: 14)
    --json_file: 任务JSON文件路径
    --output_dir: 输出目录路径
    --debug_vis: 启用旋转搜索调试可视化
    --configs: 要测试的配置 (all/basic/standard/advanced)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import ray
import argparse

# 导入环境和3D box工具
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import (
    predict_3d_bbox,
    visualize_bbox_on_image,
    compute_bbox_iou_3d,
    depth_to_3d_points,
    transform_points_to_world_coords
)


def load_task_by_id(json_file, task_id):
    """从 JSON 文件中加载指定 task_id 的任务"""
    with open(json_file, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    
    for task in all_tasks:
        if task.get('task_id') == task_id:
            return task
    
    raise ValueError(f"Task with task_id={task_id} not found in {json_file}")


def draw_info_on_image(image, text_lines, position=(10, 10)):
    """在图像上绘制多行文本信息"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    y_offset = position[1]
    for line in text_lines:
        draw.text((position[0], y_offset), line, fill=(255, 255, 0), font=font,
                 stroke_width=1, stroke_fill=(0, 0, 0))
        y_offset += 20
    
    return image


def save_point_cloud_ply(points_3d, output_path, colors=None, bbox_3d=None, use_obb=False):
    """
    保存点云为PLY格式，可选地包含3D bounding box
    
    Args:
        points_3d: 点云数据 (N, 3)
        output_path: 输出文件路径
        colors: 点云颜色 (N, 3)，如果为None则不保存颜色
        bbox_3d: 3D bbox字典，如果提供则会添加bbox的角点和边
        use_obb: 是否使用OBB（有方向的包围盒）而不是AABB（轴对齐包围盒）
    """
    # 准备顶点数据
    vertices = []
    vertex_colors = []
    
    # 添加点云顶点
    for i, point in enumerate(points_3d):
        vertices.append(point)
        if colors is not None:
            vertex_colors.append(colors[i])
        else:
            vertex_colors.append([0, 255, 0])  # 绿色（点云）
    
    # 如果提供了bbox，添加bbox的8个角点
    bbox_vertex_start = len(vertices)
    edges = []
    
    if bbox_3d is not None:
        # 选择使用OBB或AABB的角点
        if use_obb and 'obb_corners' in bbox_3d:
            bbox_corners = bbox_3d['obb_corners']
        else:
            bbox_corners = bbox_3d['corners']
        for corner in bbox_corners:
            vertices.append(corner)
            vertex_colors.append([255, 0, 0])  # 红色（bbox）
        
        # 定义bbox的12条边（连接8个角点）
        # 角点索引从bbox_vertex_start开始
        edges = [
            # 底面4条边
            (bbox_vertex_start + 0, bbox_vertex_start + 1),
            (bbox_vertex_start + 1, bbox_vertex_start + 2),
            (bbox_vertex_start + 2, bbox_vertex_start + 3),
            (bbox_vertex_start + 3, bbox_vertex_start + 0),
            # 顶面4条边
            (bbox_vertex_start + 4, bbox_vertex_start + 5),
            (bbox_vertex_start + 5, bbox_vertex_start + 6),
            (bbox_vertex_start + 6, bbox_vertex_start + 7),
            (bbox_vertex_start + 7, bbox_vertex_start + 4),
            # 连接底面和顶面的4条边
            (bbox_vertex_start + 0, bbox_vertex_start + 4),
            (bbox_vertex_start + 1, bbox_vertex_start + 5),
            (bbox_vertex_start + 2, bbox_vertex_start + 6),
            (bbox_vertex_start + 3, bbox_vertex_start + 7),
        ]
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        
        # 如果有边，添加边元素
        if edges:
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
        
        f.write("end_header\n")
        
        # 写入顶点数据
        for i, vertex in enumerate(vertices):
            color = vertex_colors[i]
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        
        # 写入边数据
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")


def visualize_points_on_image(image, points_3d, hfov=90.0, color=(0, 255, 0), point_size=2):
    """将3D点云投影到2D图像上
    
    注意：点云坐标系中，z为负数表示在相机前方（相机朝向-Z方向）
    """
    H, W = image.shape[:2]
    
    # 计算相机内参
    focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    
    # 创建图像副本
    img_with_points = image.copy()
    
    # 投影每个点
    for point in points_3d:
        x, y, z = point
        if z >= 0:  # 在相机后面或相机平面上，跳过（z为负数表示在前方）
            continue
        
        # 投影到2D（参考官方 view_transform.py，除以负的深度）
        # 因为相机坐标系中 z 是负数，所以除以 -z 得到正的深度值
        u = int(x * focal_length / (-z) + cx)
        v = int(-y * focal_length / (-z) + cy)
        
        # 检查是否在图像范围内
        if 0 <= u < W and 0 <= v < H:
            # 绘制点
            cv2.circle(img_with_points, (u, v), point_size, color, -1)
    
    return img_with_points


def create_point_cloud_visualization(rgb_np, points_3d, bbox_3d, hfov=90.0, use_obb=False):
    """创建包含点云和bbox的综合可视化
    
    Args:
        use_obb: 是否使用OBB（有方向的包围盒）而不是AABB（轴对齐包围盒）
    """
    # 1. 只有RGB
    img_rgb = rgb_np.copy()
    
    # 2. RGB + 点云
    img_with_points = visualize_points_on_image(
        rgb_np.copy(), points_3d, hfov, color=(0, 255, 0), point_size=1
    )
    
    # 3. RGB + 点云 + bbox
    img_with_both = visualize_points_on_image(
        rgb_np.copy(), points_3d, hfov, color=(0, 255, 0), point_size=1
    )
    img_with_both = visualize_bbox_on_image(
        img_with_both, bbox_3d, hfov, color=(255, 0, 0), thickness=2, use_obb=use_obb
    )
    
    # 4. 只有bbox
    img_with_bbox = visualize_bbox_on_image(
        rgb_np.copy(), bbox_3d, hfov, color=(255, 0, 0), thickness=2, use_obb=use_obb
    )
    
    return img_rgb, img_with_points, img_with_both, img_with_bbox


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试3D Bounding Box预测效果')
    parser.add_argument('--task_id', type=int, default=14,
                        help='目标任务ID (默认: 14)')
    parser.add_argument('--json_file', type=str,
                        default='/data1/tct_data/habitat/eval_data/replicacad_10-segment/task_infos.json',
                        help='任务JSON文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/tct_data/habitat/test_outputs/3dbox',
                        help='输出目录路径')
    parser.add_argument('--debug_vis', action='store_true',
                        help='启用旋转搜索调试可视化')
    parser.add_argument('--configs', type=str, default='all',
                        choices=['all', 'basic', 'standard', 'advanced'],
                        help='要测试的配置: all(全部), basic(基础), standard(标准), advanced(高精度)')
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 配置
    json_file = args.json_file
    target_task_id = args.task_id
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("3D Bounding Box 测试")
    print("="*80)
    print(f"加载任务 ID: {target_task_id}")
    print(f"从文件: {json_file}\n")
    
    # 加载任务
    task_info = load_task_by_id(json_file, target_task_id)
    
    print(f"任务信息:")
    print(f"  Task ID: {task_info['task_id']}")
    print(f"  Task Type: {task_info['task_type']}")
    print(f"  Target Category: {task_info['target_category']}")
    print(f"  Scene ID: {task_info['scene_id']}")
    print(f"  Task Prompt: {task_info['task_prompt']}")
    print(f"  Instance ID: {task_info.get('instance_id')}")
    
    # 初始化环境
    print("\n初始化环境...")
    dataset_name = "ReplicaCAD"
    seed = 42
    scenes_size = 10
    max_scene_instance = 20
    max_step_length = 10
    
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    
    # 使用 reset_eval 加载任务
    print("\n加载任务到环境...")
    obs, info = env.reset_eval(sync_info=task_info)
    
    # for _ in range(4):
    #     env.sim.step("move_forward")

    print(f"\n环境已加载:")
    print(f"  观察图像大小: {obs.size}")
    
    # 获取仿真器和传感器信息
    print("\n获取传感器配置...")
    sim = env.sim
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    
    # 获取传感器位置
    sensor_spec = agent.agent_config.sensor_specifications[0]  # rgb传感器
    sensor_position = np.array(sensor_spec.position)
    
    # 获取hfov (可能是magnum.Deg对象，需要转换为float)
    hfov_raw = sensor_spec.hfov
    if hasattr(hfov_raw, '__float__'):
        hfov = float(hfov_raw)
    else:
        hfov = 90.0  # 默认值
    
    print(f"  hfov: {hfov}° (原始: {hfov_raw})")
    print(f"  传感器位置: {sensor_position}")
    print(f"  Agent位置: {agent_state.position}")
    print(f"  Agent旋转: {agent_state.rotation}")
    
    # 获取观测数据
    print("\n获取观测数据...")
    obs_dict = sim.get_sensor_observations()
    semantic_obs = obs_dict["semantic"]
    depth_obs = obs_dict["depth"]
    rgb_obs = obs_dict["rgb"]
    
    print(f"  语义图大小: {semantic_obs.shape}")
    print(f"  深度图大小: {depth_obs.shape}")
    print(f"  深度范围: {depth_obs.min():.2f} - {depth_obs.max():.2f} 米")
    
    # 获取目标实例ID
    instance_id = task_info.get('semantic_id')
    target_category = task_info.get('target_category')
    
    if instance_id is None:
        print("\n错误: 任务中没有instance_id信息")
        env.close()
        return
    
    print(f"\n目标物体:")
    print(f"  Instance ID: {instance_id}")
    print(f"  Category: {target_category}")
    
    # 检查目标物体是否可见
    unique_ids = np.unique(semantic_obs)
    if instance_id not in unique_ids:
        print(f"\n警告: 目标物体 (ID={instance_id}) 在当前视角不可见！")
        print(f"  可见的物体ID: {unique_ids[:20]}...")
    else:
        mask = (semantic_obs == instance_id)
        visible_pixels = np.sum(mask)
        print(f"  可见像素数: {visible_pixels}")
    
    # 测试不同配置的3D bbox预测
    print("\n" + "="*80)
    print("测试不同配置的3D Bounding Box预测")
    print("="*80)
    
    # 定义所有可用的配置
    all_configs = {
        'basic': {
            "name": "基础模式（无去噪）",
            "denoise": False,
            "align_to_ground": False,
            "use_rotation_search": False
        },
        'standard': {
            "name": "标准模式（去噪+地面对齐）",
            "denoise": True,
            "align_to_ground": True,
            "use_rotation_search": False
        },
        'advanced': {
            "name": "高精度模式（去噪+地面对齐+旋转搜索）",
            "denoise": True,
            "align_to_ground": True,
            "use_rotation_search": True
        }
    }
    
    # 根据命令行参数选择配置
    if args.configs == 'all':
        configs = [all_configs['basic'], all_configs['standard'], all_configs['advanced']]
    else:
        configs = [all_configs[args.configs]]
    
    print(f"测试配置: {[c['name'] for c in configs]}")
    print(f"调试可视化: {'启用' if args.debug_vis else '禁用'}")
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}")
        print("-" * 60)
        
        import time
        start_time = time.time()
        
        # 预测3D bbox（相机坐标系）
        bbox_camera, points_3d = predict_3d_bbox(
            semantic_obs=semantic_obs,
            depth_obs=depth_obs,
            instance_id=instance_id,
            hfov=hfov,
            return_world_coords=False,
            denoise=config['denoise'],
            align_to_ground=config['align_to_ground'],
            use_rotation_search=config['use_rotation_search'],
            debug_vis=args.debug_vis
        )
        
        elapsed_time = time.time() - start_time
        
        if bbox_camera is None:
            print(f"  ✗ 预测失败")
            results.append(None)
            continue
        
        # 预测3D bbox（世界坐标系）
        bbox_world, _ = predict_3d_bbox(
            semantic_obs=semantic_obs,
            depth_obs=depth_obs,
            instance_id=instance_id,
            agent_state=agent_state,
            hfov=hfov,
            return_world_coords=True,
            denoise=config['denoise'],
            align_to_ground=config['align_to_ground'],
            use_rotation_search=config['use_rotation_search'],
            sensor_name='rgb'  # 使用sensor_name代替sensor_position
        )
        
        # 打印结果
        print(f"  ✓ 预测成功 (耗时: {elapsed_time*1000:.2f}ms)")
        print(f"  原始点数: {bbox_camera['num_points_original']}")
        print(f"  过滤后点数: {bbox_camera['num_points_filtered']}")
        
        # 显示OBB轴信息（如果启用了align_to_ground）
        if 'obb_axes' in bbox_camera and config.get('align_to_ground', False):
            obb_axes = bbox_camera['obb_axes']
            print(f"\n  OBB轴方向 (相机坐标系):")
            print(f"    X轴: [{obb_axes[0,0]:+.3f}, {obb_axes[1,0]:+.3f}, {obb_axes[2,0]:+.3f}]")
            print(f"    Y轴: [{obb_axes[0,1]:+.3f}, {obb_axes[1,1]:+.3f}, {obb_axes[2,1]:+.3f}]  ← 应该是 [0, 1, 0]")
            print(f"    Z轴: [{obb_axes[0,2]:+.3f}, {obb_axes[1,2]:+.3f}, {obb_axes[2,2]:+.3f}]")
            
            # 验证Y轴对齐
            y_axis = obb_axes[:, 1]
            expected_y = np.array([0.0, 1.0, 0.0])
            y_alignment_error = np.linalg.norm(y_axis - expected_y)
            if y_alignment_error < 1e-6:
                print(f"    ✓ Y轴完美对齐 (误差: {y_alignment_error:.2e})")
            else:
                print(f"    ⚠ Y轴对齐误差: {y_alignment_error:.2e}")
        
        print(f"\n  相机坐标系:")
        print(f"    AABB中心: [{bbox_camera['center'][0]:.3f}, {bbox_camera['center'][1]:.3f}, {bbox_camera['center'][2]:.3f}]")
        print(f"    AABB尺寸: [{bbox_camera['size'][0]:.3f}, {bbox_camera['size'][1]:.3f}, {bbox_camera['size'][2]:.3f}]")
        if 'obb_extents' in bbox_camera:
            obb_size = bbox_camera['obb_extents'] * 2
            print(f"    OBB尺寸:  [{obb_size[0]:.3f}, {obb_size[1]:.3f}, {obb_size[2]:.3f}]")
        
        print(f"\n  世界坐标系:")
        print(f"    AABB中心: [{bbox_world['center'][0]:.3f}, {bbox_world['center'][1]:.3f}, {bbox_world['center'][2]:.3f}]")
        print(f"    AABB尺寸: [{bbox_world['size'][0]:.3f}, {bbox_world['size'][1]:.3f}, {bbox_world['size'][2]:.3f}]")
        
        # 保存点云数据
        # 1. 保存为.npy格式（用于后续分析）
        npy_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points.npy')
        np.save(npy_path, points_3d)
        print(f"  ✓ 点云已保存 (.npy): {npy_path}")
        
        # 2. 保存为.ply格式（只有点云）
        ply_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_only.ply')
        save_point_cloud_ply(points_3d, ply_path, bbox_3d=None)
        print(f"  ✓ 点云已保存 (.ply): {ply_path}")
        
        # 3. 保存为.ply格式（点云+bbox，相机坐标系）
        # 保存OBB（旋转搜索找到的最优框）
        ply_bbox_obb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_with_bbox_camera_OBB.ply')
        save_point_cloud_ply(points_3d, ply_bbox_obb_path, bbox_3d=bbox_camera, use_obb=True)
        print(f"  ✓ 点云+BBox已保存 (相机坐标系, OBB): {ply_bbox_obb_path}")
        
        # 保存AABB（轴对齐框）作为对比
        ply_bbox_aabb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_with_bbox_camera_AABB.ply')
        save_point_cloud_ply(points_3d, ply_bbox_aabb_path, bbox_3d=bbox_camera, use_obb=False)
        print(f"  ✓ 点云+BBox已保存 (相机坐标系, AABB): {ply_bbox_aabb_path}")
        
        # 4. 保存为.ply格式（点云+bbox，世界坐标系）- 需要转换点云
        # 将点云转换到世界坐标系（使用通用函数）
        agent = env.sim.get_agent(0)
        agent_state = agent.get_state()
        points_3d_world = transform_points_to_world_coords(points_3d, agent_state, sensor_name='rgb')
        
        # 保存OBB（旋转搜索找到的最优框）
        ply_bbox_world_obb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_with_bbox_world_OBB.ply')
        save_point_cloud_ply(points_3d_world, ply_bbox_world_obb_path, bbox_3d=bbox_world, use_obb=True)
        print(f"  ✓ 点云+BBox已保存 (世界坐标系, OBB): {ply_bbox_world_obb_path}")
        
        # 保存AABB（轴对齐框）作为对比
        ply_bbox_world_aabb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_with_bbox_world_AABB.ply')
        save_point_cloud_ply(points_3d_world, ply_bbox_world_aabb_path, bbox_3d=bbox_world, use_obb=False)
        print(f"  ✓ 点云+BBox已保存 (世界坐标系, AABB): {ply_bbox_world_aabb_path}")
        
        # 保存结果
        results.append({
            'config': config,
            'bbox_camera': bbox_camera,
            'bbox_world': bbox_world,
            'points_3d': points_3d,
            'elapsed_time': elapsed_time
        })
    
    # 可视化对比
    print("\n" + "="*80)
    print("生成可视化（包含点云）")
    print("="*80)
    
    # 转换RGB观测为numpy数组
    rgb_np = np.array(rgb_obs)
    
    # 为每个配置创建可视化
    vis_images = []
    for i, result in enumerate(results):
        if result is None:
            # 创建一个空白图像
            vis_img = Image.fromarray(rgb_np)
            draw = ImageDraw.Draw(vis_img)
            draw.text((10, 10), "预测失败", fill=(255, 0, 0))
            vis_images.append(vis_img)
            continue
        
        config = result['config']
        bbox_camera = result['bbox_camera']
        points_3d = result['points_3d']
        
        print(f"\n配置 {i+1}: {config['name']}")
        
        # 创建OBB可视化（旋转搜索找到的最优框）
        print(f"  生成OBB可视化...")
        img_rgb, img_with_points, img_with_both_obb, img_with_bbox_obb = create_point_cloud_visualization(
            rgb_np, points_3d, bbox_camera, hfov=hfov, use_obb=True
        )
        
        # 创建AABB可视化（轴对齐框）作为对比
        print(f"  生成AABB可视化...")
        _, _, img_with_both_aabb, img_with_bbox_aabb = create_point_cloud_visualization(
            rgb_np, points_3d, bbox_camera, hfov=hfov, use_obb=False
        )
        
        # 保存单独的可视化
        # 1. 只有点云
        cv2.imwrite(
            os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_only.png'),
            cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR)
        )
        print(f"  ✓ 保存点云可视化")
        
        # 2. 点云 + bbox (OBB)
        cv2.imwrite(
            os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_bbox_OBB.png'),
            cv2.cvtColor(img_with_both_obb, cv2.COLOR_RGB2BGR)
        )
        print(f"  ✓ 保存点云+bbox可视化 (OBB)")
        
        # 3. 点云 + bbox (AABB)
        cv2.imwrite(
            os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_points_bbox_AABB.png'),
            cv2.cvtColor(img_with_both_aabb, cv2.COLOR_RGB2BGR)
        )
        print(f"  ✓ 保存点云+bbox可视化 (AABB)")
        
        # 4. 只有OBB（带信息标注）
        img_with_bbox_obb_pil = Image.fromarray(img_with_bbox_obb)
        obb_size = bbox_camera['obb_extents'] * 2 if 'obb_extents' in bbox_camera else bbox_camera['size']
        info_lines_obb = [
            config['name'] + " [OBB]",
            f"Time: {result['elapsed_time']*1000:.1f}ms",
            f"Points: {bbox_camera['num_points_original']} -> {bbox_camera['num_points_filtered']}",
            f"OBB Size: [{obb_size[0]:.2f}, {obb_size[1]:.2f}, {obb_size[2]:.2f}]m",
            f"OBB Center: [{bbox_camera.get('obb_center', bbox_camera['center'])[0]:.2f}, {bbox_camera.get('obb_center', bbox_camera['center'])[1]:.2f}, {bbox_camera.get('obb_center', bbox_camera['center'])[2]:.2f}]m"
        ]
        vis_img_obb = draw_info_on_image(img_with_bbox_obb_pil, info_lines_obb, position=(10, 10))
        
        # 5. 只有AABB（带信息标注）
        img_with_bbox_aabb_pil = Image.fromarray(img_with_bbox_aabb)
        info_lines_aabb = [
            config['name'] + " [AABB]",
            f"Time: {result['elapsed_time']*1000:.1f}ms",
            f"Points: {bbox_camera['num_points_original']} -> {bbox_camera['num_points_filtered']}",
            f"AABB Size: [{bbox_camera['size'][0]:.2f}, {bbox_camera['size'][1]:.2f}, {bbox_camera['size'][2]:.2f}]m",
            f"AABB Center: [{bbox_camera['center'][0]:.2f}, {bbox_camera['center'][1]:.2f}, {bbox_camera['center'][2]:.2f}]m"
        ]
        vis_img_aabb = draw_info_on_image(img_with_bbox_aabb_pil, info_lines_aabb, position=(10, 10))
        
        # 使用OBB作为主可视化
        vis_images.append(vis_img_obb)
        
        # 保存带标注的bbox图像
        obb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_OBB.png')
        vis_img_obb.save(obb_path)
        print(f"  ✓ 保存OBB可视化（带标注）")
        
        aabb_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_AABB.png')
        vis_img_aabb.save(aabb_path)
        print(f"  ✓ 保存AABB可视化（带标注）")
        
        # 6. 创建对比图: RGB | Points | OBB | AABB
        combined = np.hstack([img_rgb, img_with_points, img_with_bbox_obb, img_with_bbox_aabb])
        combined_path = os.path.join(output_dir, f'task_{target_task_id}_config_{i+1}_combined.png')
        cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  ✓ 保存4合1对比图 (RGB|Points|OBB|AABB)")
    
    # 创建对比图（3列）
    if vis_images:
        print("\n创建对比图...")
        w, h = vis_images[0].size
        comparison_img = Image.new('RGB', (w * len(vis_images), h))
        
        for i, img in enumerate(vis_images):
            comparison_img.paste(img, (w * i, 0))
        
        comparison_path = os.path.join(output_dir, f'task_{target_task_id}_comparison.png')
        comparison_img.save(comparison_path)
        print(f"  ✓ 对比图已保存到: {comparison_path}")
    
    # 比较不同配置之间的IoU
    if len(results) >= 2 and all(r is not None for r in results):
        print("\n" + "="*80)
        print("配置间的3D IoU对比")
        print("="*80)
        
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                iou = compute_bbox_iou_3d(
                    results[i]['bbox_camera'],
                    results[j]['bbox_camera']
                )
                print(f"  配置{i+1} vs 配置{j+1}: IoU = {iou:.4f}")
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("测试摘要")
    print("="*80)
    print(f"任务 ID: {target_task_id}")
    print(f"类别: {target_category}")
    print(f"实例 ID: {instance_id}")
    print(f"场景: {task_info['scene_id']}")
    print(f"\n成功预测: {sum(1 for r in results if r is not None)}/{len(configs)}")
    
    if any(r is not None for r in results):
        print(f"\n性能对比:")
        for i, result in enumerate(results):
            if result is not None:
                print(f"  配置{i+1}: {result['elapsed_time']*1000:.2f}ms")
        
        print(f"\n点云统计:")
        for i, result in enumerate(results):
            if result is not None:
                bbox = result['bbox_camera']
                print(f"  配置{i+1}: {bbox['num_points_original']} -> {bbox['num_points_filtered']} 点")
        
        print(f"\nBBox尺寸对比 (相机坐标系):")
        for i, result in enumerate(results):
            if result is not None:
                size = result['bbox_camera']['size']
                print(f"  配置{i+1}: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] 米")
    
    print("="*80)
    
    # 保存详细结果到JSON
    output_json = {
        'task_id': target_task_id,
        'task_info': task_info,
        'sensor_position': sensor_position.tolist(),
        'agent_position': agent_state.position.tolist(),
        'results': []
    }
    
    for i, result in enumerate(results):
        if result is not None:
            output_json['results'].append({
                'config_name': result['config']['name'],
                'elapsed_time_ms': result['elapsed_time'] * 1000,
                'num_points_original': int(result['bbox_camera']['num_points_original']),
                'num_points_filtered': int(result['bbox_camera']['num_points_filtered']),
                'bbox_camera': {
                    'center': result['bbox_camera']['center'].tolist(),
                    'size': result['bbox_camera']['size'].tolist(),
                    'min_bound': result['bbox_camera']['min_bound'].tolist(),
                    'max_bound': result['bbox_camera']['max_bound'].tolist(),
                },
                'bbox_world': {
                    'center': result['bbox_world']['center'].tolist(),
                    'size': result['bbox_world']['size'].tolist(),
                    'min_bound': result['bbox_world']['min_bound'].tolist(),
                    'max_bound': result['bbox_world']['max_bound'].tolist(),
                }
            })
    
    json_path = os.path.join(output_dir, f'task_{target_task_id}_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2)
    print(f"\n✓ 详细结果已保存到: {json_path}")
    
    # 清理
    env.close()
    print("\n测试完成！")


if __name__ == "__main__":
    # 初始化 ray
    ray.init(_temp_dir="/data1/tct_data/verl-agent/tmp", ignore_reinit_error=True)
    
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

