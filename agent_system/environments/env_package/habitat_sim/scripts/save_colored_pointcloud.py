"""
简单脚本：从任务场景构建3D彩色点云

功能：
- 加载指定任务场景
- 从RGB图和深度图构建完整的3D彩色点云
- 保存为PLY格式
- 可选：将点云转换到世界坐标系

使用方法:
    # 保存指定任务的点云（相机坐标系）
    python save_colored_pointcloud.py --task_id 14
    
    # 保存多个任务
    python save_colored_pointcloud.py --task_id 14 24 16
    
    # 保存世界坐标系的点云
    python save_colored_pointcloud.py --task_id 14 --save_world_coords
    
    # 自定义输出目录
    python save_colored_pointcloud.py --task_id 14 --output_dir /path/to/output
    
    # 限制点云密度（采样）
    python save_colored_pointcloud.py --task_id 14 --sample_rate 4
    
    # 限制深度范围
    python save_colored_pointcloud.py --task_id 14 --max_depth 10.0

参数说明:
    --task_id: 目标任务ID，可以指定多个
    --json_file: 任务JSON文件路径
    --output_dir: 输出目录路径
    --sample_rate: 采样率，每N个像素取1个 (默认: 2)
    --max_depth: 最大深度限制，单位米 (默认: 15.0)
    --min_depth: 最小深度限制，单位米 (默认: 0.01)
    --save_world_coords: 保存世界坐标系的点云（默认只保存相机坐标系）
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import numpy as np
import argparse
import ray

# 导入环境
from agent_system.environments.env_package.habitat_sim.utils.habitat_envs import CreateHabitatEnv
from agent_system.environments.env_package.habitat_sim.utils.habitat_3dbox_utils import transform_points_to_world_coords


def load_task_by_id(json_file, task_id):
    """从 JSON 文件中加载指定 task_id 的任务"""
    with open(json_file, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    
    for task in all_tasks:
        if task.get('task_id') == task_id:
            return task
    
    raise ValueError(f"Task with task_id={task_id} not found in {json_file}")


def depth_to_colored_pointcloud(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    hfov: float = 90.0,
    sample_rate: int = 1,
    max_depth: float = 15.0,
    min_depth: float = 0.01
) -> tuple:
    """
    将RGB图像和深度图转换为彩色3D点云
    
    Args:
        rgb_image: RGB图像 (H, W, 3)，值范围0-255
        depth_map: 深度图 (H, W)，单位米
        hfov: 水平视场角（degrees）
        sample_rate: 采样率，每N个像素取1个
        max_depth: 最大深度限制
        min_depth: 最小深度限制
        
    Returns:
        points_3d: (N, 3) 3D点云坐标
        colors: (N, 3) 对应的RGB颜色，值范围0-255
    """
    H, W = depth_map.shape
    
    # 计算相机内参
    focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    
    # 生成像素网格（采样）
    v_coords, u_coords = np.meshgrid(
        np.arange(0, H, sample_rate),
        np.arange(0, W, sample_rate),
        indexing='ij'
    )
    
    # 展平
    v_coords = v_coords.flatten()
    u_coords = u_coords.flatten()
    
    # 获取对应的深度值和颜色
    z_values = depth_map[v_coords, u_coords]
    colors = rgb_image[v_coords, u_coords]
    
    # 过滤无效深度值
    valid_mask = (z_values > min_depth) & (z_values < max_depth)
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    z_values = z_values[valid_mask]
    colors = colors[valid_mask]
    
    if len(z_values) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # 将像素坐标转换为3D相机坐标
    # 参考 habitat 官方 view_transform.py:
    # 1. 相机朝向 -Z 方向，所以深度应该是负数
    # 2. 图像坐标系是 y-down（行从上到下），相机坐标系是 y-up，所以 y 需要取负
    x_coords = (u_coords - cx) * z_values / focal_length
    y_coords = -(v_coords - cy) * z_values / focal_length  # 注意负号
    z_coords = -z_values  # 注意负号，相机朝向 -Z
    
    # 组合成3D点云
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    
    return points_3d, colors


def save_colored_pointcloud_ply(points_3d, colors, output_path):
    """
    保存彩色点云为PLY格式
    
    Args:
        points_3d: (N, 3) 3D点云坐标
        colors: (N, 3) RGB颜色，值范围0-255
        output_path: 输出文件路径
    """
    if len(points_3d) == 0:
        print(f"  ⚠ 点云为空，跳过保存")
        return
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        # 写入PLY头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入顶点数据
        for i, point in enumerate(points_3d):
            color = colors[i].astype(int)
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{color[0]} {color[1]} {color[2]}\n")
    
    print(f"  ✓ 点云已保存: {output_path} ({len(points_3d)} 个点)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从任务场景构建3D彩色点云')
    parser.add_argument('--task_id', type=int, nargs='+', default=[14],
                        help='目标任务ID，可以指定多个 (默认: [14])')
    parser.add_argument('--json_file', type=str,
                        default='/data1/tct_data/habitat/eval_data/replicacad_10-segment/task_infos.json',
                        help='任务JSON文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/tct_data/habitat/test_outputs/pointcloud',
                        help='输出目录路径')
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='采样率，每N个像素取1个 (默认: 2)')
    parser.add_argument('--max_depth', type=float, default=15.0,
                        help='最大深度限制，单位米 (默认: 15.0)')
    parser.add_argument('--min_depth', type=float, default=0.01,
                        help='最小深度限制，单位米 (默认: 0.01)')
    parser.add_argument('--save_world_coords', action='store_true',
                        help='保存世界坐标系的点云（默认只保存相机坐标系）')
    return parser.parse_args()


def process_task(task_id, env, args):
    """处理单个任务"""
    print(f"\n{'='*80}")
    print(f"处理任务 ID: {task_id}")
    print(f"{'='*80}")
    
    # 加载任务
    try:
        task_info = load_task_by_id(args.json_file, task_id)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        return
    
    print(f"任务信息:")
    print(f"  Task ID: {task_info['task_id']}")
    print(f"  Task Type: {task_info['task_type']}")
    print(f"  Target Category: {task_info.get('target_category', 'N/A')}")
    print(f"  Scene ID: {task_info['scene_id']}")
    print(f"  Task Prompt: {task_info['task_prompt']}")
    
    # 加载任务到环境
    print("\n加载场景...")
    obs, info = env.reset_eval(sync_info=task_info)
    
    for _ in range(4):
        env.sim.step("look_down")

    # 获取传感器配置
    sim = env.sim
    agent = sim.get_agent(0)
    sensor_spec = agent.agent_config.sensor_specifications[0]
    
    # 获取hfov
    hfov_raw = sensor_spec.hfov
    if hasattr(hfov_raw, '__float__'):
        hfov = float(hfov_raw)
    else:
        hfov = 90.0
    
    print(f"  传感器配置:")
    print(f"    HFOV: {hfov}°")
    print(f"    图像尺寸: {obs.size}")
    
    # 获取观测数据
    obs_dict = sim.get_sensor_observations()
    rgb_obs = obs_dict["rgb"]
    depth_obs = obs_dict["depth"]
    
    print(f"\n观测数据:")
    print(f"  RGB图像: {rgb_obs.shape}")
    print(f"  深度图: {depth_obs.shape}")
    print(f"  深度范围: {depth_obs.min():.2f} - {depth_obs.max():.2f} 米")
    
    # 构建彩色点云
    print(f"\n构建3D彩色点云...")
    print(f"  采样率: 1/{args.sample_rate}")
    print(f"  深度范围: {args.min_depth} - {args.max_depth} 米")
    
    points_3d, colors = depth_to_colored_pointcloud(
        rgb_image=rgb_obs,
        depth_map=depth_obs,
        hfov=hfov,
        sample_rate=args.sample_rate,
        max_depth=args.max_depth,
        min_depth=args.min_depth
    )
    
    if len(points_3d) == 0:
        print(f"  ❌ 无有效点云")
        return
    
    print(f"  ✓ 生成点云: {len(points_3d)} 个点")
    print(f"  点云范围:")
    print(f"    X: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] 米")
    print(f"    Y: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] 米")
    print(f"    Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] 米")
    
    # 保存点云（相机坐标系）
    output_filename = f"task_{task_id}_pointcloud_camera.ply"
    output_path = os.path.join(args.output_dir, output_filename)
    save_colored_pointcloud_ply(points_3d, colors, output_path)
    
    # 保存为numpy格式（可选）
    npy_path = os.path.join(args.output_dir, f"task_{task_id}_pointcloud_camera.npy")
    np.save(npy_path, {'points': points_3d, 'colors': colors})
    print(f"  ✓ 点云数据已保存 (npy): {npy_path}")
    
    # 保存世界坐标系的点云（如果启用）
    if args.save_world_coords:
        print(f"\n转换点云到世界坐标系...")
        
        # 获取agent状态（用于获取sensor_states）
        agent = sim.get_agent(0)
        agent_state = agent.get_state()
        
        # 获取相机信息用于打印
        sensor_state = agent_state.sensor_states['rgb']
        camera_position = sensor_state.position
        camera_quaternion = sensor_state.rotation
        
        print(f"  相机世界位置: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
        print(f"  相机世界旋转: {camera_quaternion}")
        
        # 使用通用函数进行坐标转换
        points_3d_world = transform_points_to_world_coords(points_3d, agent_state, sensor_name='rgb')
        
        print(f"  ✓ 转换完成: {len(points_3d_world)} 个点")
        print(f"  世界坐标系点云范围:")
        print(f"    X: [{points_3d_world[:, 0].min():.2f}, {points_3d_world[:, 0].max():.2f}] 米")
        print(f"    Y: [{points_3d_world[:, 1].min():.2f}, {points_3d_world[:, 1].max():.2f}] 米")
        print(f"    Z: [{points_3d_world[:, 2].min():.2f}, {points_3d_world[:, 2].max():.2f}] 米")
        
        # 保存世界坐标系点云（PLY格式）
        output_filename_world = f"task_{task_id}_pointcloud_world.ply"
        output_path_world = os.path.join(args.output_dir, output_filename_world)
        save_colored_pointcloud_ply(points_3d_world, colors, output_path_world)
        
        # 保存世界坐标系点云（numpy格式）
        npy_path_world = os.path.join(args.output_dir, f"task_{task_id}_pointcloud_world.npy")
        np.save(npy_path_world, {'points': points_3d_world, 'colors': colors})
        print(f"  ✓ 世界坐标系点云数据已保存 (npy): {npy_path_world}")
    
    # 保存元数据
    metadata = {
        'task_id': task_id,
        'scene_id': task_info['scene_id'],
        'task_type': task_info['task_type'],
        'num_points': len(points_3d),
        'sample_rate': args.sample_rate,
        'depth_range': [args.min_depth, args.max_depth],
        'hfov': hfov,
        'image_size': list(rgb_obs.shape[:2]),
        'camera_coords_bounds': {
            'x': [float(points_3d[:, 0].min()), float(points_3d[:, 0].max())],
            'y': [float(points_3d[:, 1].min()), float(points_3d[:, 1].max())],
            'z': [float(points_3d[:, 2].min()), float(points_3d[:, 2].max())],
        },
        'save_world_coords': args.save_world_coords
    }
    
    # 如果保存了世界坐标系，也添加到元数据中
    if args.save_world_coords and 'points_3d_world' in locals():
        metadata['world_coords_bounds'] = {
            'x': [float(points_3d_world[:, 0].min()), float(points_3d_world[:, 0].max())],
            'y': [float(points_3d_world[:, 1].min()), float(points_3d_world[:, 1].max())],
            'z': [float(points_3d_world[:, 2].min()), float(points_3d_world[:, 2].max())],
        }
        # 使用 sensor_state 中的相机世界位置
        if hasattr(agent_state, 'sensor_states') and 'rgb' in agent_state.sensor_states:
            camera_position = agent_state.sensor_states['rgb'].position
            metadata['camera_world_position'] = [float(camera_position[0]), float(camera_position[1]), float(camera_position[2])]
    
    metadata_path = os.path.join(args.output_dir, f"task_{task_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ 元数据已保存: {metadata_path}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("3D彩色点云构建工具")
    print("="*80)
    print(f"任务ID: {args.task_id}")
    print(f"JSON文件: {args.json_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"采样率: 1/{args.sample_rate}")
    print(f"深度范围: {args.min_depth} - {args.max_depth} 米")
    print(f"保存世界坐标系: {'是' if args.save_world_coords else '否'}")
    
    # 初始化环境
    print("\n初始化环境...")
    dataset_name = "ReplicaCAD"
    seed = 42
    scenes_size = 10
    max_scene_instance = 20
    max_step_length = 10
    
    env = CreateHabitatEnv(seed, dataset_name, scenes_size, max_scene_instance, max_step_length)
    
    # 处理每个任务
    for task_id in args.task_id:
        try:
            process_task(task_id, env, args)
        except Exception as e:
            print(f"\n❌ 处理任务 {task_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 清理
    env.close()
    
    print(f"\n{'='*80}")
    print("所有任务处理完成！")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*80}")


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

