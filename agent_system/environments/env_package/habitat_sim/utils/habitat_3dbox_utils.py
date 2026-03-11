import numpy as np
import habitat_sim
from typing import Tuple, Optional, Dict, Union
import magnum as mn
from pycocotools import mask as mask_utils


def transform_points_to_world_coords(
    points_camera: np.ndarray,
    agent_state: 'habitat_sim.AgentState',
    sensor_name: str = 'rgb'
) -> np.ndarray:
    """
    将点云从相机坐标系转换到世界坐标系
    
    使用官方推荐方式（参考 habitat view_transform.py）：
    1. 直接使用 sensor_states 中的传感器世界坐标
    2. 使用 quaternion 库进行旋转矩阵转换
    3. 使用 4x4 变换矩阵和齐次坐标进行转换
    
    Args:
        points_camera: (N, 3) 相机坐标系下的点云
        agent_state: Agent状态对象，需要包含 sensor_states
        sensor_name: 传感器名称，默认 'rgb'
        
    Returns:
        points_world: (N, 3) 世界坐标系下的点云
        
    Raises:
        ValueError: 如果 sensor_states 不可用或指定的传感器不存在
    """
    # 检查 sensor_states 是否可用
    if not hasattr(agent_state, 'sensor_states') or sensor_name not in agent_state.sensor_states:
        raise ValueError(
            f"sensor_states 不可用或传感器 '{sensor_name}' 不存在。"
            f"可用的传感器: {list(agent_state.sensor_states.keys()) if hasattr(agent_state, 'sensor_states') else 'None'}"
        )
    
    # 获取传感器世界坐标（已经是世界坐标系）
    sensor_state = agent_state.sensor_states[sensor_name]
    camera_position = sensor_state.position
    camera_quaternion = sensor_state.rotation
    
    # 使用 quaternion 库转换（habitat 官方方式）
    try:
        import quaternion
        
        # 将四元数转换为旋转矩阵
        # habitat-sim 的四元数格式需要转换为 numpy-quaternion 格式
        if hasattr(camera_quaternion, 'w'):
            # magnum.Quaternion 格式
            quat = np.quaternion(
                camera_quaternion.w,
                camera_quaternion.x, 
                camera_quaternion.y,
                camera_quaternion.z
            )
        else:
            # 已经是 quaternion 格式
            quat = camera_quaternion
        
        rotation_matrix = quaternion.as_rotation_matrix(quat)
        
    except ImportError:
        raise ImportError(
            "需要安装 numpy-quaternion 库来进行坐标转换。"
            "请运行: pip install numpy-quaternion"
        )
    
    # 构建 4x4 变换矩阵（相机坐标系 -> 世界坐标系）
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = rotation_matrix
    T_world_camera[0:3, 3] = camera_position
    
    # 将点云转换为齐次坐标 (N, 4)
    points_homogeneous = np.hstack([points_camera, np.ones((len(points_camera), 1))])
    
    # 应用变换矩阵
    points_world_homogeneous = np.matmul(T_world_camera, points_homogeneous.T).T
    points_world = points_world_homogeneous[:, 0:3]
    
    return points_world


def depth_to_3d_points(
    depth_map: np.ndarray,
    mask: np.ndarray,
    hfov: float = 90.0,
) -> np.ndarray:
    """
    将深度图和mask转换为3D点云（相机坐标系）
    
    Args:
        depth_map: 深度图 (H, W)，单位米，表示到观测平面的距离（Z方向距离）
        mask: 二值mask (H, W)，标识目标物体的像素
        hfov: 水平视场角（degrees），默认90度
        
    Returns:
        points_3d: (N, 3) 的3D点云，在相机坐标系下，N是mask中True的点数
    """
    H, W = depth_map.shape
    
    # 计算相机内参
    # hfov是水平视场角，计算焦距
    focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    
    # 相机中心（假设在图像中心）
    cx = W / 2.0
    cy = H / 2.0
    
    # 获取mask中为True的像素坐标
    v_coords, u_coords = np.where(mask)  # v是行（y），u是列（x）
    
    # 获取对应的深度值
    z_values = depth_map[v_coords, u_coords]
    
    # 过滤掉无效深度值（0或非常小的值）
    valid_depth_mask = z_values > 0.01
    u_coords = u_coords[valid_depth_mask]
    v_coords = v_coords[valid_depth_mask]
    z_values = z_values[valid_depth_mask]

    
    if len(z_values) == 0:
        return np.array([]).reshape(0, 3)
    
    # 将像素坐标转换为3D相机坐标
    # 参考 habitat 官方 view_transform.py:
    # 1. 相机朝向 -Z 方向，所以深度应该是负数
    # 2. 图像坐标系是 y-down（行从上到下），相机坐标系是 y-up，所以 y 需要取负
    x_coords = (u_coords - cx) * z_values / focal_length
    y_coords = -(v_coords - cy) * z_values / focal_length  # 注意负号
    z_coords = -z_values  # 注意负号，相机朝向 -Z
    
    # 组合成3D点云
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    
    return points_3d


def _statistical_outlier_removal(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """
    统计离群点去除
    
    Args:
        points: (N, 3) 点云
        k: 每个点考虑的邻居数量
        std_ratio: 标准差倍数，超过此阈值的点被视为离群点
        
    Returns:
        filtered_points: 过滤后的点云
    """
    if len(points) < k:
        return points
    
    from scipy.spatial import cKDTree
    
    # 构建KD树
    tree = cKDTree(points)
    
    # 计算每个点到其k个最近邻的平均距离
    distances, _ = tree.query(points, k=k+1)  # k+1因为包含点自身
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身距离
    
    # 计算全局平均距离和标准差
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # 过滤离群点
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold
    
    return points[inlier_mask]


def _voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """
    体素下采样
    
    Args:
        points: (N, 3) 点云
        voxel_size: 体素大小（米）
        
    Returns:
        downsampled_points: 下采样后的点云
    """
    if len(points) == 0:
        return points
    
    # 将点云映射到体素网格
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # 使用字典来存储每个体素中的点
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    
    # 计算每个体素的中心点
    downsampled_points = np.array([
        np.mean(voxel_points, axis=0) 
        for voxel_points in voxel_dict.values()
    ])
    
    return downsampled_points


def _rotate_search_minimum_area(points_2d: np.ndarray, num_angles: int = 180, debug_vis: bool = False) -> Tuple[float, float]:
    """
    在2D平面上旋转搜索最小面积包围盒
    
    Args:
        points_2d: (N, 2) 2D点云
        num_angles: 搜索的角度数量
        debug_vis: 是否显示调试可视化
        
    Returns:
        best_angle: 最佳旋转角度（弧度）
        min_area: 最小面积
    """
    angles = np.linspace(0, np.pi/2, num_angles)
    min_area = float('inf')
    best_angle = 0
    
    # 用于可视化的数据收集
    if debug_vis:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        
        vis_data = []
    
    for angle in angles:
        # 旋转矩阵
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 旋转点云
        rotated_points = points_2d @ rotation.T
        
        # 计算AABB面积
        min_coords = np.min(rotated_points, axis=0)
        max_coords = np.max(rotated_points, axis=0)
        area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        
        if debug_vis:
            vis_data.append({
                'angle': angle,
                'rotated_points': rotated_points.copy(),
                'min_coords': min_coords.copy(),
                'max_coords': max_coords.copy(),
                'area': area,
                'is_best': False
            })
        
        if area < min_area:
            min_area = area
            best_angle = angle
    
    # 标记最佳角度
    if debug_vis:
        for data in vis_data:
            if np.isclose(data['angle'], best_angle):
                data['is_best'] = True
        
        _visualize_rotation_search(points_2d, vis_data, best_angle, min_area)
    
    return best_angle, min_area


def _visualize_rotation_search(original_points: np.ndarray, vis_data: list, best_angle: float, min_area: float):
    """
    可视化旋转搜索过程
    
    Args:
        original_points: 原始2D点云
        vis_data: 包含每个角度的可视化数据
        best_angle: 最佳角度
        min_area: 最小面积
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    
    # 选择要可视化的角度样本（均匀采样）
    num_samples = min(12, len(vis_data))  # 最多显示12个子图
    sample_indices = np.linspace(0, len(vis_data)-1, num_samples, dtype=int)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    rows = int(np.ceil(num_samples / 4))
    cols = min(4, num_samples)
    
    for idx, sample_idx in enumerate(sample_indices):
        data = vis_data[sample_idx]
        ax = plt.subplot(rows + 1, cols, idx + 1)
        
        # 绘制旋转后的点云
        rotated_points = data['rotated_points']
        ax.scatter(rotated_points[:, 0], rotated_points[:, 1], 
                  c='blue' if not data['is_best'] else 'red', 
                  s=20, alpha=0.6, zorder=2)
        
        # 绘制AABB框
        min_coords = data['min_coords']
        max_coords = data['max_coords']
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        
        rect_color = 'green' if not data['is_best'] else 'red'
        rect_linewidth = 2 if not data['is_best'] else 4
        rect = patches.Rectangle(min_coords, width, height, 
                                linewidth=rect_linewidth, 
                                edgecolor=rect_color, 
                                facecolor='none',
                                zorder=3)
        ax.add_patch(rect)
        
        # 设置标题
        angle_deg = np.degrees(data['angle'])
        title = f"Angle: {angle_deg:.1f}°\nArea: {data['area']:.4f}"
        if data['is_best']:
            title = f"★ BEST ★\n{title}"
            ax.set_facecolor('#fff5f5')
        
        ax.set_title(title, fontsize=10, fontweight='bold' if data['is_best'] else 'normal')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # 在底部添加面积随角度变化的曲线图
    ax_curve = plt.subplot(rows + 1, 1, rows + 1)
    angles_deg = [np.degrees(d['angle']) for d in vis_data]
    areas = [d['area'] for d in vis_data]
    
    ax_curve.plot(angles_deg, areas, 'b-', linewidth=2, label='Area vs Angle')
    
    # 标记最小面积点
    best_angle_deg = np.degrees(best_angle)
    ax_curve.plot(best_angle_deg, min_area, 'r*', markersize=20, 
                 label=f'Best: {best_angle_deg:.1f}° (Area={min_area:.4f})', zorder=5)
    
    ax_curve.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax_curve.set_ylabel('AABB Area', fontsize=12)
    ax_curve.set_title('Area vs Rotation Angle', fontsize=14, fontweight='bold')
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/data1/tct_data/habitat/test_outputs/rotation_search_debug.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: /data1/tct_data/habitat/test_outputs/rotation_search_debug.png")
    print(f"最佳角度: {best_angle_deg:.2f}°, 最小面积: {min_area:.6f}")
    plt.show()


def compute_3d_bbox_from_points(
    points_3d: np.ndarray,
    denoise: bool = True,
    align_to_ground: bool = True,
    use_rotation_search: bool = False,
    debug_vis: bool = False,
) -> Dict:
    """
    从3D点云计算axis-aligned bounding box (AABB)，支持去噪和朝向估计
    
    Args:
        points_3d: (N, 3) 的3D点云
        denoise: 是否进行去噪处理（统计离群点去除）
        align_to_ground: 是否将盒子的一个轴对齐到地面法向（Y轴）
        use_rotation_search: 是否使用旋转搜索找最小面积包围盒（更抗噪但更慢）
        debug_vis: 是否显示旋转搜索的调试可视化
        
    Returns:
        bbox_dict: 包含3D box信息的字典
            - center: (3,) box中心点坐标（世界坐标系）
            - size: (3,) box尺寸 (width, height, depth)
            - min_bound: (3,) 最小边界点（AABB）
            - max_bound: (3,) 最大边界点（AABB）
            - corners: (8, 3) box的8个角点坐标（AABB）
            - obb_center: (3,) OBB中心
            - obb_axes: (3, 3) OBB主轴方向
            - obb_extents: (3,) OBB半长度
            - num_points_original: 原始点数
            - num_points_filtered: 过滤后点数
    """
    if len(points_3d) == 0:
        return None
    
    num_points_original = len(points_3d)
    
    # 1. 去噪与异常值剔除
    if denoise and len(points_3d) > 20:
        # 统计离群点去除
        points_filtered = _statistical_outlier_removal(points_3d, k=min(20, len(points_3d)//2), std_ratio=2.0)
        
        # 体素下采样（如果点云很密集）
        if len(points_filtered) > 100:
            # 自适应体素大小
            bbox_size = np.max(points_filtered, axis=0) - np.min(points_filtered, axis=0)
            voxel_size = np.mean(bbox_size) / 50  # 大约保留50个体素每边
            points_filtered = _voxel_downsample(points_filtered, voxel_size=voxel_size)
    else:
        points_filtered = points_3d
    
    if len(points_filtered) == 0:
        points_filtered = points_3d  # 如果过滤后没有点，使用原始点云
    
    num_points_filtered = len(points_filtered)
    
    # 2. 计算点云中心
    obb_center = np.mean(points_filtered, axis=0)
    centered_points = points_filtered - obb_center
    
    # 3. 初始化为标准坐标系（Y轴竖直）
    # 直接使用竖直朝向，跳过PCA（避免特征值分解的数值问题）
    obb_axes = np.eye(3)  # X=[1,0,0], Y=[0,1,0], Z=[0,0,1]
    
    # 4. 在XZ平面上旋转搜索最小面积包围盒
    if use_rotation_search and len(points_filtered) > 10:
        # Y轴是垂直轴，在XZ平面搜索
        points_2d = centered_points[:, [0, 2]]  # X和Z
        best_angle, _ = _rotate_search_minimum_area(points_2d, num_angles=90, debug_vis=debug_vis)
        
        # 应用旋转到主轴（只在XZ平面旋转，保持Y轴不变）
        cos_a, sin_a = np.cos(best_angle), np.sin(best_angle)
        rotation_y = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        obb_axes = obb_axes @ rotation_y
    
    # 5. 计算extents和调整中心
    projected_points = centered_points @ obb_axes
    min_proj = np.min(projected_points, axis=0)
    max_proj = np.max(projected_points, axis=0)
    obb_extents = (max_proj - min_proj) / 2.0
    obb_center_offset = obb_axes @ ((max_proj + min_proj) / 2.0)
    obb_center = obb_center + obb_center_offset
    
    # 4. 计算OBB的8个角点
    obb_corners_local = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]) * obb_extents
    
    obb_corners = obb_corners_local @ obb_axes.T + obb_center
    
    # 5. 将OBB转换为AABB（axis-aligned）
    min_bound = np.min(obb_corners, axis=0)
    max_bound = np.max(obb_corners, axis=0)
    
    # 计算AABB的中心和尺寸
    aabb_center = (min_bound + max_bound) / 2.0
    aabb_size = max_bound - min_bound
    
    # 计算AABB的8个角点
    aabb_corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
    ])
    
    bbox_dict = {
        # AABB信息
        'center': aabb_center,
        'size': aabb_size,
        'min_bound': min_bound,
        'max_bound': max_bound,
        'corners': aabb_corners,
        # OBB信息（用于高级应用）
        'obb_center': obb_center,
        'obb_axes': obb_axes,
        'obb_extents': obb_extents,
        'obb_corners': obb_corners,
        # 统计信息
        'num_points_original': num_points_original,
        'num_points_filtered': num_points_filtered,
    }
    
    return bbox_dict


def transform_bbox_to_world(
    bbox_camera: Dict,
    agent_state: habitat_sim.AgentState,
    sensor_name: str = 'rgb'
) -> Dict:
    """
    将相机坐标系下的bbox转换到世界坐标系
    
    使用与 transform_points_to_world_coords 相同的方式：
    1. 直接使用 sensor_states 中的传感器世界坐标
    2. 使用 quaternion 库进行旋转矩阵转换
    3. 使用 4x4 变换矩阵和齐次坐标进行转换
    
    Args:
        bbox_camera: 相机坐标系下的bbox字典
        agent_state: agent的状态，需要包含 sensor_states
        sensor_name: 传感器名称，默认 'rgb'
        
    Returns:
        bbox_world: 世界坐标系下的bbox字典
        
    Raises:
        ValueError: 如果 sensor_states 不可用或指定的传感器不存在
    """
    if bbox_camera is None:
        return None
    
    # 检查 sensor_states 是否可用
    if not hasattr(agent_state, 'sensor_states') or sensor_name not in agent_state.sensor_states:
        raise ValueError(
            f"sensor_states 不可用或传感器 '{sensor_name}' 不存在。"
            f"可用的传感器: {list(agent_state.sensor_states.keys()) if hasattr(agent_state, 'sensor_states') else 'None'}"
        )
    
    # 获取传感器世界坐标（已经是世界坐标系）
    sensor_state = agent_state.sensor_states[sensor_name]
    camera_position = sensor_state.position
    camera_quaternion = sensor_state.rotation
    
    # 使用 quaternion 库转换（habitat 官方方式）
    try:
        import quaternion
        
        # 将四元数转换为旋转矩阵
        if hasattr(camera_quaternion, 'w'):
            # magnum.Quaternion 格式
            quat = np.quaternion(
                camera_quaternion.w,
                camera_quaternion.x, 
                camera_quaternion.y,
                camera_quaternion.z
            )
        else:
            # 已经是 quaternion 格式
            quat = camera_quaternion
        
        rotation_matrix = quaternion.as_rotation_matrix(quat)
        
    except ImportError:
        raise ImportError(
            "需要安装 numpy-quaternion 库来进行坐标转换。"
            "请运行: pip install numpy-quaternion"
        )
    
    # 构建 4x4 变换矩阵（相机坐标系 -> 世界坐标系）
    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = rotation_matrix
    T_world_camera[0:3, 3] = camera_position
    
    # 转换中心点
    center_camera = bbox_camera['center']
    center_homogeneous = np.append(center_camera, 1.0)
    center_world = (T_world_camera @ center_homogeneous)[0:3]
    
    # 转换AABB角点
    corners_camera = bbox_camera['corners']
    corners_homogeneous = np.hstack([corners_camera, np.ones((len(corners_camera), 1))])
    corners_world = (T_world_camera @ corners_homogeneous.T).T[:, 0:3]
    
    # 转换OBB角点（如果存在）
    obb_corners_world = None
    if 'obb_corners' in bbox_camera:
        obb_corners_camera = bbox_camera['obb_corners']
        obb_corners_homogeneous = np.hstack([obb_corners_camera, np.ones((len(obb_corners_camera), 1))])
        obb_corners_world = (T_world_camera @ obb_corners_homogeneous.T).T[:, 0:3]
    
    # 转换OBB中心（如果存在）
    obb_center_world = None
    if 'obb_center' in bbox_camera:
        obb_center_camera = bbox_camera['obb_center']
        obb_center_homogeneous = np.append(obb_center_camera, 1.0)
        obb_center_world = (T_world_camera @ obb_center_homogeneous)[0:3]
    
    # 转换OBB轴方向（如果存在）
    # 注意：方向向量不需要平移，只需要旋转
    obb_axes_world = None
    if 'obb_axes' in bbox_camera:
        obb_axes_camera = bbox_camera['obb_axes']
        # 方向向量只旋转，不平移
        obb_axes_world = rotation_matrix @ obb_axes_camera
    
    # 重新计算世界坐标系下的min/max bounds
    min_bound_world = np.min(corners_world, axis=0)
    max_bound_world = np.max(corners_world, axis=0)
    
    bbox_world = {
        'center': center_world,
        'size': bbox_camera['size'],  # 尺寸在局部坐标系下不变
        'min_bound': min_bound_world,
        'max_bound': max_bound_world,
        'corners': corners_world,
    }
    
    # 添加OBB信息（如果存在）
    if obb_corners_world is not None:
        bbox_world['obb_corners'] = obb_corners_world
    if obb_center_world is not None:
        bbox_world['obb_center'] = obb_center_world
    if obb_axes_world is not None:
        bbox_world['obb_axes'] = obb_axes_world
    if 'obb_extents' in bbox_camera:
        bbox_world['obb_extents'] = bbox_camera['obb_extents']  # extents在局部坐标系下不变
    
    return bbox_world


def _create_default_bbox_3d(default_distance: float = 2.0, default_size: float = 0.1) -> Dict:
    """
    创建一个默认的3D bounding box（用于检测失败时）
    
    Args:
        default_distance: 默认距离（相机前方，米）
        default_size: 默认尺寸（边长，米）
        
    Returns:
        默认的 bbox_3d 字典
    """
    # 默认位置：相机前方 default_distance 米处
    center = np.array([0.0, 0.0, -default_distance])
    
    # 默认尺寸：一个小立方体
    size = np.array([default_size, default_size, default_size])
    half_size = size / 2.0
    
    # AABB 边界
    min_bound = center - half_size
    max_bound = center + half_size
    
    # AABB 角点
    corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
    ])
    
    # OBB 信息（默认对齐到坐标轴）
    obb_axes = np.eye(3)  # 单位矩阵
    obb_extents = half_size
    
    return {
        'center': center,
        'size': size,
        'min_bound': min_bound,
        'max_bound': max_bound,
        'corners': corners,
        'obb_center': center,
        'obb_axes': obb_axes,
        'obb_extents': obb_extents,
        'obb_corners': corners,
        'num_points_original': 0,
        'num_points_filtered': 0,
    }


def predict_3d_bbox(
    semantic_obs: np.ndarray,
    depth_obs: np.ndarray,
    instance_id: int,
    agent_state: Optional[habitat_sim.AgentState] = None,
    hfov: float = 90.0,
    return_world_coords: bool = False,
    denoise: bool = True,
    align_to_ground: bool = True,
    use_rotation_search: bool = True,
    sensor_name: str = 'rgb',
    debug_vis: bool = False,
) -> Tuple[Dict, Optional[np.ndarray]]:
    """
    通过语义分割图和深度图预测物体的3D bounding box
    
    Args:
        semantic_obs: 语义分割图 (H, W)，每个像素值是object ID
        depth_obs: 深度图 (H, W)，单位米，表示到观测平面的距离
        instance_id: 目标物体的ID
        agent_state: agent状态，如果提供且return_world_coords=True，则返回世界坐标系下的bbox
        hfov: 水平视场角（degrees），默认90度
        return_world_coords: 是否返回世界坐标系下的bbox（需要提供agent_state）
        denoise: 是否进行去噪处理（统计离群点去除+体素下采样）
        align_to_ground: 是否将盒子的一个轴对齐到地面法向（Y轴）
        use_rotation_search: 是否使用旋转搜索找最小面积包围盒（更抗噪但更慢）
        sensor_name: 传感器名称，默认 'rgb'
        debug_vis: 是否显示调试可视化
        
    Returns:
        bbox_3d: 3D bounding box信息字典（检测失败时返回默认bbox），包含：
            - center: (3,) box中心点坐标
            - size: (3,) box尺寸 (width, height, depth)
            - min_bound: (3,) 最小边界点
            - max_bound: (3,) 最大边界点
            - corners: (8, 3) box的8个角点坐标（AABB）
            - obb_center: (3,) OBB中心
            - obb_axes: (3, 3) OBB主轴方向
            - obb_extents: (3,) OBB半长度
            - obb_corners: (8, 3) OBB的8个角点
            - num_points_original: 原始3D点数量
            - num_points_filtered: 过滤后的点数量
        points_3d: (N, 3) 的3D点云（原始点云，用于调试；检测失败时为None）
    """
    # 创建目标物体的mask
    mask = (semantic_obs == instance_id)
    
    # 检查mask是否有效
    if not np.any(mask):
        print(f"Warning: No pixels found for instance_id {instance_id}, returning default bbox")
        return _create_default_bbox_3d(), None
    
    # 将深度图和mask转换为3D点云（相机坐标系）
    points_3d = depth_to_3d_points(depth_obs, mask, hfov)
    
    if len(points_3d) == 0:
        print(f"Warning: No valid 3D points generated for instance_id {instance_id}, returning default bbox")
        return _create_default_bbox_3d(), None
    
    # 计算3D bounding box（相机坐标系），支持去噪和朝向估计
    bbox_3d = compute_3d_bbox_from_points(
        points_3d,
        denoise=denoise,
        align_to_ground=align_to_ground,
        use_rotation_search=use_rotation_search,
        debug_vis=debug_vis
    )
    
    if bbox_3d is None:
        print(f"Warning: Failed to compute 3D bbox for instance_id {instance_id}, returning default bbox")
        return _create_default_bbox_3d(), None
    
    # 如果需要世界坐标系，进行坐标转换
    if return_world_coords and agent_state is not None:
        bbox_3d = transform_bbox_to_world(bbox_3d, agent_state, sensor_name)
    elif return_world_coords and agent_state is None:
        print("Warning: return_world_coords=True but agent_state is None, returning camera coords")
    
    return bbox_3d, points_3d

def compute_3dbox_geometric_confidence(
    bbox_3d: Dict,
    depth_obs: np.ndarray,
    mask_rle: dict,
    hfov: float = 90.0
) -> float:
    """
    基于几何特征计算3D bbox的置信度
    
    考虑因素：
    1. 点云数量：足够的点云才能可靠估计
    2. 点云质量：过滤率（噪点少说明质量高）
    3. 深度一致性：点云深度方差小说明物体表面平整/一致
    
    Args:
        bbox_3d: 3D bbox字典
        depth_obs: 深度图 (H, W)
        mask_rle: RLE格式的mask
        hfov: 水平视场角（degrees）
        
    Returns:
        confidence: [0, 1] 的置信度值
    """
    if bbox_3d is None:
        return 0.0
    
    # 解码mask
    mask_binary = mask_utils.decode(mask_rle)
    
    # 1. 点云数量置信度（使用sigmoid函数）
    num_points = bbox_3d.get('num_points_filtered', 0)
    # 100个点以上认为比较可靠
    point_count_conf = 1.0 / (1.0 + np.exp(-(num_points - 100) / 50))
    
    # 2. 点云质量（过滤率）
    num_original = bbox_3d.get('num_points_original', 1)
    num_filtered = bbox_3d.get('num_points_filtered', 0)
    filter_ratio = num_filtered / max(1, num_original)  # 保留率
    quality_conf = max(0.0, min(1.0, filter_ratio))
    
    # 3. 深度一致性（计算mask区域内深度的标准差）
    mask_depth = depth_obs[mask_binary > 0]
    valid_depth = mask_depth[mask_depth > 0.01]
    
    if len(valid_depth) > 10:
        depth_std = np.std(valid_depth)
        # 标准差小于0.2m认为一致性好
        consistency_conf = 1.0 / (1.0 + depth_std / 0.2)
    else:
        consistency_conf = 0.5
    
    # 4. 综合置信度（加权平均）
    weights = np.array([0.4, 0.3, 0.3])  # 各因素权重
    confidences = np.array([
        point_count_conf,
        quality_conf,
        consistency_conf
    ])
    
    geometric_conf = np.dot(weights, confidences)
    
    return float(np.clip(geometric_conf, 0.0, 1.0))


def predict_3d_bbox_from_mask(
    mask_rle: dict,
    depth_obs: np.ndarray,
    agent_state: Optional[habitat_sim.AgentState] = None,
    hfov: float = 90.0,
    return_world_coords: bool = False,
    denoise: bool = True,
    align_to_ground: bool = True,
    use_rotation_search: bool = True,
    sensor_name: str = 'rgb',
    debug_vis: bool = False,
) -> Dict:
    """
    通过语义分割图和深度图预测物体的3D bounding box
    
    Args:
        mask_rle: 目标物体的mask (H, W)
        depth_obs: 深度图 (H, W)，单位米，表示到观测平面的距离
        agent_state: agent状态，如果提供且return_world_coords=True，则返回世界坐标系下的bbox
        hfov: 水平视场角（degrees），默认90度
        return_world_coords: 是否返回世界坐标系下的bbox（需要提供agent_state）
        denoise: 是否进行去噪处理（统计离群点去除+体素下采样）
        align_to_ground: 是否将盒子的一个轴对齐到地面法向（Y轴）
        use_rotation_search: 是否使用旋转搜索找最小面积包围盒（更抗噪但更慢）
        sensor_name: 传感器名称，默认 'rgb'
        debug_vis: 是否显示调试可视化
        
    Returns:
        bbox_3d: 3D bounding box信息字典（检测失败时返回默认bbox），包含：
            - center: (3,) box中心点坐标
            - size: (3,) box尺寸 (width, height, depth)
            - min_bound: (3,) 最小边界点
            - max_bound: (3,) 最大边界点
            - corners: (8, 3) box的8个角点坐标（AABB）
            - obb_center: (3,) OBB中心
            - obb_axes: (3, 3) OBB主轴方向
            - obb_extents: (3,) OBB半长度
            - obb_corners: (8, 3) OBB的8个角点
            - num_points_original: 原始3D点数量
            - num_points_filtered: 过滤后的点数量
    """
    # 解码RLE格式的mask
    mask = mask_utils.decode(mask_rle)
    
    # 检查mask是否有效
    if not np.any(mask):
        print(f"Warning: No pixels found for mask, returning default bbox")
        return _create_default_bbox_3d()
    
    # 将深度图和mask转换为3D点云（相机坐标系）
    points_3d = depth_to_3d_points(depth_obs, mask, hfov)
    
    if len(points_3d) == 0:
        print(f"Warning: No valid 3D points generated for mask, returning default bbox")
        return _create_default_bbox_3d()
    
    # 计算3D bounding box（相机坐标系），支持去噪和朝向估计
    bbox_3d = compute_3d_bbox_from_points(
        points_3d,
        denoise=denoise,
        align_to_ground=align_to_ground,
        use_rotation_search=use_rotation_search,
        debug_vis=debug_vis
    )
    
    if bbox_3d is None:
        print(f"Warning: Failed to compute 3D bbox for mask, returning default bbox")
        return _create_default_bbox_3d()
    
    # 如果需要世界坐标系，进行坐标转换
    if return_world_coords and agent_state is not None:
        bbox_3d = transform_bbox_to_world(bbox_3d, agent_state, sensor_name)
    elif return_world_coords and agent_state is None:
        print("Warning: return_world_coords=True but agent_state is None, returning camera coords")
    
    # 计算几何置信度
    geometric_confidence = compute_3dbox_geometric_confidence(
        bbox_3d=bbox_3d,
        depth_obs=depth_obs,
        mask_rle=mask_rle,
        hfov=hfov
    )
    
    # 将置信度添加到返回的字典中
    bbox_3d['geometric_confidence'] = geometric_confidence
    
    return bbox_3d

def _compute_polygon_area(corners_2d: np.ndarray) -> float:
    """
    使用Shapely计算2D多边形的面积
    
    Args:
        corners_2d: (N, 2) 多边形的角点
        
    Returns:
        area: 多边形面积
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        poly = Polygon(corners_2d)
        
        # 验证并修正多边形（确保是有效的）
        if not poly.is_valid:
            poly = make_valid(poly)
        
        return poly.area
    
    except ImportError:
        raise ImportError(
            "需要安装 shapely 库来计算多边形面积。"
            "请运行: pip install shapely"
        )


def _compute_rotated_rect_intersection_area(rect1_corners: np.ndarray, rect2_corners: np.ndarray) -> float:
    """
    计算两个2D旋转矩形的交集面积（使用Shapely库）
    
    Args:
        rect1_corners: (4, 2) 第一个矩形的4个角点（2D）
        rect2_corners: (4, 2) 第二个矩形的4个角点（2D）
        
    Returns:
        intersection_area: 交集面积
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        # 创建多边形
        poly1 = Polygon(rect1_corners)
        poly2 = Polygon(rect2_corners)
        
        # 验证并修正多边形（确保是有效的）
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)
        
        # 计算交集
        intersection = poly1.intersection(poly2)
        
        return intersection.area
    
    except ImportError:
        raise ImportError(
            "需要安装 shapely 库来计算旋转矩形的交集。"
            "请运行: pip install shapely"
        )


def compute_bbox_iou_3d(bbox1: Dict, bbox2: Dict, use_obb: bool = True) -> float:
    """
    计算两个3D bounding box的IoU
    
    支持两种模式：
    1. AABB模式（use_obb=False）: 计算axis-aligned bounding box的IoU
    2. OBB模式（use_obb=True）: 计算水平对齐的oriented bounding box的IoU
       （假设boxes在Y轴方向对齐，只在XZ平面上有旋转）
    
    Args:
        bbox1: 第一个bbox字典，需要包含：
            - AABB模式: 'min_bound', 'max_bound', 'size'
            - OBB模式: 'obb_center', 'obb_axes', 'obb_extents', 'obb_corners'
        bbox2: 第二个bbox字典
        use_obb: 是否使用OBB模式计算IoU
        
    Returns:
        iou: 3D IoU值
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    if not use_obb:
        # AABB模式：原有的快速计算方法
        # 获取两个box的边界
        min1, max1 = bbox1['min_bound'], bbox1['max_bound']
        min2, max2 = bbox2['min_bound'], bbox2['max_bound']
        
        # 计算交集的边界
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        # 检查是否有交集
        if np.any(inter_min >= inter_max):
            return 0.0
        
        # 计算交集体积
        inter_size = inter_max - inter_min
        inter_volume = np.prod(inter_size)
        
        # 计算各自的体积
        volume1 = np.prod(bbox1['size'])
        volume2 = np.prod(bbox2['size'])
        
        # 计算并集体积
        union_volume = volume1 + volume2 - inter_volume
        
        # 计算IoU
        iou = inter_volume / union_volume if union_volume > 0 else 0.0
        
        return iou
    
    else:
        # OBB模式：计算水平对齐的OBB的IoU
        # 检查必要的字段
        required_fields = ['obb_center', 'obb_axes', 'obb_extents', 'obb_corners']
        for field in required_fields:
            if field not in bbox1 or field not in bbox2:
                raise ValueError(f"OBB模式需要bbox包含字段: {required_fields}")
        
        center1 = bbox1['obb_center']
        center2 = np.array(bbox2['obb_center'])
        extents1 = bbox1['obb_extents']  # (3,) 半长度
        extents2 = np.array(bbox2['obb_extents'])
        corners1 = bbox1['obb_corners']  # (8, 3)
        corners2 = np.array(bbox2['obb_corners'])
        
        # 1. 计算Y方向（垂直）的交集
        # 获取每个OBB在Y方向的范围
        y_min1 = np.min(corners1[:, 1])
        y_max1 = np.max(corners1[:, 1])
        y_min2 = np.min(corners2[:, 1])
        y_max2 = np.max(corners2[:, 1])
        
        # 计算Y方向的交集
        y_inter_min = max(y_min1, y_min2)
        y_inter_max = min(y_max1, y_max2)
        
        if y_inter_min >= y_inter_max:
            return 0.0  # Y方向没有交集
        
        y_inter_height = y_inter_max - y_inter_min
        
        # 2. 在XZ平面（水平面）上计算交集面积
        # 提取XZ平面上的4个角点（底面或顶面）
        # 假设corners的顺序：前4个是底面，后4个是顶面
        # 我们使用底面的4个角点投影到XZ平面
        
        # 获取底面4个角点（索引0，1，4，5），提取XZ坐标
        rect1_corners_xz = corners1[[0, 1, 4, 5], :][:, [0, 2]]  # (4, 2) - X和Z坐标
        rect2_corners_xz = corners2[[0, 1, 4, 5], :][:, [0, 2]]
        
        # 确保角点顺序形成有效多边形（逆时针或顺时针）
        # 通过计算中心点，按角度排序
        def sort_corners_by_angle(corners_2d):
            center = np.mean(corners_2d, axis=0)
            angles = np.arctan2(corners_2d[:, 1] - center[1], 
                               corners_2d[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            return corners_2d[sorted_indices]
        
        rect1_corners_xz = sort_corners_by_angle(rect1_corners_xz)
        rect2_corners_xz = sort_corners_by_angle(rect2_corners_xz)
        
        is_debug = False
        if is_debug:
            # 计算2D旋转矩形的交集面积
            # 可视化XZ平面上的两个矩形
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            
            # 绘制rect1 (红色)
            rect1_plot = np.vstack([rect1_corners_xz, rect1_corners_xz[0]])  # 闭合多边形
            plt.plot(rect1_plot[:, 0], rect1_plot[:, 1], 'r-', linewidth=2, label='BBox1')
            plt.scatter(rect1_corners_xz[:, 0], rect1_corners_xz[:, 1], c='red', s=100, zorder=5)
            
            # 绘制rect2 (蓝色)
            rect2_plot = np.vstack([rect2_corners_xz, rect2_corners_xz[0]])  # 闭合多边形
            plt.plot(rect2_plot[:, 0], rect2_plot[:, 1], 'b-', linewidth=2, label='BBox2')
            plt.scatter(rect2_corners_xz[:, 0], rect2_corners_xz[:, 1], c='blue', s=100, zorder=5)
            
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.title('XZ Plane Projection of OBBs')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            # 保存可视化结果
            plt.savefig('/data1/tct_data/xz_projection.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"XZ平面投影可视化已保存到: /data1/tct_data/xz_projection.png")
        
        intersection_area_xz = _compute_rotated_rect_intersection_area(
            rect1_corners_xz, rect2_corners_xz
        )
        
        #NOTE:Debug
        if intersection_area_xz <= 0:
            return 0.0  # XZ平面没有交集
        
        # 3. 计算3D交集体积
        inter_volume = intersection_area_xz * y_inter_height
        
        # 4. 计算各自的体积（使用投影面积 × 高度）
        # 计算box1的底面投影面积
        area1_xz = _compute_polygon_area(rect1_corners_xz)
        height1 = y_max1 - y_min1
        volume1 = area1_xz * height1
        
        # 计算box2的底面投影面积
        area2_xz = _compute_polygon_area(rect2_corners_xz)
        height2 = y_max2 - y_min2
        volume2 = area2_xz * height2
        
        # 5. 计算并集体积
        union_volume = volume1 + volume2 - inter_volume
        
        # 6. 计算IoU
        iou = inter_volume / union_volume if union_volume > 0 else 0.0
        
        return iou


def transform_object_aabb_to_camera_obb(
    obj: 'habitat_sim.physics.ManagedRigidObject',
    agent_state: 'habitat_sim.AgentState',
    sensor_name: str = 'rgb'
) -> Dict:
    """
    将物体坐标系下的AABB转换到相机坐标系下的OBB
    
    Args:
        obj: Habitat-sim的刚体物体对象
        agent_state: Agent状态对象
        sensor_name: 传感器名称，默认 'rgb'
        
    Returns:
        bbox_dict: 包含OBB信息的字典（相机坐标系）
            - obb_center: (3,) OBB中心
            - obb_axes: (3, 3) OBB主轴方向
            - obb_extents: (3,) OBB半长度
            - obb_corners: (8, 3) OBB的8个角点
            - corners: (8, 3) AABB的8个角点（为了兼容visualize函数）
    """
    # 1. 获取物体局部坐标系下的AABB
    aabb = obj.collision_shape_aabb
    local_min = np.array([aabb.min.x, aabb.min.y, aabb.min.z])
    local_max = np.array([aabb.max.x, aabb.max.y, aabb.max.z])
    
    # 2. 计算AABB的8个角点（物体局部坐标系）
    local_corners = np.array([
        [local_min[0], local_min[1], local_min[2]],
        [local_max[0], local_min[1], local_min[2]],
        [local_max[0], local_max[1], local_min[2]],
        [local_min[0], local_max[1], local_min[2]],
        [local_min[0], local_min[1], local_max[2]],
        [local_max[0], local_min[1], local_max[2]],
        [local_max[0], local_max[1], local_max[2]],
        [local_min[0], local_max[1], local_max[2]],
    ])
    
    # 3. 获取物体的世界变换矩阵
    obj_transform = obj.transformation  # 4x4 Matrix
    
    # 4. 将局部角点转换到世界坐标系
    world_corners = np.zeros_like(local_corners)
    for i, corner in enumerate(local_corners):
        corner_vec = mn.Vector3(*corner)
        world_corner = obj_transform.transform_point(corner_vec)
        world_corners[i] = [world_corner.x, world_corner.y, world_corner.z]
    
    # 5. 获取相机的世界变换
    if not hasattr(agent_state, 'sensor_states') or sensor_name not in agent_state.sensor_states:
        raise ValueError(
            f"sensor_states 不可用或传感器 '{sensor_name}' 不存在。"
            f"可用的传感器: {list(agent_state.sensor_states.keys()) if hasattr(agent_state, 'sensor_states') else 'None'}"
        )
    
    sensor_state = agent_state.sensor_states[sensor_name]
    camera_position = sensor_state.position
    camera_quaternion = sensor_state.rotation
    
    # 6. 使用 quaternion 库转换相机旋转
    try:
        import quaternion
        
        if hasattr(camera_quaternion, 'w'):
            quat = np.quaternion(
                camera_quaternion.w,
                camera_quaternion.x, 
                camera_quaternion.y,
                camera_quaternion.z
            )
        else:
            quat = camera_quaternion
        
        rotation_matrix = quaternion.as_rotation_matrix(quat)
        
    except ImportError:
        raise ImportError(
            "需要安装 numpy-quaternion 库来进行坐标转换。"
            "请运行: pip install numpy-quaternion"
        )
    
    # 7. 构建世界到相机的变换矩阵（相机坐标系 <- 世界坐标系）
    T_camera_world = np.eye(4)
    T_camera_world[0:3, 0:3] = rotation_matrix.T  # 转置 = 逆变换（旋转矩阵是正交矩阵）
    T_camera_world[0:3, 3] = -rotation_matrix.T @ camera_position
    
    # 8. 将世界坐标系的角点转换到相机坐标系
    camera_corners = np.zeros_like(world_corners)
    for i, corner in enumerate(world_corners):
        corner_homogeneous = np.append(corner, 1.0)
        camera_corner_homogeneous = T_camera_world @ corner_homogeneous
        camera_corners[i] = camera_corner_homogeneous[0:3]
    
    # 9. 计算OBB的中心（相机坐标系）
    obb_center = np.mean(camera_corners, axis=0)
    
    # 10. 提取OBB的主轴方向和extents
    # 物体的AABB在其局部坐标系中，转换后在相机坐标系中变成OBB
    # 我们需要提取物体坐标系的三个轴在相机坐标系中的方向
    
    # 获取物体坐标系的三个轴方向（世界坐标系）
    obj_x_axis = obj_transform.transform_vector(mn.Vector3(1, 0, 0))
    obj_y_axis = obj_transform.transform_vector(mn.Vector3(0, 1, 0))
    obj_z_axis = obj_transform.transform_vector(mn.Vector3(0, 0, 1))
    
    # 转换到相机坐标系
    obj_x_axis_cam = rotation_matrix.T @ np.array([obj_x_axis.x, obj_x_axis.y, obj_x_axis.z])
    obj_y_axis_cam = rotation_matrix.T @ np.array([obj_y_axis.x, obj_y_axis.y, obj_y_axis.z])
    obj_z_axis_cam = rotation_matrix.T @ np.array([obj_z_axis.x, obj_z_axis.y, obj_z_axis.z])
    
    # 归一化
    obj_x_axis_cam = obj_x_axis_cam / np.linalg.norm(obj_x_axis_cam)
    obj_y_axis_cam = obj_y_axis_cam / np.linalg.norm(obj_y_axis_cam)
    obj_z_axis_cam = obj_z_axis_cam / np.linalg.norm(obj_z_axis_cam)
    
    # OBB的主轴（列向量）
    obb_axes = np.column_stack([obj_x_axis_cam, obj_y_axis_cam, obj_z_axis_cam])
    
    # OBB的半长度（extents）
    local_size = local_max - local_min
    obb_extents = local_size / 2.0
    
    bbox_dict = {
        'obb_center': obb_center,
        'obb_axes': obb_axes,
        'obb_extents': obb_extents,
        'obb_corners': camera_corners,
        'corners': camera_corners,  # 为了兼容visualize函数
        'center': obb_center,
        'size': local_size,
        'min_bound': np.min(camera_corners, axis=0),
        'max_bound': np.max(camera_corners, axis=0),
    }
    
    return bbox_dict


def visualize_bbox_on_image(
    image: np.ndarray,
    bbox_3d: Dict,
    hfov: float = 90.0,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    use_obb: bool = True
) -> np.ndarray:
    """
    将3D bbox投影到2D图像上进行可视化
    
    注意：此函数假设bbox_3d是在相机坐标系下的。
    如果您有世界坐标系下的bbox，请先使用相应的转换函数转回相机坐标系。
    
    Args:
        image: RGB图像 (H, W, 3)
        bbox_3d: 3D bounding box字典（必须是相机坐标系）
        hfov: 水平视场角（degrees）
        color: 边框颜色 (R, G, B)
        thickness: 线条粗细
        use_obb: 是否使用OBB（有方向的包围盒）而不是AABB（轴对齐包围盒）
        
    Returns:
        image_with_bbox: 绘制了bbox的图像
    """
    import cv2
    
    if bbox_3d is None:
        return image
    
    image_out = image.copy()
    H, W = image.shape[:2]
    
    # 计算相机内参
    focal_length = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    
    # 获取8个角点（相机坐标系）
    if use_obb and 'obb_corners' in bbox_3d:
        corners = bbox_3d['obb_corners']
    else:
        corners = bbox_3d['corners']
    
    # 投影到2D（参考官方 view_transform.py）
    # 注意：点云坐标系中，z为负数表示在相机前方（相机朝向-Z方向）
    # 保留所有8个角点的投影坐标，即使在视野外也不过滤
    # 这样可以正确处理部分在视野外的3D box
    projected_points = []
    valid_points = []  # 标记每个点是否有效（在相机前面）
    
    for corner in corners:
        x, y, z = corner
        if z >= 0:  # 在相机后面或相机平面上（z为负数表示在前方）
            projected_points.append(None)
            valid_points.append(False)
        else:
            # 投影公式：除以负的深度（因为 z 是负数）
            u = x * focal_length / (-z) + cx
            v = -y * focal_length / (-z) + cy
            # 不再过滤超出图像边界的点，让cv2.line自动裁剪
            projected_points.append((int(u), int(v)))
            valid_points.append(True)
    
    # 定义bbox的12条边（连接8个角点）
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 前面4条边
        (4, 5), (5, 6), (6, 7), (7, 4),  # 后面4条边
        (0, 4), (1, 5), (2, 6), (3, 7),  # 连接前后的4条边
    ]
    
    # 绘制bbox的边
    # 只绘制两个端点都有效（都在相机前面）的边
    for edge in edges:
        idx1, idx2 = edge
        if valid_points[idx1] and valid_points[idx2]:
            pt1 = projected_points[idx1]
            pt2 = projected_points[idx2]
            cv2.line(image_out, pt1, pt2, color, thickness)
    
    return image_out

