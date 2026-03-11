import os
import psutil
import pynvml
from agent_system.environments.env_package.habitat_sim.utils.habitat_utils import create_hm3d_simulator

# 10个测试场景（从 train 文件夹实际内容）
HM3D_TRAINING_SCENES = [
    '00055-HxmXPBbFCkH',
    '00057-1UnKg1rAb8A',
    '00207-FRQ75PjD278',
    '00234-nACV8wLu1u5',
    '00307-vDfkYo5VqEQ',
    '00440-wPLokgvCnuk',
    '00466-xAHnY3QzFUN',
    '00537-oahi4u45xMf',
    '00680-YmWinf3mhb5',
    '00733-GtM3JtRvvvR',
]

# 初始化 NVML
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 使用第0号GPU

def get_gpu_memory_mb():
    """获取GPU显存使用量（MB）"""
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return mem_info.used / 1024 / 1024

def get_ram_usage_mb():
    """获取当前进程RAM使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 记录初始资源占用
initial_gpu_mem = get_gpu_memory_mb()
initial_ram = get_ram_usage_mb()
print(f"Initial GPU Memory: {initial_gpu_mem:.2f} MB")
print(f"Initial RAM Usage: {initial_ram:.2f} MB")
print("=" * 70)

# 存储结果
results = []

for scene_name in HM3D_TRAINING_SCENES:
    scene_code = scene_name.split('-')[1]
    scene_id = f"/data/tct/habitat/data/hm3d/train/{scene_name}/{scene_code}.basis.glb"
    
    # 加载前的显存
    gpu_before = get_gpu_memory_mb()
    ram_before = get_ram_usage_mb()
    
    print(f"\nLoading scene: {scene_name}")
    sim = create_hm3d_simulator("HM3D", scene_id)
    
    # 加载后的显存
    gpu_after = get_gpu_memory_mb()
    ram_after = get_ram_usage_mb()
    
    gpu_used = gpu_after - gpu_before
    ram_used = ram_after - ram_before
    
    results.append({
        'scene': scene_name,
        'gpu_used': gpu_used,
        'ram_used': ram_used,
        'gpu_total': gpu_after - initial_gpu_mem,
        'ram_total': ram_after - initial_ram
    })
    
    print(f"  GPU: +{gpu_used:.2f} MB (total: {gpu_after - initial_gpu_mem:.2f} MB)")
    print(f"  RAM: +{ram_used:.2f} MB (total: {ram_after - initial_ram:.2f} MB)")
    
    # 关闭simulator释放资源
    sim.close()

# 汇总报告
print("\n" + "=" * 70)
print("Summary Report")
print("=" * 70)
print(f"{'Scene':<25} {'GPU Used (MB)':<15} {'RAM Used (MB)':<15}")
print("-" * 70)

total_gpu = 0
total_ram = 0
for r in results:
    print(f"{r['scene']:<25} {r['gpu_used']:>12.2f}   {r['ram_used']:>12.2f}")
    total_gpu += r['gpu_used']
    total_ram += r['ram_used']

print("-" * 70)
print(f"{'Average':<25} {total_gpu/len(results):>12.2f}   {total_ram/len(results):>12.2f}")
print(f"{'Total (累计)':<25} {total_gpu:>12.2f}   {total_ram:>12.2f}")

# 最终显存状态
final_gpu = get_gpu_memory_mb()
final_ram = get_ram_usage_mb()
print(f"\nFinal GPU Memory: {final_gpu:.2f} MB (delta: {final_gpu - initial_gpu_mem:.2f} MB)")
print(f"Final RAM Usage: {final_ram:.2f} MB (delta: {final_ram - initial_ram:.2f} MB)")

# 清理 NVML
pynvml.nvmlShutdown()