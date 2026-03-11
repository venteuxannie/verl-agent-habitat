import base64
from openai import OpenAI
from PIL import Image
import io

# 初始化 vLLM OpenAI 客户端
client = OpenAI(api_key='', base_url='http://0.0.0.0:8010/v1')
model_name = client.models.list().data[0].id

# 使用与 eval_notebook.py 相同的提示模板格式
# HABITAT_VISUAL_GROUNDING_COT_TEMPLATE 的内容
task_caption = "table near sofa"  # 示例目标物体
conf_score = 0.532      # 示例观察分数

# 构建提示词 (与 eval_notebook.py 中 build_text_obs 函数的逻辑一致)
# prompt_template = """<image>You are an agent in an indoor environment, and you will see the current RGB observation image. The target object is "{task_caption}. " Your goal is to improve the observation score of the target object (score range 0~1) by controlling your own movement. The current observation score is "{conf_score:.3f}". You can choose between ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'stop']. You can only select 1 action at a time. Your response should be a valid json file in the following format:
# {{
# "thoughts": "{{First, describe the target object, such as its position and occlusion. Then, check the current observation score. Finally, think about what action to take. }}", 
# "action": "{{action}}" 
# }}"""

# prompt_template = """<image>Supported task types: Visual Grounding, Segment, 3D Box Prediction
# Task instruction: I want the mask of the apple on the table.

# Please determine which task this instruction belongs to."""
prompt_template = """<image>Supported task types: Visual Grounding, Segment, 3D Box Prediction
Task instruction: I want to get the 3D box of the apple on the table.

Please determine which task this instruction belongs to."""
# prompt_template = """<image>Supported task types: Visual Grounding, Segment, 3D Box Prediction
# Task instruction: I want the mask of the apple on the table.

# Please extract the location description of the target object that is unrelated to the task."""

prompt_text = prompt_template.format(task_caption=task_caption, conf_score=conf_score)

# 对于 Qwen 模型，需要将 <image> 替换为特殊标记
# 这里保持 <image> 标签，vLLM 会自动处理
# 如果需要手动替换，使用: prompt_text.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')

# 将本地图像转换为 base64 编码的 data URL
def image_to_data_url(image_path):
    """将本地图像转换为 base64 编码的 data URL"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{base64_str}"

# 本地图像路径
local_image_path = '/data1/tct_data/habitat/sft_data/vg_replica_CoT_10/000000.png'
image_url = image_to_data_url(local_image_path)

# 发起推理请求
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {'url': image_url},
            },
            {
                'type': 'text',
                'text': prompt_text.replace('<image>', ''),  # 移除 <image> 标签，因为图像已单独提供
            },
        ],
    }],
    temperature=0.0,
    max_tokens=512  # 设置最大生成token数（可根据需要调整：512, 1024, 2048等）
)

print("=" * 80)
print("Task Caption:", task_caption)
print("Confidence Score:", conf_score)
print("=" * 80)
print("Model Response:")
print(response.choices[0].message.content)
print("=" * 80)