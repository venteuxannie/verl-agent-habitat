"""
模型加载与推理相关函数
支持两种模式：
1. API 模式：通过 vLLM OpenAI 兼容 API 调用
2. Local 模式：直接加载 HuggingFace 模型
"""
import base64
import io
import torch
from PIL import Image
from typing import List, Optional, Union
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)

# 尝试导入 Qwen3-VL（可能需要较新版本的 transformers）
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False

# =============================================================================
# 图像编码工具
# =============================================================================

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """将 PIL Image 编码为 base64 字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# =============================================================================
# VLM API 客户端 (vLLM OpenAI 兼容接口)
# =============================================================================

class VLMClient:
    """
    VLM API 客户端，支持 vLLM 和 OpenAI 兼容的 API
    
    用法示例:
        client = VLMClient(
            base_url="http://localhost:8045/v1",
            model_name="InternVL2-8B"
        )
        response = client.inference(image, prompt)
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8045/v1",
        api_key: str = "empty",
        model_name: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        try:
            from openai import OpenAI
            import httpx
        except ImportError:
            raise ImportError("请安装 openai 和 httpx: pip install openai httpx")
        
        # 创建带超时设置的客户端
        self.client = OpenAI(
            base_url=base_url, 
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=30.0),  # 总超时和连接超时
            max_retries=max_retries  # 自动重试次数
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        print(f"VLMClient initialized: base_url={base_url}, model={model_name}, timeout={timeout}s, max_retries={max_retries}")
    
    def inference(self, image: Image.Image, prompt: str) -> List[str]:
        """
        通过 OpenAI 兼容 API 进行 VLM 推理
        
        Args:
            image: PIL Image 对象
            prompt: 文本提示，可包含 <image> 占位符
            
        Returns:
            List[str]: 模型响应列表
        """
        import time
        
        # 将图像编码为 base64
        base64_image = encode_image_to_base64(image)
        
        # 移除 prompt 中的 <image> 标记（API 模式不需要）
        clean_prompt = prompt.replace("<image>", "").strip()
        # 同时处理 Qwen 格式的图像标记
        clean_prompt = clean_prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", "").strip()
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": clean_prompt
                    }
                ]
            }
        ]
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                result = response.choices[0].message.content
                return [result] if isinstance(result, str) else result
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                print(f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): [{error_type}] {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries:
                    wait_time = min(2 ** attempt, 10)  # 指数退避，最多等待10秒
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        print(f"API 调用最终失败，已重试 {self.max_retries} 次: {last_error}")
        # raise RuntimeError(f"API 调用最终失败，已重试 {self.max_retries} 次: {last_error}")
        return [""]


# =============================================================================
# InternVL 图像预处理 (Local 模式)
# =============================================================================

def load_internvl_image(image: Image.Image, config, input_size=448, max_num=12):
    """Load and preprocess image for InternVL models using dynamic preprocessing."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        
        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    # Use the config to get the image size if available
    if hasattr(config, 'force_image_size') and config.force_image_size:
        input_size = config.force_image_size
    elif hasattr(config, 'vision_config') and hasattr(config.vision_config, 'image_size'):
        input_size = config.vision_config.image_size
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values.to(torch.bfloat16)


# =============================================================================
# 模型加载函数 (Local 模式)
# =============================================================================

def load_model_and_processor(model_path: str):
    """Loads the VL model (Qwen-VL or InternVL) and its associated processor."""
    from transformers import AutoModel, AutoConfig
    
    # Detect model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type.lower()
    
    if "internvl" in model_type:
        print("Loading InternVL model from", model_path)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        # InternVL uses tokenizer, not processor in the standard way
        # But we'll use the custom processor from verl if available
        try:
            from verl.utils.tokenizer import hf_processor
            processor = hf_processor(model_path, trust_remote_code=True)
            if processor is None:
                # Fallback to tokenizer
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        except ImportError:
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    elif "qwen3" in model_type:
        print("Loading Qwen3-VL model from", model_path)
        if QWEN3_VL_AVAILABLE:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        else:
            # 回退到 AutoModel（需要 trust_remote_code）
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2",
                device_map="auto",
                trust_remote_code=True
            )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif "qwen2.5" in model_type or "qwen2_5" in model_type:
        print("Loading Qwen2.5-VL model from", model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    elif "qwen2" in model_type:
        print("Loading Qwen2-VL model from", model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please provide a valid Qwen2, Qwen2.5, Qwen3, or InternVL model path.")
    
    return model, processor, model_type


# =============================================================================
# 推理函数 (Local 模式)
# =============================================================================

def inference(model, processor, image: Image.Image, prompt: str, model_type: str, max_new_tokens: int = 1024) -> List[str]:
    """Performs inference with the model given an image and a text prompt."""
    if "internvl" in model_type:
        # InternVL-specific inference
        # Check if processor has apply_chat_template method
        if hasattr(processor, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # Fallback for tokenizer-only case
            text_prompt = prompt
        
        # Replace <image> tag with <IMG_CONTEXT> for older InternVL versions
        if "<image>" in text_prompt and not hasattr(model, 'chat'):
            text_prompt = text_prompt.replace("<image>", "<IMG_CONTEXT>")
        
        # Use InternVL's chat method if available
        if hasattr(model, 'chat'):
            # Load image using InternVL's expected format
            from transformers import AutoTokenizer
            if not isinstance(processor, AutoTokenizer):
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, use_fast=False)
            else:
                tokenizer = processor
            
            # For InternVL, we need to preprocess the image properly
            # Using dynamic_preprocess if available
            pixel_values = load_internvl_image(image, model.config)
            if hasattr(model, 'device'):
                pixel_values = pixel_values.to(model.device)
            else:
                pixel_values = pixel_values.cuda()
            
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return [response] if isinstance(response, str) else response
        else:
            # Use standard generate method with processor
            if hasattr(processor, '__call__'):
                inputs = processor(
                    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
                ).to(model.device)
            else:
                # Tokenizer-only case
                inputs = processor(
                    text_prompt, return_tensors="pt"
                ).to(model.device)
                # Need to add pixel_values manually
                pixel_values = load_internvl_image(image, model.config)
                if hasattr(model, 'device'):
                    pixel_values = pixel_values.to(model.device)
                else:
                    pixel_values = pixel_values.cuda()
                inputs['pixel_values'] = pixel_values
            
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return output_text if output_text else [""]
    else:
        # Qwen-VL (Qwen2/Qwen2.5/Qwen3) inference
        messages = [{"role": "user", "content": prompt}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)
        
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return output_text if output_text else [""]


# =============================================================================
# 统一接口：创建 VLM 客户端
# =============================================================================

def create_vlm_client(
    backend: str = "api",
    model_path: str = None,
    api_url: str = "http://localhost:8045/v1",
    api_key: str = "empty",
    model_name: str = None
) -> Union[VLMClient, tuple]:
    """
    创建 VLM 客户端的统一接口
    
    Args:
        backend: "api" 使用 vLLM API，"local" 直接加载模型
        model_path: 模型路径（local 模式必需）
        api_url: API 服务地址（api 模式使用）
        api_key: API 密钥
        model_name: 模型名称（api 模式使用）
        
    Returns:
        api 模式: VLMClient 实例
        local 模式: (model, processor, model_type) 元组
    """
    if backend == "api":
        if not model_name:
            raise ValueError("api 模式需要指定 model_name")
        return VLMClient(
            base_url=api_url,
            api_key=api_key,
            model_name=model_name
        )
    elif backend == "local":
        if not model_path:
            raise ValueError("local 模式需要指定 model_path")
        return load_model_and_processor(model_path)
    else:
        raise ValueError(f"不支持的 backend: {backend}，请使用 'api' 或 'local'")


def unified_inference(
    client_or_model,
    processor,
    model_type: str,
    image: Image.Image,
    prompt: str,
    backend: str = "api"
) -> List[str]:
    """
    统一的推理接口
    
    Args:
        client_or_model: VLMClient 实例或 model
        processor: tokenizer/processor（local 模式）
        model_type: 模型类型（local 模式）
        image: PIL Image 对象
        prompt: 文本提示
        backend: "api" 或 "local"
        
    Returns:
        List[str]: 模型响应列表
    """
    if backend == "api":
        return client_or_model.inference(image, prompt)
    else:
        return inference(client_or_model, processor, image, prompt, model_type)
