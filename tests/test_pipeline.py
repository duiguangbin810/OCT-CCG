import os
import torch
from diffusers import StableDiffusionXLPipeline

def test_run():
    # 优化1：使用低内存模式加载，移除不推荐的参数
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.float32,  # CPU推荐使用float32
        use_safetensors=True,
        low_cpu_mem_usage=True  # 启用CPU内存优化
    )
    
    # 优化2：正确禁用安全检查（不在from_pretrained中传参）
    pipe.safety_checker = None
    
    # 优化3：明确指定CPU设备
    pipe = pipe.to("cpu")
    
    # 优化4：添加生成参数控制，减少内存使用
    prompt = "a cute cat"
    image = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=20,  # 减少步数加快速度
        guidance_scale=7.5,
        num_images_per_prompt=1  # 一次只生成1张
    ).images[0]

    output_dir = "d:/projects/OCT-CCG/outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "test_cpu.png")
    image.save(save_path)
    print(f"图像已保存至：{save_path}")

if __name__ == "__main__":
    try:
        test_run()
    except Exception as e:
        print(f"运行出错：{str(e)}")
