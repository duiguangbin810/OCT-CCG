import os
import torch
from diffusers import StableDiffusionXLPipeline
from oct_ccg.pipeline.oct_ccg_sdxl_pipeline import OCTCCGSDPipeline
from oct_ccg.attention.attn_hook import get_slot_heatmaps_resized
from oct_ccg.counting.soft_count_gumbel_sinkhorn import soft_count_from_final_gs

def test_oct_ccg_minimal():
    """
    最小可用的OCT-CCG测试脚本
    """
    print("开始测试OCT-CCG管道...")
    
    # 1. 检查设备
    if torch.cuda.is_available():
        device = "cuda"
        print(f"检测到GPU，使用设备: {device}")
    else:
        device = "cpu"
        print(f"未检测到GPU，使用设备: {device}")
    
    # 2. 加载模型
    print("加载模型...")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # 默认使用GPU模式
    pipe = OCTCCGSDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # 默认使用float16以节省GPU内存
        variant="fp16",
        use_safetensors=True,
        safety_checker=None
    )
    
    # 将模型移动到指定设备
    pipe = pipe.to(device)
    
    # 3. 设置测试参数
    prompt = "2x cat; on green grass"
    class_counts = {"cat": 2}  # 目标：生成2只猫
    
    # 4. 生成图像
    print("生成图像...")
    output_dir = "d:/projects/OCT-CCG/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用最小参数集
    output = pipe(
        prompt=prompt,
        class_targets=class_counts,
        get_slot_heatmaps_resized_fn=get_slot_heatmaps_resized,
        soft_count_from_final_gs_fn=soft_count_from_final_gs,
        num_inference_steps=10,  # 减少步数以加快测试
        guidance_scale=7.5,
        height=512,  # 减小尺寸以加快测试
        width=512,
        use_oct_embeddings=True,  # 启用OCT功能
        cnt_weight=1.0,  # 计数权重
        orth_weight=0.5,  # 正交权重
        cross_overlap_weight=0.5,  # 跨类重叠权重
        eta_dual=0.2,  # 对偶变量更新率
        clamp_lambda=10.0,  # 对偶变量范围
        sinkhorn_kwargs={
            "temp": 0.05,
            "theta": 0.01,
            "gamma": 1e-2,
            "agg_mode": "sum",
            "cross_overlap_weight": 0.5
        }
    )
    
    # 5. 保存图像
    image = output.images[0]
    save_path = os.path.join(output_dir, "test_oct_ccg_minimal.png")
    image.save(save_path)
    print(f"图像已保存至：{save_path}")
    
    print("测试完成！")

if __name__ == "__main__":
    try:
        test_oct_ccg_minimal()
    except Exception as e:
        print(f"运行出错：{str(e)}")
        import traceback
        traceback.print_exc()