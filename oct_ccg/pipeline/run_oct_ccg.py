import os
import json
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

from oct_ccg.pipeline.oct_ccg_sdxl_pipeline import OCTCCGSDPipeline
from oct_ccg.attention.attn_hook import get_slot_heatmaps_resized
from oct_ccg.counting.soft_count_gumbel_sinkhorn import soft_count_from_final_gs


def read_yaml(file_path: str) -> dict:
    """读取YAML配置文件"""
    with open(file_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


def set_seed(seed: int):
    """
    设置随机种子以确保结果可复现
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def load_dataset(dataset_path: str) -> List[Dict]:
    """加载JSON格式的数据集"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def parse_class_counts_from_prompt(prompt: str) -> Dict[str, int]:
    """
    从prompt中解析类别和数量，格式为 "Nx class1, Nx class2; 场景"
    例如: "2x cat, 1x dog; on green grass" -> {"cat": 2, "dog": 1}
    """
    class_counts = {}
    
    # 分离类别部分和场景部分
    parts = prompt.split(";")
    class_part = parts[0].strip()
    
    # 解析类别部分
    class_items = class_part.split(",")
    for item in class_items:
        item = item.strip()
        # 格式应为 "Nx class"
        if "x" in item:
            count_str, class_name = item.split("x", 1)
            try:
                count = int(count_str.strip())
                class_name = class_name.strip()
                class_counts[class_name] = count
            except ValueError:
                # 如果解析失败，跳过此项
                continue
    
    return class_counts


def init_pipeline(config: dict):
    """
    初始化OCT-CCG管道
    """
    model_id = config.get("model", {}).get("id", "stabilityai/stable-diffusion-xl-base-1.0")
    device = config.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    pipe = OCTCCGSDPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
        safety_checker=None
    )
    
    # 将模型移动到指定设备
    pipe = pipe.to(device)
    
    # 设置调度器
    scheduler_type = config.get("scheduler", {}).get("type", "euler_ancestral")
    if scheduler_type == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:  # 默认使用euler_ancestral
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    return pipe, device


def run_oct_ccg_pipeline(
    pipe: OCTCCGSDPipeline,
    prompt: str,
    class_counts: Dict[str, int],
    output_path: str,
    config: dict,
    seed: Optional[int] = None,
    idx: Optional[int] = None
):
    """
    运行OCT-CCG管道生成图像
    """
    # 设置随机种子
    if seed is not None:
        set_seed(seed)
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # 获取OCT配置
    oct_config = config.get("oct", {})
    phase_scale = oct_config.get("phase_scale", 1e-2)
    strict_orthogonal = oct_config.get("strict_orthogonal", False)
    subtoken_strategy = oct_config.get("subtoken_strategy", "average")
    
    # 获取计数配置
    counting_config = config.get("counting", {})
    temp = counting_config.get("temp", 0.05)
    theta = counting_config.get("theta", 0.01)
    gamma = counting_config.get("gamma", 1e-2)
    agg_mode = counting_config.get("agg_mode", "sum")
    
    # 获取生成配置
    generation_config = config.get("generation", {})
    num_inference_steps = generation_config.get("num_inference_steps", 50)
    guidance_scale = generation_config.get("guidance_scale", 7.5)
    height = generation_config.get("height", 1024)
    width = generation_config.get("width", 1024)
    
    # 获取约束配置
    constraint_config = config.get("constraint", {})
    use_oct_embeddings = constraint_config.get("use_oct_embeddings", True)
    use_slot_splat = constraint_config.get("use_slot_splat", False)
    slot_splat_weight = constraint_config.get("slot_splat_weight", 1e-2)
    cnt_weight = constraint_config.get("cnt_weight", 1.0)
    orth_weight = constraint_config.get("orth_weight", 0.5)
    cross_overlap_weight = constraint_config.get("cross_overlap_weight", 0.5)
    eta_dual = constraint_config.get("eta_dual", 0.2)
    clamp_lambda = constraint_config.get("clamp_lambda", 10.0)
    
    # Sinkhorn参数
    sinkhorn_kwargs = {
        "temp": temp,
        "theta": theta,
        "gamma": gamma,
        "agg_mode": agg_mode,
        "cross_overlap_weight": cross_overlap_weight
    }
    
    # 生成图像
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            class_targets=class_counts,
            get_slot_heatmaps_resized_fn=get_slot_heatmaps_resized,
            soft_count_from_final_gs_fn=soft_count_from_final_gs,
            sinkhorn_kwargs=sinkhorn_kwargs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            use_oct_embeddings=use_oct_embeddings,
            use_slot_splat=use_slot_splat,
            slot_splat_weight=slot_splat_weight,
            oct_phase_scale=phase_scale,
            oct_strict_orthogonal=strict_orthogonal,
            oct_subtoken_strategy=subtoken_strategy,
            cnt_weight=cnt_weight,
            orth_weight=orth_weight,
            cross_overlap_weight=cross_overlap_weight,
            eta_dual=eta_dual,
            clamp_lambda=clamp_lambda
        )
    
    # 保存图像
    image = output.images[0]
    
    # 从class_counts中提取类别和数量信息
    class_info = []
    for class_name, count in class_counts.items():
        class_info.append(f"{class_name}_num={count}")
    class_info_str = "_".join(class_info)
    
    if idx is not None:
        filename = f"{class_info_str}_idx={idx:05d}.png"
    else:
        filename = f"{class_info_str}_seed={seed}.png"
    image_path = os.path.join(output_path, filename)
    image.save(image_path)
    
    return image_path


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="OCT-CCG: Orthogonal Class Token Controlled Counting Generation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--dataset", type=str, default="dataset/data_with_class_counts.json", help="数据集文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--prompt", type=str, help="直接指定的prompt，如果提供则忽略数据集")
    parser.add_argument("--class_counts", type=str, help="直接指定的类别计数，JSON格式，如'{\"cat\": 2, \"dog\": 1}'")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--num_samples", type=int, default=1, help="生成样本数量")
    parser.add_argument("--start_idx", type=int, default=0, help="从数据集的哪个索引开始")
    args = parser.parse_args()
    
    # 读取配置文件
    config = read_yaml(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化管道
    pipe, device = init_pipeline(config)
    print(f"模型已加载到设备: {device}")
    
    # 如果直接指定了prompt，则使用该prompt
    if args.prompt:
        # 解析类别计数
        if args.class_counts:
            try:
                class_counts = json.loads(args.class_counts)
            except json.JSONDecodeError:
                print("无法解析class_counts，尝试从prompt中解析")
                class_counts = parse_class_counts_from_prompt(args.prompt)
        else:
            class_counts = parse_class_counts_from_prompt(args.prompt)
        
        print(f"使用prompt: {args.prompt}")
        print(f"类别计数: {class_counts}")
        
        # 生成图像
        for i in range(args.num_samples):
            seed = args.seed + i if args.seed else None
            image_path, metadata_path = run_oct_ccg_pipeline(
                pipe=pipe,
                prompt=args.prompt,
                class_counts=class_counts,
                output_path=args.output_dir,
                config=config,
                seed=seed,
                idx=i
            )
            print(f"图像已保存到: {image_path}")
            print(f"元数据已保存到: {metadata_path}")
    else:
        # 从数据集加载prompt
        dataset = load_dataset(args.dataset)
        print(f"已加载数据集，共{len(dataset)}条样本")
        
        # 限制样本数量
        end_idx = min(args.start_idx + args.num_samples, len(dataset))
        samples_to_process = dataset[args.start_idx:end_idx]
        
        print(f"将处理样本 {args.start_idx} 到 {end_idx-1}")
        
        # 处理每个样本
        for i, sample in enumerate(tqdm(samples_to_process, desc="生成图像")):
            idx = args.start_idx + i
            prompt = sample["prompt"]
            class_counts = sample.get("class_counts", parse_class_counts_from_prompt(prompt))
            seed = sample.get("seed", args.seed)
            
            # 生成图像
            try:
                image_path, metadata_path = run_oct_ccg_pipeline(
                    pipe=pipe,
                    prompt=prompt,
                    class_counts=class_counts,
                    output_path=args.output_dir,
                    config=config,
                    seed=seed,
                    idx=idx
                )
                tqdm.write(f"样本 {idx}: 图像已保存到: {image_path}")
            except Exception as e:
                tqdm.write(f"样本 {idx} 生成失败: {str(e)}")
                continue


if __name__ == "__main__":
    main()