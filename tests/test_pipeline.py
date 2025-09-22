import os
from huggingface_hub import snapshot_download

def download_sdxl_fp16():
    # 本地保存目录
    local_dir = os.path.abspath("./models/sdxl-base-fp16")
    os.makedirs(local_dir, exist_ok=True)

    print(f"📂 模型将保存到: {local_dir}")

    # 下载 SDXL fp16 最小必要文件
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 避免 Windows 下的 symlink 警告
        allow_patterns=[
            "diffusion_pytorch_model.fp16.safetensors",   # UNet
            "text_encoder/model.fp16.safetensors",        # 文本编码器1
            "text_encoder_2/model.fp16.safetensors",      # 文本编码器2
            "tokenizer/*",                                # tokenizer1
            "tokenizer_2/*",                              # tokenizer2
            "vae/diffusion_pytorch_model.safetensors",    # VAE (fp32, 官方推荐)
            "model_index.json",                           # 模型索引
            "scheduler/*.json"                            # 采样器配置
        ]
    )

    print("✅ SDXL fp16 模型下载完成！")

if __name__ == "__main__":
    download_sdxl_fp16()
