import torch
import torch.nn.functional as F
import numpy as np

# =============================
# Step 1: 生成 Gaussian Slot Splat
# =============================

def make_gaussian_splat(h, w, center, sigma=0.15):
    """
    生成单个 slot mask
    h,w: 特征图分辨率
    center: (x,y) in [0,1]
    sigma: 高斯扩散范围
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h),
        torch.linspace(0, 1, w),
        indexing="ij"
    )
    d2 = (xx - center[0])**2 + (yy - center[1])**2
    gauss = torch.exp(-d2 / (2 * sigma**2))
    return gauss / gauss.max()   # 归一化到 [0,1]


def slot_splat_prior(slot_dict, h=64, w=64, sigma=0.15, seed=42):
    """
    slot_dict: {"cat":3, "dog":2}
    输出: layout prior [1,1,h,w]
    """
    torch.manual_seed(seed)
    layout = torch.zeros(h, w)
    for cname, n in slot_dict.items():
        centers = torch.rand(n, 2)  # 随机slot位置
        for c in centers:
            layout += make_gaussian_splat(h, w, c, sigma)
    layout = layout / layout.max()
    return layout[None, None]  # [1,1,h,w]


# =============================
# Step 2: 注册到 U-Net 浅层
# =============================

def register_slot_splat_feature_injection(unet, slot_dict, weight=1e-2, h=64, w=64):
    """
    在 U-Net 最浅层（通常是 encoder 第1层）加 slot prior
    """
    prior = slot_splat_prior(slot_dict, h, w).to(next(unet.parameters()).device)

    def forward_hook(module, input, output):
        # output: [B,C,H,W]
        B, C, H, W = output.shape
        prior_resized = F.interpolate(prior, size=(H, W), mode="bilinear")
        return output + weight * prior_resized.expand(B, -1, H, W)

    # 在最浅层 conv 注册hook (通常是 down_blocks[0])
    first_block = unet.down_blocks[0].resnets[0]
    first_block.register_forward_hook(forward_hook)

    return unet


# =============================
# Step 3: 在 Stable Diffusion 中使用
# =============================

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 注册 Slot-Splat
slot_dict = {"cat": 3, "dog": 2}
pipe.unet = register_slot_splat_feature_injection(pipe.unet, slot_dict, weight=1e-2, h=64, w=64)

# 采样
prompt = "a photo of cats and dogs in a park"
image = pipe(prompt, guidance_scale=7.5).images[0]
image.save("slot_splat_feature.png")
