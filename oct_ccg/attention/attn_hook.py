import torch
import torch.nn as nn
from diffusers.models.attention import Attention as DiffusersAttention
from diffusers.models.unets import UNet2DConditionModel


import torch
import torch.nn as nn
from diffusers.models.attention import Attention as DiffusersAttention
from diffusers.models.unets import UNet2DConditionModel
import logging

# 配置日志，方便查看替换的模块
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionWithHook(DiffusersAttention):
    """
    改造版 Attention：
    - 仅记录关键模块（DownBlock2/MidBlock/UpBlock2）的cross-attn
    - 优化内存占用（仅保留最新注意力图）
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_attn_map = None  # 存储 (cls, idx) -> [B, heads, q_len]
        self.module_name = None     # 记录当前模块的路径（用于判断是否为关键模块）

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        token_index_map = cross_attention_kwargs.get("token_index_map", None)
        is_cross = encoder_hidden_states is not None

        # 非关键模块/非交叉注意力，不记录（节省内存）
        if not is_cross or self.module_name is None:
            return super().forward(hidden_states, encoder_hidden_states, attention_mask,** cross_attention_kwargs)

        # 关键模块的交叉注意力计算
        query = self.to_q(hidden_states)
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        B, N, C = query.shape
        head_dim = C // self.heads
        q = query.view(B, N, self.heads, head_dim).transpose(1, 2)  # [B, heads, q_len, head_dim]
        k = key.view(B, -1, self.heads, head_dim).transpose(1, 2)   # [B, heads, k_len, head_dim]
        v = value.view(B, -1, self.heads, head_dim).transpose(1, 2) # [B, heads, k_len, head_dim]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, q_len, k_len]
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, heads, q_len, k_len]

        # 仅记录关键模块的注意力图（过滤无效token位置）
        if token_index_map:
            slot_attn = {}
            for (cls, idx), token_pos in token_index_map.items():
                if 0 <= token_pos < attn_probs.shape[-1]:  # 确保token位置有效
                    attn_map = attn_probs[:, :, :, token_pos].detach().clone()  # [B, heads, q_len]
                    slot_attn[(cls, idx)] = attn_map
            self._last_attn_map = slot_attn  # 覆盖旧值，减少内存占用

        # 复用父类输出计算逻辑
        hidden_states = torch.matmul(attn_probs, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, N, C)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


def replace_cross_attention(unet: UNet2DConditionModel):
    """
    优化版：仅替换SDXL计数关键模块的交叉注意力
    关键模块：DownBlock2（down_blocks.2）、MidBlock（mid_block）、UpBlock2（up_blocks.2）
    """
    # SDXL关键模块的路径特征（基于官方UNet结构定义）
    KEY_MODULE_PATTERNS = [
        "down_blocks.2",  # DownBlock2（64×64分辨率，空间细节）
        "mid_block",      # MidBlock（32×32分辨率，全局语义）
        "up_blocks.2"     # UpBlock2（64×64分辨率，细节补全）
    ]
    target_modules = []

    # 筛选关键模块的交叉注意力
    for name, module in unet.named_modules():
        # 条件1：是Diffusers的Attention模块
        # 条件2：是交叉注意力（is_cross_attention=True）
        # 条件3：属于关键模块路径
        if (isinstance(module, DiffusersAttention) 
            and getattr(module, "is_cross_attention", False)
            and any(pattern in name for pattern in KEY_MODULE_PATTERNS)):
            target_modules.append(name)

    if not target_modules:
        logger.warning("未找到关键模块，可能是SDXL结构不匹配！")
        return unet

    # 替换关键模块
    for full_name in target_modules:
        # 拆分父模块路径（处理数字索引，如down_blocks.2.attentions.0）
        parent = unet
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        last_part = parts[-1]
        original_module = parent[int(last_part)] if last_part.isdigit() else getattr(parent, last_part)

        # 创建带Hook的新模块，复制权重并绑定模块名称
        new_attn = AttentionWithHook.from_config(original_module.config)
        new_attn.load_state_dict(original_module.state_dict())  # 权重无缝迁移
        new_attn.module_name = full_name  # 标记模块路径，用于后续过滤
        new_attn.to(next(original_module.parameters()).device)  # 同步设备

        # 替换模块
        if last_part.isdigit():
            parent[int(last_part)] = new_attn
        else:
            setattr(parent, last_part, new_attn)

    logger.info(f"成功替换 {len(target_modules)} 个关键交叉注意力模块：{target_modules}")
    return unet



import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def get_slot_heatmaps_resized(
    unet: UNet2DConditionModel,
    target_h: int,
    target_w: int,
    to_cpu: bool = True,
    agg_heads: str = "mean",
    head_reduce_before_resize: bool = False
):
    """
    恢复原逻辑版本：仅负责注意力图的分辨率适配、头聚合与跨层平均
    核心功能与原函数一致：
    - 自动推断不同模块注意力图的原始分辨率 (H_i, W_i)
    - 将注意力图插值到目标尺寸 (target_h, target_w)
    - 支持注意力头的均值/求和聚合，可选先聚合再插值以节省内存
    
    返回: dict {slot_key: tensor([B, C, target_h, target_w])}
          C == 1 if agg_heads in ("mean","sum") else C == 注意力头数量
    """
    # 校验聚合方式合法性
    assert agg_heads in ("mean", "sum"), f"agg_heads 仅支持 'mean' 或 'sum'，当前输入: {agg_heads}"
    results = {}

    # 遍历所有带 Hook 的注意力模块（仅处理已记录注意力图的模块）
    for module_path, module in unet.named_modules():
        if not isinstance(module, AttentionWithHook):
            continue
        if module._last_attn_map is None:
            logger.debug(f"模块 {module_path} 未记录注意力图，跳过")
            continue

        # 处理当前模块下的所有 slot 注意力图
        for slot_key, attn_map in module._last_attn_map.items():
            # attn_map 原始形状: [B, heads, q_len]（B=批次，heads=注意力头数，q_len=特征图像素总数）
            B, heads, q_len = attn_map.shape
            logger.debug(f"处理 slot {slot_key}：批次{B}，注意力头{heads}，特征像素数{q_len}")

            # --------------------------
            # 恢复原逻辑：自动推断原始分辨率 (H_i, W_i)
            # 优先按完全平方数推断（SDXL 多数模块满足），否则按目标宽度适配，最后 fallback 到 (q_len, 1)
            # --------------------------
            # 1. 优先尝试完全平方数（如 q_len=4096 → 64×64，q_len=1024→32×32）
            sq_root = int(q_len ** 0.5)
            if sq_root * sq_root == q_len:
                h_i, w_i = sq_root, sq_root
                logger.debug(f"模块 {module_path} 分辨率推断：{q_len} = {h_i}×{w_i}（完全平方）")
            # 2. 若不是完全平方，尝试按目标宽度 target_w 适配（确保宽高比合理）
            elif q_len % target_w == 0:
                w_i = target_w
                h_i = q_len // target_w
                logger.debug(f"模块 {module_path} 分辨率推断：{q_len} = {h_i}×{w_i}（按目标宽度适配）")
            # 3. 兜底：按 (q_len, 1) 处理（避免推断失败）
            else:
                h_i, w_i = q_len, 1
                logger.warning(f"模块 {module_path} 分辨率无法合理推断，兜底为 {h_i}×{w_i}（建议检查 q_len={q_len} 是否正常）")

            # 可选：将注意力图转移到 CPU（避免 GPU 显存占用过高）
            if to_cpu:
                attn_map = attn_map.cpu()
                logger.debug(f"slot {slot_key} 注意力图已转移到 CPU")

            # --------------------------
            # 恢复原逻辑：注意力头聚合与分辨率插值
            # --------------------------
            if head_reduce_before_resize and agg_heads in ("mean", "sum"):
                # 策略1：先聚合注意力头（减少后续插值计算量）
                logger.debug(f"先聚合注意力头（方式：{agg_heads}），再插值")
                if agg_heads == "mean":
                    # 均值聚合：[B, heads, q_len] → [B, 1, q_len]
                    attn_map = attn_map.mean(dim=1, keepdim=True)
                else:  # agg_heads == "sum"
                    # 求和聚合：[B, heads, q_len] → [B, 1, q_len]
                    attn_map = attn_map.sum(dim=1, keepdim=True)
                
                # 重塑为空间特征图：[B, 1, q_len] → [B×1, 1, h_i, w_i]（适配插值接口）
                B2, C2, _ = attn_map.shape  # C2=1（聚合后仅1个通道）
                spatial_attn = attn_map.view(B2 * C2, 1, h_i, w_i)
            else:
                # 策略2：先插值再聚合（保留所有注意力头信息，适合需要分析单头的场景）
                logger.debug(f"先插值，再聚合注意力头（方式：{agg_heads}）")
                # 重塑为空间特征图：[B, heads, q_len] → [B×heads, 1, h_i, w_i]
                spatial_attn = attn_map.view(B * heads, 1, h_i, w_i)

            # 双线性插值到目标尺寸（保持原逻辑的插值方式，确保空间信息平滑）
            resized_attn = F.interpolate(
                spatial_attn,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False  # 避免边缘像素失真
            )
            logger.debug(f"slot {slot_key} 注意力图插值完成：{h_i}×{w_i} → {target_h}×{target_w}")

            # 恢复批次和通道维度（反向重塑）
            if head_reduce_before_resize and agg_heads in ("mean", "sum"):
                # 对应策略1：[B×1, 1, H, W] → [B, 1, H, W]
                resized_attn = resized_attn.view(B2, C2, target_h, target_w)
            else:
                # 对应策略2：[B×heads, 1, H, W] → [B, heads, H, W]
                resized_attn = resized_attn.view(B, heads, target_h, target_w)

            # 将当前模块的注意力图加入结果集（按 slot_key 分组）
            if slot_key not in results:
                results[slot_key] = []
            results[slot_key].append(resized_attn)
            logger.debug(f"slot {slot_key} 已添加模块 {module_path} 的注意力图（累计 {len(results[slot_key])} 层）")

    # --------------------------
    # 恢复原逻辑：跨层平均（整合同一 slot 在不同模块的注意力图）
    # --------------------------
    final_heatmaps = {}
    for slot_key, layer_maps in results.items():
        # layer_maps: 列表，元素为 [B, C, H, W]（C=1或heads）
        num_layers = len(layer_maps)
        if num_layers == 0:
            logger.warning(f"slot {slot_key} 无有效注意力图层，跳过")
            continue

        # 堆叠所有层的注意力图：[num_layers, B, C, target_h, target_w]
        stacked_maps = torch.stack(layer_maps, dim=0)
        # 跨层均值：[num_layers, B, C, H, W] → [B, C, H, W]
        mean_cross_layer = stacked_maps.mean(dim=0)
        logger.debug(f"slot {slot_key} 跨 {num_layers} 层平均完成")

        # 若未提前聚合注意力头，此处按 agg_heads 聚合（保持与原逻辑一致）
        if not head_reduce_before_resize and agg_heads in ("mean", "sum"):
            if mean_cross_layer.shape[1] != 1:  # 仅当通道数不是1时聚合（避免重复）
                logger.debug(f"跨层后聚合注意力头（方式：{agg_heads}）")
                if agg_heads == "mean":
                    mean_cross_layer = mean_cross_layer.mean(dim=1, keepdim=True)
                else:
                    mean_cross_layer = mean_cross_layer.sum(dim=1, keepdim=True)

        # 保存最终热力图
        final_heatmaps[slot_key] = mean_cross_layer
        logger.info(
            f"slot {slot_key} 最终热力图生成完成："
            f"形状 {mean_cross_layer.shape}（B={mean_cross_layer.shape[0]}, "
            f"C={mean_cross_layer.shape[1]}, H={target_h}, W={target_w}）"
        )

    return final_heatmaps
#   4. 最终输出
#   最终得到的是每个实例单独的平均注意力图：
#   输出字典 final 的键是 (cls, idx)（每个实例），
#   值是该实例在所有相关层的注意力图经过 “插值→平均” 后的结果（形状 [B, C, target_h, target_w]）。
#   按实例分组，每个实例的注意力图会整合它在所有目标层的信息（先插值统一尺寸，再算术平均），最终每个实例对应一个唯一的平均热力图



# --------------------- 示例 ---------------------
if __name__ == "__main__":
    # 1. 加载SDXL UNet并替换交叉注意力（用你修改后的 replace_cross_attention 函数）
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet"
    ).to("cuda")
    unet = replace_cross_attention(unet)

    # 2. 构造假数据（与原始测试逻辑一致）
    B, C, H, W = 1, 4, 128, 128
    latents = torch.randn(B, C, H, W).to("cuda")
    t = torch.tensor([10]).to("cuda")
    encoder_hidden_states = torch.randn(B, 77, unet.config.cross_attention_dim).to("cuda")

    # 3. 定义token映射（原始格式）
    token_index_map = {("cat", 0): 2, ("dog", 0): 5}

    # 4. 前向传播记录注意力
    with torch.no_grad():
        unet(
            latents, t,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs={"token_index_map": token_index_map}
        )

    # 5. 提取注意力热力图（调用恢复后的函数，参数与原始一致）
    heatmaps = get_slot_heatmaps_resized(
        unet=unet,
        target_h=64,  # 目标高度
        target_w=64,  # 目标宽度
        to_cpu=True,    # 转移到CPU
        agg_heads="mean",  # 注意力头均值聚合
        head_reduce_before_resize=True  # 先聚合再插值
    )

    # 6. 查看结果（原始格式）
    for slot_key, hmap in heatmaps.items():
        print(f"Slot {slot_key} 热力图形状：{hmap.shape}") 
