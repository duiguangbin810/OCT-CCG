# oct_ccg_sdxl_pipeline.py
import os
import torch
import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import math
import logging
# 设置 logger
logger = logging.getLogger(__name__)
from diffusers.loaders import TextualInversionLoaderMixin, StableDiffusionXLLoraLoaderMixin
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import adjust_lora_scale_text_encoder, scale_lora_layers, unscale_lora_layers
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

from oct_ccg.attention.attn_hook import replace_cross_attention
# 导入slot_splat模块
from oct_ccg.core.slot_splat import make_gaussian_splat, slot_splat_prior, register_slot_splat_feature_injection

# 复用你定义的工具函数
EPS = 1e-8

def _ensure_hw_tensor(t):
    if t is None:
        return None
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim == 3:
        return t
    if t.ndim == 4:
        return t.squeeze(1)
    raise ValueError(f"不支持的张量维度: attn_map ndim={t.ndim}")

def compute_orth_loss_from_final_dict(final_dict: Dict[Tuple[str,int], torch.Tensor],
                                      slot_index_map: Dict[str, list],
                                      device: torch.device):
    orth_loss = torch.tensor(0.0, device=device)
    for cls, keys in slot_index_map.items():
        if len(keys) <= 1:
            continue
        maps = []
        for k in keys:
            m = final_dict.get(k, None)
            if m is None:
                continue
            m = _ensure_hw_tensor(m)
            B, H, W = m.shape
            maps.append(m.reshape(B, -1))
        if len(maps) <= 1:
            continue
        M = torch.stack(maps, dim=1)
        M = F.normalize(M, dim=2)
        inner = torch.einsum('bsh,bth->bst', M, M)
        diag = torch.eye(M.shape[1], device=device).unsqueeze(0)
        offdiag = inner * (1.0 - diag)
        orth_loss += torch.sum(offdiag**2) / (M.shape[0] if M.shape[0] > 0 else 1.0)
    return orth_loss if isinstance(orth_loss, torch.Tensor) else torch.tensor(0., device=device)

def orthogonalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """对向量组进行QR分解实现严格正交化（OCT核心工具函数）"""
    if vectors.ndim != 2:
        raise ValueError(f"正交化需要二维张量，实际输入维度: {vectors.ndim}")
    num_vectors, dim = vectors.shape
    if num_vectors > dim:
        raise ValueError(f"向量数量({num_vectors})超过维度({dim})，无法正交化")
    q, _ = torch.linalg.qr(vectors)
    return q

def apply_oct_to_embeddings(
    base_embeddings: torch.Tensor,  # 单个编码器的基础嵌入（已过clip_skip）
    prompt: str,
    class_counts: Dict[str, int],
    tokenizer,
    device: torch.device,
    phase_scale: float = 1e-2,
    strict_orthogonal: bool = False,
    subtoken_strategy: str = "average"
) -> Tuple[torch.Tensor, Dict[Tuple[str, int], int]]:
    """
    对单个编码器的基础嵌入应用OCT逻辑（改造自原make_oct_embeddings的核心逻辑）
    输入：基础嵌入（seq_len, hidden_dim）、prompt、类别映射等
    输出：OCT增强后的嵌入（new_seq_len, hidden_dim）、token_index_map
    """
    # 1. 分词获取原始token序列（用于匹配类别词位置）
    tok = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
    input_ids = tok.input_ids[0].tolist()  # 原始token ID序列
    attention_mask = tok.attention_mask[0].tolist()  # 注意力掩码（过滤padding）
    seq_len, hidden_dim = base_embeddings.shape

    # 2. 缓存类别词的token序列（用于后续匹配）
    cls2ids = {
        cls: tokenizer(cls, add_special_tokens=False).input_ids[0].tolist()
        for cls in class_counts
    }

    # 3. 遍历token序列，对类别词嵌入添加相位偏移并复制
    new_embs = []
    token_index_map = {}
    processed_cls = set()  # 标记已处理的类别（避免重复）
    i = 0  # 遍历原始token序列的指针

    while i < len(input_ids):
        # 跳过padding token（注意力掩码为0）
        if attention_mask[i] == 0:
            i += 1
            continue

        matched_any = False
        # 遍历所有类别，检查当前位置是否匹配类别词的token序列
        for cls, count in class_counts.items():
            if cls in processed_cls:
                continue  # 跳过已处理的类别
            cls_ids = cls2ids[cls]
            cls_len = len(cls_ids)  # 类别词的子词数量（如"cat"可能是1个token，"black cat"是2个）

            # 检查当前位置是否有足够长度匹配，且token序列一致
            if i + cls_len > len(input_ids):
                continue
            if input_ids[i:i+cls_len] == cls_ids:
                # 3.1 处理多子词：按策略取类别词的嵌入（平均或最后一个子词）
                cls_embeddings = base_embeddings[i:i+cls_len]  # 类别词对应的基础嵌入片段
                if subtoken_strategy == "average":
                    cls_base_emb = cls_embeddings.mean(dim=0)  # 子词嵌入平均
                else:  # "last"
                    cls_base_emb = cls_embeddings[-1]  # 取最后一个子词嵌入

                # 3.2 生成正交相位向量（sin/cos编码）
                # 维度：count（实例数）× hidden_dim（嵌入维度）
                idx = torch.arange(count, device=device).unsqueeze(1)  # (count, 1)
                dims = torch.arange(hidden_dim, device=device).unsqueeze(0)  # (1, hidden_dim)
                phase_matrix = torch.sin(idx * dims * math.pi / hidden_dim)  # 相位矩阵

                # 归一化相位向量（避免幅度影响）
                phases = phase_matrix / (phase_matrix.norm(dim=1, keepdim=True) + 1e-9)
                # 严格正交化（可选）
                if strict_orthogonal and count > 1:
                    phases = orthogonalize_vectors(phases)
                # 缩放相位偏移（控制影响强度）
                phases = phases * phase_scale

                # 3.3 复制类别词嵌入并添加相位偏移（生成count个实例）
                for slot in range(count):
                    enhanced_emb = cls_base_emb + phases[slot]  # 基础嵌入 + 相位偏移
                    new_embs.append(enhanced_emb.unsqueeze(0))  # 增加序列维度（1, hidden_dim）
                    # 记录（类别, 实例索引）到新序列位置的映射
                    token_index_map[(cls, slot)] = len(new_embs) - 1

                # 标记类别已处理，跳过当前类别词的token长度
                matched_any = True
                processed_cls.add(cls)
                i += cls_len
                break

        # 非类别词：直接保留原始基础嵌入
        if not matched_any:
            new_embs.append(base_embeddings[i].unsqueeze(0))
            i += 1

    # 4. 警告未匹配的类别（避免用户拼写错误）
    unmatched_cls = [cls for cls in class_counts if cls not in processed_cls]
    if unmatched_cls:
        warnings.warn(f"OCT逻辑：以下类别在prompt中未找到匹配，请检查拼写: {unmatched_cls}")

    # 5. 拼接新嵌入序列（new_seq_len, hidden_dim）
    enhanced_embeddings = torch.cat(new_embs, dim=0)
    return enhanced_embeddings, token_index_map

    
class OCTCCGSDPipeline(StableDiffusionXLPipeline):
    """
    基于StableDiffusionXLPipeline的正交类别token可控计数生成管道
    整合了OCT-CCG的所有核心功能，包括正交相位编码、注意力钩子、软计数和原始-对偶引导
    现在集成了slot_splat功能，用于在U-Net浅层注入空间先验
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换U-Net中的交叉注意力模块
        self.unet = replace_cross_attention(self.unet)
        # 初始化slot_splat相关变量
        self.slot_splat_hooks = []
        
    def register_slot_splat(self, class_targets: Dict[str, int], weight: float = 1e-2, h: int = 64, w: int = 64):
        """
        注册slot_splat到U-Net浅层
        """
        # 移除之前注册的hook
        self.remove_slot_splat_hooks()
        
        # 注册新的hook并保存引用
        hook_handle = register_slot_splat_feature_injection(
            self.unet, 
            class_targets, 
            weight=weight, 
            h=h, 
            w=w
        )
        
        # 将钩子添加到列表中以便后续移除
        self.slot_splat_hooks.append(hook_handle)
        
    def remove_slot_splat_hooks(self):
        """
        移除所有slot_splat相关的hook
        """
        for hook in self.slot_splat_hooks:
            hook.remove()
        self.slot_splat_hooks = []
        
    # --------------------------
    # 第二步：融合OCT的encode_prompt函数
    # --------------------------
    def encode_prompt(
            self,
            prompt: str,
            prompt_2: Optional[str] = None,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Optional[str] = None,
            negative_prompt_2: Optional[str] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
            # --------------------------
            # 新增：OCT相关参数（默认开启）
            # --------------------------
            use_oct_embeddings: bool = True,  # 是否启用正交相位编码
            class_targets: Optional[Dict[str, int]] = None,  # 类别-数量映射（如{"cat":3, "dog":2}）
            oct_phase_scale: float = 1e-2,  # 相位偏移缩放因子
            oct_strict_orthogonal: bool = False,  # 是否严格正交化相位向量
            oct_subtoken_strategy: str = "average"  # 多子词处理策略（average/last）
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[Tuple[str, int], int]]]:
        """
        融合正交相位编码（OCT）的SDXL提示词编码函数
        新增返回值：token_index_map - （类别, 实例索引）到新序列位置的映射（仅use_oct_embeddings=True时有效）
        """
        device = device or self._execution_device
        token_index_map = None  # 初始化OCT的token映射（默认None）

        # --------------------------
        # 原有逻辑：LoRA缩放调整
        # --------------------------
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale
            # 调整第一个文本编码器的LoRA
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)
            # 调整第二个文本编码器的LoRA（SDXL特有）
            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        # --------------------------
        # 原有逻辑：提示词格式处理与批量大小确定
        # --------------------------
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        # 定义SDXL双编码器与分词器（兼容单编码器场景）
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]

        # --------------------------
        # 核心逻辑1：生成正向提示词嵌入（含OCT增强）
        # --------------------------
        if prompt_embeds is None:
            # SDXL默认用prompt_2补充提示，无则复用prompt
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompts = [prompt, prompt_2]  # 双编码器对应两个提示（可相同）
            prompt_embeds_list = []  # 存储两个编码器的嵌入

            # 遍历双编码器，分别生成基础嵌入并应用OCT
            for idx, (prompt_batch, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
                # 原有逻辑：Textual Inversion（自定义嵌入）处理
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt_batch = self.maybe_convert_prompt(prompt_batch, tokenizer)

                # 原有逻辑：分词（转换为token ID）
                text_inputs = tokenizer(
                    prompt_batch,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt_batch, padding="longest", return_tensors="pt").input_ids

                # 原有逻辑：截断警告（CLIP最大token长度限制）
                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warnloggering(
                        f"提示词截断警告：CLIP仅支持{tokenizer.model_max_length}个token，被截断内容: {removed_text}"
                    )

                # 原有逻辑：文本编码器生成hidden_states（含所有中间层）
                encoder_outputs = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                    return_dict=True  # 用dict格式便于获取hidden_states
                )

                # 原有逻辑：处理池化嵌入（Pooled Embedding）
                if pooled_prompt_embeds is None and encoder_outputs.pooler_output.ndim == 2:
                    pooled_prompt_embeds = encoder_outputs.pooler_output

                # 原有逻辑：根据clip_skip选择特定层的hidden_states（SDXL默认倒数第二层）
                if clip_skip is None:
                    base_embeds = encoder_outputs.hidden_states[-2]  # 基础嵌入（未增强）
                else:
                    base_embeds = encoder_outputs.hidden_states[-(clip_skip + 2)]  # 按clip_skip调整层

                # --------------------------
                # 新增：应用OCT逻辑（增强基础嵌入）
                # 注意：仅处理第一个编码器的第一个batch（默认单batch，多batch需扩展）
                # --------------------------
                if use_oct_embeddings and class_targets is not None:
                    # 仅支持单batch（多batch需循环处理，此处简化）
                    if batch_size > 1:
                        warnings.warn("OCT逻辑暂不支持多batch，仅处理第一个batch的嵌入")
                    # 提取第一个batch的基础嵌入（seq_len, hidden_dim）
                    single_base_embeds = base_embeds[0]  # （batch=0, seq_len, hidden_dim）→（seq_len, hidden_dim）
                    # 应用OCT增强（传入当前编码器的tokenizer和基础嵌入）
                    enhanced_embeds, current_map = apply_oct_to_embeddings(
                        base_embeddings=single_base_embeds,
                        prompt=prompt_batch[0],  # 第一个batch的prompt
                        class_counts=class_targets,
                        tokenizer=tokenizer,
                        device=device,
                        phase_scale=oct_phase_scale,
                        strict_orthogonal=oct_strict_orthogonal,
                        subtoken_strategy=oct_subtoken_strategy
                    )
                    if idx == 0:
                        token_index_map = current_map
                    # 恢复batch维度（1, new_seq_len, hidden_dim）
                    enhanced_embeds = enhanced_embeds.unsqueeze(0)
                    # 替换原基础嵌入（仅第一个batch，多batch需扩展）
                    base_embeds = torch.cat([enhanced_embeds, base_embeds[1:]], dim=0) if batch_size > 1 else enhanced_embeds

                # 将（增强后）的基础嵌入加入列表
                prompt_embeds_list.append(base_embeds)

            # 原有逻辑：拼接双编码器的嵌入（维度：batch × new_seq_len × (dim1 + dim2)）
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # --------------------------
        # 原有逻辑2：生成负向提示词嵌入（分类器-free引导）
        # --------------------------
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # 情况1：无负提示且配置强制零向量→负嵌入为零
            if zero_out_negative_prompt:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            # 情况2：有负提示→生成负嵌入（逻辑同正向，暂不支持OCT）
            else:
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt
                # 统一负提示格式（批量大小匹配）
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
                uncond_tokens = [negative_prompt, negative_prompt_2]
                negative_prompt_embeds_list = []

                # 遍历双编码器生成负嵌入（无OCT，保持原有逻辑）
                for negative_batch, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                    if isinstance(self, TextualInversionLoaderMixin):
                        negative_batch = self.maybe_convert_prompt(negative_batch, tokenizer)
                    # 分词
                    uncond_input = tokenizer(
                        negative_batch,
                        padding="max_length",
                        max_length=prompt_embeds.shape[1],  # 与正向嵌入序列长度一致
                        truncation=True,
                        return_tensors="pt",
                    )
                    # 文本编码
                    uncond_outputs = text_encoder(
                        uncond_input.input_ids.to(device),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    # 处理池化嵌入
                    if negative_pooled_prompt_embeds is None and uncond_outputs.pooler_output.ndim == 2:
                        negative_pooled_prompt_embeds = uncond_outputs.pooler_output
                    # 选择clip_skip层
                    if clip_skip is None:
                        uncond_embeds = uncond_outputs.hidden_states[-2]
                    else:
                        uncond_embeds = uncond_outputs.hidden_states[-(clip_skip + 2)]
                    negative_prompt_embeds_list.append(uncond_embeds)

                # 拼接双编码器负嵌入
                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        # --------------------------
        # 原有逻辑3：嵌入格式调整（数据类型、批量生成扩展）
        # --------------------------
        # 调整正向嵌入数据类型（与编码器一致）
        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        # 扩展嵌入以支持“每个提示生成多张图”（mps友好的重复方式）
        bs_embed, seq_len, embed_dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)  # 按图像数量重复
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, embed_dim)  # 调整batch维度

        # 负嵌入格式调整（同正向）
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)
            # 扩展负嵌入
            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)

        # 池化嵌入格式调整
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
        if do_classifier_free_guidance and negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                batch_size * num_images_per_prompt, -1
            )

        # --------------------------
        # 原有逻辑4：恢复LoRA层原始缩放（清理工作）
        # --------------------------
        if self.text_encoder is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder_2, lora_scale)

        # --------------------------
        # 新增：返回OCT的token_index_map（最后一个返回值，兼容原有调用）
        # --------------------------
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, token_index_map
    
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        # 新增：原始-对偶约束相关参数
        use_oct_embeddings: bool = True,  # 是否启用正交相位编码
        class_targets: Dict[str, int] = None,  # 目标类别-数量映射（如{"cat":2, "dog":1}）
        lambdas: Optional[Dict[str, torch.Tensor]] = None,  # 对偶变量初始值
        get_slot_heatmaps_resized_fn: Callable = None,  # 获取槽位热图的回调
        soft_count_from_final_gs_fn: Callable = None,  # Sinkhorn可微计数回调
        # 原始-对偶超参数
        cnt_beta: float = 0.1,
        cnt_weight: float = 1.0,
        orth_weight: float = 0.5,
        cross_overlap_weight: float = 0.5,
        eta_dual: float = 0.2,
        clamp_lambda: float = 10.0,
        sinkhorn_kwargs: Dict[str, Any] = None,
        debug: bool = False,
        
        # 新增：slot_splat相关参数
        use_slot_splat: bool = False,  # 是否启用slot_splat
        slot_splat_weight: float = 1e-2,  # slot_splat的权重
        slot_splat_h: int = 64,  # slot_splat的高度分辨率
        slot_splat_w: int = 64,  # slot_splat的宽度分辨率
        **kwargs,
    ):
        """
        重写__call__方法以整合OCT-CCG的所有功能，包括slot_splat
        返回:
            StableDiffusionXLPipelineOutput对象
        """       
        # # 如果需要，注入slot-splat先验
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. 注册slot_splat（如果启用）
        if use_slot_splat and class_targets is not None:
            self.register_slot_splat(
                class_targets=class_targets,
                weight=slot_splat_weight,
                h=slot_splat_h,
                w=slot_splat_w
            )
        else:
            # 如果不使用slot_splat，移除之前注册的hook
            self.remove_slot_splat_hooks()

        # 4. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            token_index_map,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            class_targets=class_targets,
            use_oct_embeddings=use_oct_embeddings,  # 是否启用正交相位编码
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        cross_attention_kwargs = self.cross_attention_kwargs or {}
        cross_attention_kwargs["token_index_map"] = token_index_map  # 关键：添加 token_index_map

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # 9.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 10. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        # 新增：初始化原始-对偶相关变量
        device = self.device
        B = batch_size
        sinkhorn_kwargs = sinkhorn_kwargs or {}
        sinkhorn_kwargs["cross_overlap_weight"] = cross_overlap_weight  # 传递跨类重叠权重
        # 初始化对偶变量（若未提供）
        if lambdas is None:
            lambdas = {cls: torch.zeros((B,), device=device) for cls in class_targets.keys()} if class_targets else {}
        # 诊断信息容器
        diagnostics = []

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latents = latents.clone().requires_grad_(True)  # 关键：开启梯度以计算约束对latents的影响
                t_tensor = torch.tensor([t], device=device).repeat(B)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # --------------------------
                # 新增：原始-对偶约束调整（噪声预测后 → latents更新前）
                # --------------------------
                if class_targets and get_slot_heatmaps_resized_fn and soft_count_from_final_gs_fn:
                    # a. 获取槽位热图（调整为latents空间尺寸）
                    H_lat, W_lat = latents.shape[2], latents.shape[3]
                    final_dict = get_slot_heatmaps_resized_fn(self.unet, H_lat, W_lat, to_cpu=False)

                    # b. Sinkhorn可微计数（计算实例计数误差）
                    class_hat_counts = soft_count_from_final_gs_fn(final_dict,** sinkhorn_kwargs)
                    # 补全缺失类别（防止计数函数漏返）
                    for cls in class_targets.keys():
                        if cls not in class_hat_counts:
                            class_hat_counts[cls] = torch.zeros((B,), device=device)

                    # c. 构建损失函数（原始-对偶计数损失 + 正交损失 + 跨类重叠损失）
                    # 向量化处理类别预测/目标/对偶变量
                    classes = list(class_targets.keys())
                    num_classes = len(classes)
                    pred_tensor = torch.stack([class_hat_counts[c] for c in classes], dim=0)  # [C, B]
                    target_tensor = torch.tensor([class_targets[c] for c in classes], device=device).view(num_classes, 1).expand(-1, B)  # [C, B]
                    lambda_tensor = torch.stack([lambdas[c] for c in classes], dim=0)  # [C, B]

                    # 计数损失（原始-对偶形式）
                    diff_tensor = pred_tensor - target_tensor
                    cnt_loss_tensor = lambda_tensor * diff_tensor + cnt_beta * torch.abs(diff_tensor)
                    cnt_loss = torch.sum(cnt_loss_tensor)

                    # 正交损失
                    orth_loss = compute_orth_loss_from_final_dict(final_dict, token_index_map, device=device)

                    # 跨类重叠损失
                    cross_overlap_loss = torch.tensor(0.0, device=device)
                    class_agg_maps = {}
                    # 构建类别级聚合热图（槽位求和+归一化）
                    for cls, keys in token_index_map.items():
                        maps_list = []
                        for key in keys:
                            m = final_dict.get(key, None)
                            if m is None:
                                continue
                            m = _ensure_hw_tensor(m).reshape(B, -1)  # [B, H*W]
                            maps_list.append(m)
                        if not maps_list:
                            class_agg_maps[cls] = torch.zeros((B, 1), device=device)
                            continue
                        agg = torch.sum(torch.stack(maps_list, dim=1), dim=1)  # [B, H*W]
                        class_agg_maps[cls] = F.normalize(agg + EPS, dim=1)
                    # 成对计算跨类重叠
                    class_list = list(class_agg_maps.keys())
                    for i in range(len(class_list)):
                        for j in range(i+1, len(class_list)):
                            ai = class_agg_maps[class_list[i]]
                            aj = class_agg_maps[class_list[j]]
                            cross_overlap_loss += torch.sum(torch.sum(ai * aj, dim=1))  # 逐样本重叠求和
                    cross_overlap_loss *= cross_overlap_weight

                    # 总损失
                    total_loss = cnt_weight * cnt_loss + orth_weight * orth_loss + cross_overlap_loss

                    # d. 反向传播：计算损失对latents的梯度（约束转化为latents修正方向）
                    grads = torch.autograd.grad(total_loss, latents, retain_graph=False, create_graph=False)[0]
                    grads = grads if grads is not None else torch.zeros_like(latents)

                    # e. 对偶变量更新（对偶上升）
                    with torch.no_grad():
                        lambda_tensor_new = lambda_tensor + eta_dual * diff_tensor.detach()
                        lambda_tensor_new = torch.clamp(lambda_tensor_new, -clamp_lambda, clamp_lambda)
                        # 更新对偶变量字典
                        for idx, cls in enumerate(classes):
                            lambdas[cls] = lambda_tensor_new[idx].clone()

                    # f. 修正latents（将约束梯度投影到latents空间）
                    grad_norm = torch.norm(grads.view(B, -1), dim=1).view(B, 1, 1, 1)
                    grads_normed = grads / (grad_norm + EPS)  # 梯度归一化
                    latents_corrected = latents - self._guidance_scale * grads_normed  # 约束修正

                    # 记录诊断信息
                    with torch.no_grad():
                        diagnostics.append({
                            "step": i,
                            "timestep": t,
                            "cnt_loss": cnt_loss.detach().cpu(),
                            "orth_loss": orth_loss.detach().cpu(),
                            "cross_overlap_loss": cross_overlap_loss.detach().cpu(),
                            "total_loss": total_loss.detach().cpu(),
                            "grad_norm": grad_norm.mean().detach().cpu(),
                            "hat_counts": {k: v.detach().cpu() for k, v in class_hat_counts.items()},
                        })
                else:
                    # 无约束时直接使用原始latents
                    latents_corrected = latents.detach()

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents_corrected, **extra_step_kwargs, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)