"""
OCT-CCG (Orthogonal Counting Tokens for Controlled Generation) Package
"""

# 导入各个子模块
from . import attention
from . import core
from . import counting
from . import pipeline
from . import tokenization

# 从子模块导入主要类和函数
from .attention.attn_hook import (
    AttentionWithHook,
    replace_cross_attention,
    get_slot_heatmaps_resized
)

from .core.primal_dual_guidance import (
    primal_dual_guidance_step
)

from .core.slot_splat import (
    slot_splat_prior,
    register_slot_splat_feature_injection
)

from .counting.soft_count import (
    soft_count_from_attn,
    soft_count_class,
    soft_count_from_final
)

from .counting.soft_count_gumbel_sinkhorn import (
    soft_count_from_final_gs
)

from .tokenization.oct_tokens import (
    make_oct_embeddings
)

__all__ = [
    # 子模块
    "attention",
    "core",
    "counting",
    "pipeline",
    "tokenization",
    
    # attention 模块
    "AttentionWithHook",
    "replace_cross_attention",
    "get_slot_heatmaps_resized",
    
    # core 模块
    "primal_dual_guidance_step",
    "slot_splat_prior",
    "register_slot_splat_feature_injection",
    
    # counting 模块
    "soft_count_from_attn",
    "soft_count_class",
    "soft_count_from_final",
    "soft_count_from_final_gs",
    
    # tokenization 模块
    "make_oct_embeddings"
]