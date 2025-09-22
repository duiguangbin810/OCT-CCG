"""
Attention Module for OCT-CCG
"""

from .attn_hook import (
    AttentionWithHook,
    replace_cross_attention,
    get_slot_heatmaps_resized
)

__all__ = [
    "AttentionWithHook",
    "replace_cross_attention",
    "get_slot_heatmaps_resized"
]