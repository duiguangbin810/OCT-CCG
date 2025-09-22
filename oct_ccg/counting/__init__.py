"""
Counting Module for OCT-CCG
"""

from .soft_count import (
    soft_count_from_attn,
    soft_count_class,
    soft_count_from_final
)

from .soft_count_gumbel_sinkhorn import (
    soft_count_from_final_gs
)

__all__ = [
    "soft_count_from_attn",
    "soft_count_class",
    "soft_count_from_final",
    "soft_count_from_final_gs"
]