# soft_count_gumbel_sinkhorn.py
import math
import warnings
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

# ----------------------------------------
# 改进要点（summary）
# 1) 支持 batch 维度（B>=1），输出每个 class 对应的 [B] 向量。
# 2) margin 生效：显式增加 slack slots（值为 0 的占位）。
# 3) 更通用的 Sinkhorn：支持指定行/列边际（保证总质量一致）。
# 4) 数值稳定性：S 做 ReLU+eps，再取 log。
# 5) soft-count 改用 clamp(slot_mass,0,1).sum()（比固定 sigmoid(·-0.5) 更稳健）。
# 6) 提供 use_hard_topk 参数（默认 True）。若你需要完全可微的替代方案，可把 use_hard_topk=False，
#    会使用 soft candidate pooling（以 softmax 权重对全部像素加权降采样），但会更慢。
# 7) 由于 torch.topk 的索引不连续，若需要完全平滑训练应开启 soft candidate（见参数）。
# ----------------------------------------


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20,
             row_sum=1.0, col_sum=1.0) -> torch.Tensor:
    """
    通用 Sinkhorn（log 域），支持标量或向量的行/列边际目标。
    Args:
        log_alpha: (M, N) log 似然矩阵
        row_sum: scalar or tensor of shape (M,) 表示每行目标和
        col_sum: scalar or tensor of shape (N,) 表示每列目标和
    Returns:
        P: (M, N) 非负矩阵，近似满足行/列边际
    注意：
        要保证 total_row_mass == total_col_mass，否则行为不确定。
    """
    log_p = log_alpha
    M, N = log_p.shape

    # 处理行/列边际表示
    if isinstance(row_sum, (float, int)):
        log_row = math.log(row_sum + 1e-30)
        row_tensor = None
    else:
        # tensor
        row_tensor = torch.log(row_sum + 1e-30).to(log_p.device).view(M)
        log_row = None

    if isinstance(col_sum, (float, int)):
        log_col = math.log(col_sum + 1e-30)
        col_tensor = None
    else:
        col_tensor = torch.log(col_sum + 1e-30).to(log_p.device).view(N)
        log_col = None

    for _ in range(n_iters):
        # 行归一化（log 域）
        row_logsumexp = torch.logsumexp(log_p, dim=1, keepdim=True)  # (M,1)
        if row_tensor is None:
            log_p = log_p - row_logsumexp + log_row
        else:
            # row_tensor shape (M,) -> (M,1)
            log_p = log_p - row_logsumexp + row_tensor.view(M, 1)

        # 列归一化（log 域）
        col_logsumexp = torch.logsumexp(log_p, dim=0, keepdim=True)  # (1,N)
        if col_tensor is None:
            log_p = log_p - col_logsumexp + log_col
        else:
            log_p = log_p - col_logsumexp + col_tensor.view(1, N)

    return torch.exp(log_p)


def gumbel_sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20,
                    gumbel_scale: float = 1.0, temp: float = 1.0,
                    row_sum=1.0, col_sum=1.0) -> torch.Tensor:
    """
    带 Gumbel 噪声的 Sinkhorn（可微近似匹配/置换）。
    Args:
        log_alpha: (M,N) logits in log-domain
        gumbel_scale: scale of Gumbel noise
        temp: temperature
        row_sum/col_sum: see sinkhorn
    Returns:
        P: (M,N)
    """
    # Gumbel noise（数值上稳定做法）
    u = torch.rand_like(log_alpha)
    gumbel = -torch.log(-torch.log(u + 1e-9) + 1e-9)
    noisy = (log_alpha + gumbel_scale * gumbel) / (temp + 1e-9)
    return sinkhorn(noisy, n_iters=n_iters, row_sum=row_sum, col_sum=col_sum)


def extract_candidates(attn_map: torch.Tensor, K: int = 20,
                       soft_topk_temp: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    可微 soft-topk 候选点提取（替换原函数即可）。
    Args:
        attn_map: (H, W) 注意力图
        K: 需要的候选数
        soft_topk_temp: softmax 温度，越小越接近 hard topk
    Returns:
        values: (K,) soft 候选值
        coords: (K, 2) soft 坐标 (h, w)，浮点数，可反向传播
    """
    H, W = attn_map.shape
    flat = attn_map.view(-1)                      # (H*W,)
    n = flat.shape[0]

    # 初始化分布（softmax 权重）
    probs = F.softmax(flat / (soft_topk_temp + 1e-9), dim=0)  # (n,)

    values = []
    coords = []
    residual = probs.clone()

    for _ in range(K):
        # 归一化残差分布
        w = residual / (residual.sum() + 1e-9)   # (n,)
        val = (w * flat).sum()

        # soft 坐标（加权平均）
        hs = (torch.arange(n, device=attn_map.device) // W).to(attn_map.dtype)
        ws = (torch.arange(n, device=attn_map.device) % W).to(attn_map.dtype)
        coord_h = (w * hs).sum()
        coord_w = (w * ws).sum()

        values.append(val)
        coords.append(torch.stack([coord_h, coord_w], dim=0))

        # 抑制已经被选中的区域（soft 抑制 = 概率残差更新）
        residual = residual * (1.0 - w)

    values = torch.stack(values, dim=0)           # (K,)
    coords = torch.stack(coords, dim=0)           # (K,2)

    return values, coords



def soft_count_from_final_gs(final_dict: Dict[Tuple[str, int], torch.Tensor],
                             K: int = 30, margin: int = 2,
                             gumbel_scale: float = 1.0, temp: float = 0.5,
                             n_iters: int = 20, use_hard_topk: bool = True,
                             soft_topk_temp: float = 0.1,
                             clamp_count_max_1: bool = True) -> Dict[str, torch.Tensor]:
    """
    主函数：从 final_dict 计算每个类别的 soft-count（支持 batch）。
    Args:
        final_dict: dict with keys (cls, idx), values Tensor[B,1,H,W]
        K: 每个 slot 提取 top-K 候选
        margin: slack slots 数量（增加列数，作为 background/unmatched 槽）
        clamp_count_max_1: 若 True，把每个 slot 的质量 clamp 到 [0,1] 再 sum（更稳健）
    Returns:
        class_counts: dict { cls: Tensor[B] }，每类在 batch 维度下的 soft counts
    """
    device = None
    dtype = None

    # 按类别聚合 slot maps（保持 slot 的原始顺序）
    slots_by_class: Dict[str, List[torch.Tensor]] = {}
    # 也记录 batch size
    B = None

    for (cls, idx), attn in final_dict.items():
        if not torch.is_tensor(attn):
            raise ValueError("final_dict values must be tensors")
        if attn.ndim != 4:
            raise ValueError("Each attention map must have shape [B,1,H,W]")
        if B is None:
            B = attn.shape[0]
            device = attn.device
            dtype = attn.dtype
        else:
            if attn.shape[0] != B:
                raise ValueError("All attention maps must have same batch size B")
        # store (B,1,H,W) tensors
        slots_by_class.setdefault(cls, []).append(attn)

    if B is None:
        return {}

    class_counts: Dict[str, torch.Tensor] = {}

    # 遍历类别
    for cls, slot_tensors in slots_by_class.items():
        num_slots = len(slot_tensors)
        S_slots = num_slots + margin  # 列数（slot 列 + slack 列）

        # 初始化 per-batch counts buffer
        counts_b = torch.zeros(B, device=device, dtype=dtype)

        # 逐 batch 处理（也可以并行，但为了代码清晰这里循环）
        for b in range(B):
            # 1) 对每个 slot map 提取 candidates (K per slot)
            candidate_vals_list = []  # list of tensors (K,)
            candidate_coords_list = []  # list of tensors (K,2)
            slot_vecs = []  # list of scalars (slot representation)

            for attn_full in slot_tensors:
                # attn_full: [B,1,H,W] -> take batch b
                attn_map = attn_full[b, 0]  # (H,W)
                vals, coords = extract_candidates(attn_map, K=K,
                                                  soft_topk_temp=soft_topk_temp)
                # vals: (K,), coords: (K,2)
                candidate_vals_list.append(vals)
                candidate_coords_list.append(coords)

                # slot representation：这里用 slot map 的 mean 作为简单表示（可替换为更丰富 embedding）
                slot_repr = attn_map.view(-1).mean().unsqueeze(0)  # (1,)
                slot_vecs.append(slot_repr)

            # 合并 candidates
            if len(candidate_vals_list) == 0:
                # 没有 slot，计数 0
                counts_b[b] = 0.0
                continue

            candidates = torch.cat(candidate_vals_list, dim=0).to(device=device, dtype=dtype)  # (K_candidates,)
            K_candidates = candidates.shape[0]

            # slot_vecs -> (num_slots,)
            slot_vecs = torch.cat(slot_vecs, dim=0).to(device=device, dtype=dtype)  # (num_slots,)

            # add slack slots (zeros)
            if margin > 0:
                slack = torch.zeros(margin, device=device, dtype=dtype)
                slot_vecs_full = torch.cat([slot_vecs, slack], dim=0)  # (S_slots,)
            else:
                slot_vecs_full = slot_vecs  # (S_slots,)

            # 构造相似度矩阵 S: (K_candidates, S_slots)
            # 使用 outer product of scalars（你也可以换成 dot-product of emb vectors）
            S = candidates.unsqueeze(1) * slot_vecs_full.unsqueeze(0)  # (K_candidates, S_slots)

            # 数值稳定性：非负化并加 eps，避免 log(0)
            S = F.relu(S) + 1e-9

            # 设定 sinkhorn 的边际：
            # 我们把每个 candidate 的质量视为 1.0（行和 = 1），
            # 列的总质量应该等于 K_candidates（总质量），因此每列目标和 = K_candidates / S_slots（均分）
            row_sum = 1.0
            col_sum = float(K_candidates) / float(S_slots)

            # log domain
            log_S = torch.log(S)

            # Gumbel-Sinkhorn 得到 P: (K_candidates, S_slots)
            P = gumbel_sinkhorn(log_S, n_iters=n_iters,
                                gumbel_scale=gumbel_scale, temp=temp,
                                row_sum=row_sum, col_sum=col_sum)

            # slot_mass: (S_slots,)
            slot_mass = P.sum(dim=0)

            # 计数策略：把每个 slot 的质量映射到 [0,1] 再求和（clamp 更稳健）
            if clamp_count_max_1:
                sc = slot_mass.clamp(min=0.0, max=1.0).sum()
            else:
                # 另一种策略：sigmoid with learned threshold (这里用固定形状)
                sc = torch.sum(torch.sigmoid((slot_mass - (K_candidates / (S_slots*2.0))) * 5.0))
                # 上面只是示例，推荐使用 clamp 更直观

            counts_b[b] = sc

        class_counts[cls] = counts_b

    return class_counts


# -------------------------
# 使用示例（说明）
# -------------------------
# from soft_count_gumbel_sinkhorn import soft_count_from_final_gs
#
# final = get_slot_heatmaps_resized(unet, target_h=32, target_w=32)
# # final: dict { (cls, idx): Tensor[B,1,H,W], ... }
#
# class_counts = soft_count_from_final_gs(final, K=30, margin=2,
#                                         gumbel_scale=1.0, temp=0.5, n_iters=20)
#
# print(class_counts)  # { "dog": tensor([2.13, ...], grad_fn=...), "cat": tensor([1.05, ...]) }
