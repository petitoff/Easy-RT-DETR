from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, inverse_sigmoid


@dataclass
class DenoisingOutput:
    query_class: torch.Tensor
    query_bbox_unact: torch.Tensor
    attn_mask: torch.Tensor
    dn_meta: dict[str, object]


def get_contrastive_denoising_training_group(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: nn.Embedding,
    num_denoising: int = 100,
    label_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
) -> DenoisingOutput | None:
    if num_denoising <= 0:
        return None

    num_gts = [int(target["labels"].numel()) for target in targets]
    max_gt_num = max(num_gts, default=0)
    if max_gt_num == 0:
        return None

    num_group = max(num_denoising // max_gt_num, 1)
    batch_size = len(targets)
    device = class_embed.weight.device
    dtype = class_embed.weight.dtype

    input_query_class = torch.full((batch_size, max_gt_num), num_classes, dtype=torch.long, device=device)
    input_query_bbox = torch.zeros((batch_size, max_gt_num, 4), dtype=dtype, device=device)
    pad_gt_mask = torch.zeros((batch_size, max_gt_num), dtype=torch.bool, device=device)
    for batch_index, target in enumerate(targets):
        num_gt = num_gts[batch_index]
        if num_gt == 0:
            continue
        input_query_class[batch_index, :num_gt] = target["labels"].to(device=device, dtype=torch.long)
        input_query_bbox[batch_index, :num_gt] = target["boxes"].to(device=device, dtype=dtype)
        pad_gt_mask[batch_index, :num_gt] = True

    input_query_class = input_query_class.repeat(1, 2 * num_group)
    input_query_bbox = input_query_bbox.repeat(1, 2 * num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)

    negative_gt_mask = torch.zeros((batch_size, max_gt_num * 2, 1), dtype=dtype, device=device)
    negative_gt_mask[:, max_gt_num:, :] = 1.0
    negative_gt_mask = negative_gt_mask.repeat(1, num_group, 1)
    positive_gt_mask = (1.0 - negative_gt_mask).squeeze(-1).bool() & pad_gt_mask

    dn_positive_idx: list[torch.Tensor] = []
    for batch_index, num_gt in enumerate(num_gts):
        if num_gt == 0:
            dn_positive_idx.append(torch.empty(0, dtype=torch.long, device=device))
            continue
        positive = torch.nonzero(positive_gt_mask[batch_index], as_tuple=False).squeeze(-1)
        dn_positive_idx.append(positive)

    total_num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        flat_class = input_query_class.reshape(-1)
        flat_mask = pad_gt_mask.reshape(-1)
        chosen = (torch.rand(flat_class.shape, device=device) < (label_noise_ratio * 0.5)) & flat_mask
        chosen_idx = torch.nonzero(chosen, as_tuple=False).squeeze(-1)
        if chosen_idx.numel() > 0:
            new_label = torch.randint(0, num_classes, (chosen_idx.numel(),), dtype=flat_class.dtype, device=device)
            flat_class[chosen_idx] = new_label
        input_query_class = flat_class.view(batch_size, total_num_denoising)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = input_query_bbox[..., 2:].repeat(1, 1, 2) * 0.5 * box_noise_scale
        rand_sign = torch.randint(0, 2, input_query_bbox.shape, device=device, dtype=torch.int64).to(dtype=dtype) * 2.0 - 1.0
        rand_part = torch.rand(input_query_bbox.shape, device=device, dtype=dtype)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1.0 - negative_gt_mask)
        known_bbox = (known_bbox + rand_part * rand_sign * diff).clamp(min=0.0, max=1.0)
        input_query_bbox = inverse_sigmoid(box_xyxy_to_cxcywh(known_bbox))

    null_embed = torch.zeros((1, class_embed.embedding_dim), device=device, dtype=dtype)
    class_table = torch.cat((class_embed.weight, null_embed), dim=0)
    input_query_class = class_table[input_query_class]

    tgt_size = total_num_denoising + num_queries
    attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool, device=device)
    attn_mask[total_num_denoising:, :total_num_denoising] = True
    for group_index in range(num_group):
        start = max_gt_num * 2 * group_index
        end = max_gt_num * 2 * (group_index + 1)
        if start > 0:
            attn_mask[start:end, :start] = True
        if end < total_num_denoising:
            attn_mask[start:end, end:total_num_denoising] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [total_num_denoising, num_queries],
    }
    return DenoisingOutput(
        query_class=input_query_class,
        query_bbox_unact=input_query_bbox,
        attn_mask=attn_mask,
        dn_meta=dn_meta,
    )
