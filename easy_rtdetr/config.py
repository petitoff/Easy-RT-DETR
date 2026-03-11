from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(slots=True)
class RTDETRv3Config:
    num_classes: int = 80
    backbone_name: str = "resnet18"
    pretrained_backbone: bool = False
    hidden_dim: int = 256
    num_feature_levels: int = 3
    feat_strides: tuple[int, ...] = (8, 16, 32)
    hybrid_encoder_use_idx: tuple[int, ...] = (2,)
    hybrid_encoder_layers: int = 1
    hybrid_encoder_expansion: float = 1.0
    hybrid_encoder_depth_mult: float = 1.0
    hybrid_encoder_pe_temperature: float = 10000.0
    num_queries: int = 300
    num_decoder_layers: int = 6
    num_heads: int = 8
    num_decoder_points: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.0
    num_o2o_groups: int = 3
    perturbation_keep_prob: float = 0.9
    o2m_branch: bool = True
    num_queries_o2m: int = 300
    o2m_duplicates: int = 4
    learnt_init_query: bool = True
    query_pos_head_inv_sig: bool = False
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    auxiliary_topk: int = 9
    auxiliary_hidden_dim: int = 128
    aux_reg_max: int = 16
    aux_grid_cell_scale: float = 5.0
    aux_grid_cell_offset: float = 0.5
    aux_static_assigner_epoch: int = 30
    aux_static_assigner_topk: int = 9
    aux_use_varifocal_loss: bool = True
    aux_task_aligned_topk: int = 13
    aux_task_aligned_alpha: float = 1.0
    aux_task_aligned_beta: float = 6.0
    aux_loss_weight_class: float = 1.0
    aux_loss_weight_iou: float = 2.5
    aux_loss_weight_dfl: float = 0.5
    cls_loss_weight: float = 1.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    auxiliary_loss_weight: float = 1.0
    o2o_loss_weight: float = 1.0
    o2m_loss_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    inference_topk: int = 300
    inference_score_threshold: float = 0.18
    inference_nms_threshold: float = 0.25
    anchor_eps: float = 1e-2
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "RTDETRv3Config":
        presets = {
            "rtdetrv3_r18": cls(backbone_name="resnet18", dim_feedforward=1024),
            "rtdetrv3_r34": cls(backbone_name="resnet34", dim_feedforward=1024),
            "rtdetrv3_r50": cls(backbone_name="resnet50", dim_feedforward=1024),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")
        return replace(presets[name], **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
