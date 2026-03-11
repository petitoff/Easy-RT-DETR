from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .auxiliary_head import AuxiliaryDenseHead
from .backbone import TorchvisionResNetBackbone
from .config import RTDETRv3Config
from .decoder import RTDETRDecoder, build_group_attention_mask
from .encoder import HybridEncoder
from .heads import DecoderHeadBundle
from .losses import AuxiliaryDenseCriterion, SetCriterion
from .matcher import HungarianMatcher
from .postprocess import RTDETRPostProcessor
from .queries import QuerySelection
from .utils import batch_images


class RTDETRv3(nn.Module):
    def __init__(self, config: RTDETRv3Config) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("pixel_mean", torch.tensor(config.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(config.image_std).view(1, 3, 1, 1), persistent=False)
        self.backbone = TorchvisionResNetBackbone(config.backbone_name, pretrained=config.pretrained_backbone)
        self.encoder = HybridEncoder(self.backbone.out_channels, config.hidden_dim)
        self.query_selection = QuerySelection(
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            num_o2o_groups=config.num_o2o_groups,
            o2m_branch=config.o2m_branch,
            num_queries_o2m=config.num_queries_o2m,
            o2m_duplicates=config.o2m_duplicates,
            feat_strides=config.feat_strides,
            learnt_init_query=config.learnt_init_query,
            num_denoising=config.num_denoising,
            label_noise_ratio=config.label_noise_ratio,
            box_noise_scale=config.box_noise_scale,
            anchor_eps=config.anchor_eps,
        )
        self.decoder = RTDETRDecoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_levels=config.num_feature_levels,
            num_points=config.num_decoder_points,
            query_pos_head_inv_sig=config.query_pos_head_inv_sig,
        )
        self.decoder_heads = DecoderHeadBundle(config.hidden_dim, config.num_classes, config.num_decoder_layers)
        self.auxiliary_head = AuxiliaryDenseHead(
            in_channels=config.hidden_dim,
            num_classes=config.num_classes,
            fpn_strides=config.feat_strides,
            reg_max=config.aux_reg_max,
            grid_cell_scale=config.aux_grid_cell_scale,
            grid_cell_offset=config.aux_grid_cell_offset,
        )
        self.matcher = HungarianMatcher(
            cls_cost=config.cls_loss_weight,
            bbox_cost=config.bbox_loss_weight,
            giou_cost=config.giou_loss_weight,
        )
        self.criterion = SetCriterion(
            num_classes=config.num_classes,
            matcher=self.matcher,
            cls_weight=config.cls_loss_weight,
            bbox_weight=config.bbox_loss_weight,
            giou_weight=config.giou_loss_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
        )
        self.auxiliary_criterion = AuxiliaryDenseCriterion(
            num_classes=config.num_classes,
            reg_max=config.aux_reg_max,
            static_assigner_topk=config.aux_static_assigner_topk,
            task_aligned_topk=config.aux_task_aligned_topk,
            task_aligned_alpha=config.aux_task_aligned_alpha,
            task_aligned_beta=config.aux_task_aligned_beta,
            static_assigner_epoch=config.aux_static_assigner_epoch,
            use_varifocal_loss=config.aux_use_varifocal_loss,
            loss_weight_class=config.aux_loss_weight_class,
            loss_weight_iou=config.aux_loss_weight_iou,
            loss_weight_dfl=config.aux_loss_weight_dfl,
        )
        self.postprocessor = RTDETRPostProcessor(
            topk=config.inference_topk,
            score_threshold=config.inference_score_threshold,
            nms_threshold=config.inference_nms_threshold,
        )

    def forward(
        self,
        images: torch.Tensor | list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        epoch: int | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        images = batch_images(images)
        images = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images)
        encoder_output = self.encoder(features)
        query_selection = self.query_selection(encoder_output.memory, encoder_output.spatial_shapes, targets=targets)
        attn_mask = build_group_attention_mask(
            query_selection.groups,
            keep_prob=self.config.perturbation_keep_prob,
            device=images.device,
            training=self.training,
        )
        dec_boxes, dec_logits = self.decoder(
            target=query_selection.target,
            reference_points_unact=query_selection.reference_points_unact,
            memory=query_selection.memory,
            spatial_shapes=encoder_output.spatial_shapes,
            class_heads=self.decoder_heads.class_heads,
            box_heads=self.decoder_heads.box_heads,
            attn_mask=attn_mask,
        )

        if not self.training:
            image_sizes = torch.as_tensor(images.shape[-2:], device=images.device).repeat(images.size(0), 1)
            return self.postprocessor(dec_logits[-1], dec_boxes[-1], image_sizes=image_sizes)

        if targets is None:
            raise ValueError("Targets are required while training.")

        losses = self._compute_detection_losses(dec_boxes, dec_logits, query_selection, targets)
        aux_outputs = self.auxiliary_head(encoder_output.features)
        aux_losses = self.auxiliary_criterion(
            aux_outputs,
            targets,
            image_size=tuple(images.shape[-2:]),
            epoch=epoch,
        )
        for key, value in aux_losses.items():
            losses[f"{key}_aux_o2m"] = value * self.config.auxiliary_loss_weight

        total_loss = sum(losses.values())
        losses["loss"] = total_loss
        return losses

    def _compute_detection_losses(
        self,
        dec_boxes: torch.Tensor,
        dec_logits: torch.Tensor,
        selection,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        dec_start = 0
        enc_start = 0
        normal_group_count = 0
        for group in selection.groups:
            dec_end = dec_start + group.count
            enc_end = enc_start + int(group.matching_count or group.count)
            group_dec_boxes = dec_boxes[:, :, dec_start:dec_end]
            group_dec_logits = dec_logits[:, :, dec_start:dec_end]
            dn_count = group.dn_count

            if dn_count > 0:
                dn_boxes = group_dec_boxes[-1, :, :dn_count]
                dn_logits = group_dec_logits[-1, :, :dn_count]
                match_boxes = group_dec_boxes[-1, :, dn_count:]
                match_logits = group_dec_logits[-1, :, dn_count:]
                dn_aux_outputs = [
                    {"pred_boxes": group_dec_boxes[layer_id, :, :dn_count], "pred_logits": group_dec_logits[layer_id, :, :dn_count]}
                    for layer_id in range(group_dec_boxes.size(0) - 1)
                ]
            else:
                dn_boxes = None
                dn_logits = None
                match_boxes = group_dec_boxes[-1]
                match_logits = group_dec_logits[-1]
                dn_aux_outputs = []

            group_outputs = {
                "pred_boxes": match_boxes,
                "pred_logits": match_logits,
            }
            aux_outputs = [
                {"pred_boxes": selection.encoder_boxes[:, enc_start:enc_end], "pred_logits": selection.encoder_logits[:, enc_start:enc_end]}
            ]
            for layer_id in range(group_dec_boxes.size(0) - 1):
                aux_outputs.append(
                    {
                        "pred_boxes": group_dec_boxes[layer_id, :, dn_count:],
                        "pred_logits": group_dec_logits[layer_id, :, dn_count:],
                    }
                )

            group_loss = self.criterion(
                group_outputs["pred_logits"],
                group_outputs["pred_boxes"],
                targets,
                aux_outputs=aux_outputs,
                o2m_duplicates=group.o2m_duplicates,
                dn_outputs=(
                    {
                        "pred_boxes": dn_boxes,
                        "pred_logits": dn_logits,
                        "aux_outputs": dn_aux_outputs,
                    }
                    if dn_boxes is not None and dn_logits is not None
                    else None
                ),
                dn_meta=group.dn_meta,
            )

            weight = self.config.o2m_loss_weight if group.o2m_duplicates > 1 else self.config.o2o_loss_weight
            prefix = group.name
            for key, value in group_loss.items():
                losses[f"{key}_{prefix}"] = value * weight
            if group.o2m_duplicates == 1 and not group.training_only:
                normal_group_count += 1
            dec_start = dec_end
            enc_start = enc_end

        if normal_group_count > 1:
            for key in list(losses.keys()):
                if key.endswith("o2m"):
                    continue
                if "_o2o_" in key:
                    losses[key] = losses[key] / normal_group_count
        return losses

    def extra_repr(self) -> str:
        return f"config={asdict(self.config)}"
