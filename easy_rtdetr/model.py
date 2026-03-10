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
        )
        self.decoder = RTDETRDecoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_levels=config.num_feature_levels,
            num_points=config.num_decoder_points,
        )
        self.decoder_heads = DecoderHeadBundle(config.hidden_dim, config.num_classes, config.num_decoder_layers)
        self.auxiliary_head = AuxiliaryDenseHead(config.hidden_dim, config.auxiliary_hidden_dim, config.num_classes)
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
            topk=config.auxiliary_topk,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
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
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        images = batch_images(images)
        images = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images)
        encoder_output = self.encoder(features)
        query_selection = self.query_selection(encoder_output.memory, encoder_output.spatial_shapes)
        attn_mask = build_group_attention_mask(
            query_selection.groups,
            keep_prob=self.config.perturbation_keep_prob,
            device=images.device,
            training=self.training,
        )
        dec_boxes, dec_logits = self.decoder(
            target=query_selection.target,
            reference_points_unact=query_selection.reference_points_unact,
            memory=encoder_output.memory,
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
        aux_logits, aux_boxes, aux_locations = self.auxiliary_head(encoder_output.features)
        aux_losses = self.auxiliary_criterion(aux_logits, aux_boxes, aux_locations, targets)
        for key, value in aux_losses.items():
            losses[f"{key}_auxiliary"] = value * self.config.auxiliary_loss_weight

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
        start = 0
        o2o_group_count = 0
        for group in selection.groups:
            end = start + group.count
            group_outputs = {
                "pred_boxes": dec_boxes[-1, :, start:end],
                "pred_logits": dec_logits[-1, :, start:end],
            }
            aux_outputs = [
                {"pred_boxes": selection.encoder_boxes[:, start:end], "pred_logits": selection.encoder_logits[:, start:end]}
            ]
            for layer_id in range(dec_boxes.size(0) - 1):
                aux_outputs.append(
                    {
                        "pred_boxes": dec_boxes[layer_id, :, start:end],
                        "pred_logits": dec_logits[layer_id, :, start:end],
                    }
                )

            group_loss = self.criterion(
                group_outputs["pred_logits"],
                group_outputs["pred_boxes"],
                targets,
                aux_outputs=aux_outputs,
                o2m_duplicates=group.o2m_duplicates,
            )

            weight = self.config.o2m_loss_weight if group.o2m_duplicates > 1 else self.config.o2o_loss_weight
            prefix = group.name
            for key, value in group_loss.items():
                losses[f"{key}_{prefix}"] = value * weight
            if group.o2m_duplicates == 1:
                o2o_group_count += 1
            start = end

        if o2o_group_count > 1:
            for key in list(losses.keys()):
                if key.endswith("o2m"):
                    continue
                if "_o2o_" in key:
                    losses[key] = losses[key] / o2o_group_count
        return losses

    def extra_repr(self) -> str:
        return f"config={asdict(self.config)}"
