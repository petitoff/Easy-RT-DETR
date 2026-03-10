from __future__ import annotations

import torch

from .utils import box_cxcywh_to_xyxy, nms


class RTDETRPostProcessor:
    def __init__(self, topk: int = 300, score_threshold: float = 0.05, nms_threshold: float = 0.5) -> None:
        self.topk = topk
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def __call__(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        scores = pred_logits.sigmoid()
        batch_size, num_queries, num_classes = scores.shape
        topk = min(self.topk, num_queries * num_classes)
        scores, indices = scores.flatten(1).topk(topk, dim=1)
        labels = indices % num_classes
        query_indices = indices // num_classes
        boxes = pred_boxes.gather(1, query_indices.unsqueeze(-1).expand(batch_size, topk, 4))
        boxes = box_cxcywh_to_xyxy(boxes)

        outputs = []
        for batch_index in range(batch_size):
            box = boxes[batch_index]
            score = scores[batch_index]
            label = labels[batch_index]
            if image_sizes is not None:
                height, width = image_sizes[batch_index]
                scale = torch.tensor((width, height, width, height), device=box.device, dtype=box.dtype)
                box = box * scale
                box[:, 0::2] = box[:, 0::2].clamp(min=0, max=width)
                box[:, 1::2] = box[:, 1::2].clamp(min=0, max=height)

            keep_mask = score >= self.score_threshold
            box = box[keep_mask]
            score = score[keep_mask]
            label = label[keep_mask]

            if box.numel() > 0:
                kept_indices = []
                for class_id in label.unique():
                    class_mask = label == class_id
                    class_keep = nms(box[class_mask], score[class_mask], self.nms_threshold)
                    class_indices = torch.where(class_mask)[0][class_keep]
                    kept_indices.append(class_indices)
                if kept_indices:
                    kept_indices = torch.cat(kept_indices)
                    kept_indices = kept_indices[score[kept_indices].argsort(descending=True)]
                    box = box[kept_indices]
                    score = score[kept_indices]
                    label = label[kept_indices]

            outputs.append({"scores": score, "labels": label, "boxes": box})
        return outputs
