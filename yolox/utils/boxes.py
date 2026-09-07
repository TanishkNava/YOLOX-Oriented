#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import cv2
import torch
import torchvision

from .oriented_boxes import canonicalize_obbs, oriented_box_iou

__all__ = [
    "filter_box",
    "postprocess",
    "oriented_postprocess",
    "format_oriented_detections",
    "rotated_iou",
    "rotated_nms",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
]


def _as_rotated_rect(box):
    """Convert cxcywha values to OpenCV's RotatedRect representation."""
    return (
        (float(box[0]), float(box[1])),
        (max(float(box[2]), 0.0), max(float(box[3]), 0.0)),
        float(box[4]),
    )


def rotated_iou(box1, box2):
    """Return the exact intersection-over-union of two cxcywha boxes."""
    return oriented_box_iou(box1, box2)


def _rotated_nms_fallback(boxes, scores, iou_threshold):
    """Deterministic greedy NMS using OpenCV's exact rotated intersection."""
    order = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), i))
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            index for index in order
            if rotated_iou(boxes[current], boxes[index]) <= iou_threshold
        ]
    return keep


def rotated_nms(boxes, scores, iou_threshold):
    """Run deterministic OpenCV rotated NMS and return original box indices."""
    if len(boxes) == 0:
        return []

    boxes = canonicalize_obbs(np.asarray(boxes, dtype=np.float32))
    scores = np.asarray(scores, dtype=np.float32)
    if np.max(scores) <= 0.0:
        return _rotated_nms_fallback(boxes, scores, iou_threshold)
    order = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), i))
    ordered_boxes = [_as_rotated_rect(boxes[i]) for i in order]
    ordered_scores = [float(scores[i]) for i in order]
    try:
        indices = cv2.dnn.NMSBoxesRotated(
            ordered_boxes, ordered_scores, 0.0, float(iou_threshold)
        )
        if indices is None:
            return []
        flat_indices = sorted(np.asarray(indices).reshape(-1).tolist())
        return [order[int(i)] for i in flat_indices]
    except (AttributeError, cv2.error, TypeError):
        return _rotated_nms_fallback(boxes, scores, iou_threshold)


def oriented_postprocess(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.45,
    class_agnostic=False
):
    """Postprocess [cx,cy,w,h,angle,obj,classes...] oriented predictions.

    Each retained detection is returned as
    [cx,cy,w,h,angle,obj_conf,class_conf,class_id].
    """
    output = [None for _ in range(len(prediction))]
    for image_index, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(
            image_pred[:, 6:6 + num_classes], 1, keepdim=True
        )
        confidence = image_pred[:, 5] * class_conf.squeeze(1)
        detections = torch.cat(
            (image_pred[:, :6], class_conf, class_pred.to(image_pred.dtype)), dim=1
        )
        detections = detections[confidence >= conf_thre]
        if not detections.size(0):
            continue

        boxes = detections[:, :5].detach().cpu().numpy()
        scores = (detections[:, 5] * detections[:, 6]).detach().cpu().numpy()
        if class_agnostic:
            keep = rotated_nms(boxes, scores, nms_thre)
        else:
            keep = []
            classes = detections[:, 7].detach().cpu().numpy().astype(np.int64)
            for class_id in sorted(np.unique(classes).tolist()):
                class_indices = np.flatnonzero(classes == class_id)
                class_keep = rotated_nms(
                    boxes[class_indices], scores[class_indices], nms_thre
                )
                keep.extend(class_indices[class_keep].tolist())
            keep.sort(key=lambda i: (-float(scores[i]), i))

        if keep:
            output[image_index] = detections[
                torch.as_tensor(keep, device=detections.device, dtype=torch.long)
            ]
    return output


def format_oriented_detections(detections):
    """Convert internal post-NMS rows to public ``cxcywha,score,class`` rows."""
    if detections is None:
        return None
    if detections.ndim != 2 or detections.shape[1] != 8:
        raise ValueError("oriented detections must have shape (N, 8)")
    score = (detections[:, 5] * detections[:, 6]).unsqueeze(1)
    return torch.cat((detections[:, :5], score, detections[:, 7:8]), dim=1)


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes
