from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from yolox.evaluators.oriented_coco_evaluator import OrientedCOCOEvaluator
from yolox.utils.boxes import (
    format_oriented_detections,
    oriented_postprocess,
    rotated_iou,
    rotated_nms,
)


def test_oriented_postprocess_layout_and_class_aware_nms():
    predictions = torch.tensor([[
        [50, 50, 40, 10, 30, 0.9, 0.9, 0.1],
        [50, 50, 40, 10, 30, 0.8, 0.8, 0.2],
        [50, 50, 40, 10, 30, 0.9, 0.1, 0.9],
    ]], dtype=torch.float32)

    class_aware = oriented_postprocess(
        predictions.clone(), 2, 0.1, 0.5, class_agnostic=False
    )[0]
    agnostic = oriented_postprocess(
        predictions.clone(), 2, 0.1, 0.5, class_agnostic=True
    )[0]

    assert class_aware.shape == (2, 8)
    assert agnostic.shape == (1, 8)
    assert class_aware[:, 7].tolist() == [0.0, 1.0]
    assert class_aware[0, :5].tolist() == predictions[0, 0, :5].tolist()
    public = format_oriented_detections(class_aware)
    assert public.shape == (2, 7)
    assert public[0, 5].item() == pytest.approx(0.81)


def test_rotated_nms_fallback_is_deterministic(monkeypatch):
    boxes = np.asarray([
        [0, 0, 20, 4, 0],
        [0, 0, 20, 4, 0],
        [0, 0, 20, 4, 90],
    ], dtype=np.float32)
    scores = np.asarray([0.8, 0.8, 0.7], dtype=np.float32)

    def unavailable(*args, **kwargs):
        raise cv2.error("forced fallback")

    monkeypatch.setattr(cv2.dnn, "NMSBoxesRotated", unavailable)
    assert rotated_nms(boxes, scores, 0.5) == [0, 2]
    assert rotated_iou(boxes[0], boxes[1]) == pytest.approx(1.0)
    assert rotated_iou(boxes[0], boxes[2]) < 0.5


def test_oriented_evaluator_perfect_ap_and_angle_mae():
    coco = SimpleNamespace(dataset={
        "categories": [{"id": 3, "name": "ship"}],
        "annotations": [{
            "id": 1, "image_id": 7, "category_id": 3,
            "bbox": [30, 45, 40, 10],
            "rbbox": [50, 50, 40, 10, 170],
            "area": 400, "iscrowd": 0,
        }],
    })
    dataset = SimpleNamespace(coco=coco, class_ids=[3])
    dataloader = SimpleNamespace(dataset=dataset, batch_size=1)
    evaluator = OrientedCOCOEvaluator(
        dataloader, (640, 640), 0.01, 0.5, 1
    )
    detections = [{
        "image_id": 7, "category_id": 3,
        "bbox": [50, 50, 40, 10, -10],
        "score": 0.99,
    }]

    ap50_95, ap50, info = evaluator.evaluate_prediction(
        detections, torch.tensor([0.0, 0.0, 1.0])
    )
    assert ap50_95 == pytest.approx(1.0)
    assert ap50 == pytest.approx(1.0)
    assert "ship=1.0000" in info
    assert "0.000 deg" in info
