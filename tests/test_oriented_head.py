#!/usr/bin/env python3

import torch

from yolox.models import YOLOXHeadOriented


def _features(requires_grad=False):
    return [
        torch.randn(2, 8, 4, 4, requires_grad=requires_grad),
        torch.randn(2, 16, 2, 2, requires_grad=requires_grad),
        torch.randn(2, 32, 1, 1, requires_grad=requires_grad),
    ]


def test_oriented_head_synthetic_forward_backward():
    torch.manual_seed(0)
    head = YOLOXHeadOriented(
        num_classes=3,
        width=0.25,
        in_channels=[32, 64, 128],
        strides=[8, 16, 32],
    )

    # [class, cx, cy, w, h, angle_degrees]
    labels = torch.zeros(2, 2, 6)
    labels[0, 0] = torch.tensor([1.0, 16.0, 16.0, 12.0, 6.0, 175.0])
    images = torch.randn(2, 3, 32, 32)

    head.train()
    losses = head(_features(requires_grad=True), labels, images)
    total_loss = losses[0]
    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    total_loss.backward()

    angle_grads = [layer.weight.grad for layer in head.angle_preds]
    assert all(grad is not None for grad in angle_grads)
    assert all(torch.isfinite(grad).all() for grad in angle_grads)

    head.eval()
    with torch.no_grad():
        predictions = head(_features())
    assert predictions.shape == (2, 21, 6 + head.num_classes)
    assert torch.all(predictions[..., 4] >= 0.0)
    assert torch.all(predictions[..., 4] < 180.0)
