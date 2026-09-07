#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


def _obb_to_gaussian(boxes, eps=1e-4):
    """Map ``[cx, cy, w, h, angle_degrees]`` boxes to Gaussian covariances."""
    if boxes.shape[-1] != 5:
        raise ValueError("oriented boxes must have a final dimension of 5")
    # Covariance construction and inversion are numerically unsafe in fp16:
    # adding eps to a nearly rank-one covariance can round back to a singular
    # matrix before autocast promotes torch.linalg.inv. Keep gradients while
    # doing the KLD geometry itself in fp32.
    if boxes.dtype in (torch.float16, torch.bfloat16):
        boxes = boxes.float()
    xy = boxes[..., :2]
    wh = boxes[..., 2:4].clamp_min(eps)
    angle = torch.deg2rad(boxes[..., 4])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
        boxes.shape[:-1] + (2, 2)
    )
    # A rectangle is represented by N(center, R diag(w^2/4, h^2/4) R^T),
    # matching the Gaussian-distance formulation used by MMRotate.
    diagonal = torch.diag_embed((wh * 0.5).square())
    covariance = rotation @ diagonal @ rotation.transpose(-1, -2)
    return xy, covariance


def gaussian_kld(pred, target, eps=1e-4):
    """Directional KLD between Gaussian OBB representations.

    Inputs use degrees.  The returned untransformed distance has the broadcast
    shape of the leading box dimensions and is clamped non-negative.
    """
    pred_xy, pred_cov = _obb_to_gaussian(pred, eps)
    target_xy, target_cov = _obb_to_gaussian(target, eps)
    eye = torch.eye(2, dtype=pred.dtype, device=pred.device)
    pred_jitter = (
        torch.diagonal(pred_cov, dim1=-2, dim2=-1).sum(-1) * eps
    ).clamp_min(eps)
    target_jitter = (
        torch.diagonal(target_cov, dim1=-2, dim2=-1).sum(-1) * eps
    ).clamp_min(eps)
    pred_cov = pred_cov + eye * pred_jitter.unsqueeze(-1).unsqueeze(-1)
    target_cov = target_cov + eye * target_jitter.unsqueeze(-1).unsqueeze(-1)
    pred_inv = torch.linalg.inv(pred_cov)
    delta = (pred_xy - target_xy).unsqueeze(-1)
    center = (delta.transpose(-1, -2) @ pred_inv @ delta).squeeze(-1).squeeze(-1)
    trace = torch.diagonal(pred_inv @ target_cov, dim1=-2, dim2=-1).sum(-1)
    _, pred_logdet = torch.linalg.slogdet(pred_cov)
    _, target_logdet = torch.linalg.slogdet(target_cov)
    return (0.5 * (center + trace + pred_logdet - target_logdet - 2.0)).clamp_min(0.0)


def gaussian_kld_similarity(pred, target, tau=1.0, eps=1e-4, sqrt=True):
    """Return the finite similarity used by transformed Gaussian KLD loss."""
    if tau < 1.0:
        raise ValueError("tau must be at least 1 to keep similarity in [0, 1]")
    distance = gaussian_kld(pred, target, eps=eps)
    if sqrt:
        distance = torch.where(
            distance > 0, torch.sqrt(distance.clamp_min(eps)), distance
        )
    return torch.reciprocal(float(tau) + torch.log1p(distance))


def pairwise_kld_similarity(targets, predictions, tau=1.0, eps=1e-4, sqrt=True):
    """Return an ``(N, M)`` KLD similarity matrix for two OBB sets."""
    if targets.ndim != 2 or targets.shape[-1] != 5:
        raise ValueError("targets must have shape (N, 5)")
    if predictions.ndim != 2 or predictions.shape[-1] != 5:
        raise ValueError("predictions must have shape (M, 5)")
    return gaussian_kld_similarity(
        predictions.unsqueeze(0),
        targets.unsqueeze(1),
        tau=tau,
        eps=eps,
        sqrt=sqrt,
    )


class GaussianKLDLoss(nn.Module):
    """Transformed Gaussian KLD loss for oriented boxes in degree format."""

    def __init__(
        self, reduction="none", tau=1.0, eps=1e-4, sqrt=True, loss_weight=1.0
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.reduction = reduction
        self.tau = tau
        self.eps = eps
        self.sqrt = sqrt
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = (1.0 - gaussian_kld_similarity(
            pred, target, tau=self.tau, eps=self.eps, sqrt=self.sqrt
        )) * self.loss_weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Backward-friendly short names for experiment code.
KLDLoss = GaussianKLDLoss
kld_similarity = gaussian_kld_similarity


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
