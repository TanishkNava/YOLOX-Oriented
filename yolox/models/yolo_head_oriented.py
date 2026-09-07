#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from yolox.utils import cxcywh2xyxy, meshgrid, visualize_assign

from .losses import KLDLoss, pairwise_kld_similarity
from .network_blocks import BaseConv, DWConv, TaskDecomposition
from .yolo_head import YOLOXHead


class YOLOXHeadOriented(YOLOXHead):
    """YOLOX head for boxes represented by ``(cx, cy, w, h, angle_degrees)``.

    The four axis-aligned regression channels and all existing branches retain
    their upstream names and shapes.  Angle prediction is deliberately kept in
    a separate branch so standard YOLOX weights can be loaded with only the
    expected missing ``angle_preds`` parameters.
    """

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        task_decomposition=False,
        stacked_convs=2,
    ):
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=strides,
            in_channels=in_channels,
            act=act,
            depthwise=depthwise,
        )
        self.angle_prediction = True
        self.task_decomposition = task_decomposition
        if task_decomposition:
            self._build_task_decomposition(
                int(256 * width), stacked_convs, in_channels, act, depthwise
            )
        self.angle_preds = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in in_channels
            ]
        )
        self.kld_loss = KLDLoss(reduction="none")

    def _build_task_decomposition(
        self, channels, stacked_convs, in_channels, act, depthwise
    ):
        """Replace the two per-task 3x3 stacks with a shared stack plus
        layer-attention decomposition (Oriented-YOLOX section 3.3).

        Sharing the 3x3 trunk and giving each task a 1x1 projection is what
        removes the duplicated spatial convolutions, so the head gets cheaper
        even though it gains the attention parameters.
        """
        Conv = DWConv if depthwise else BaseConv
        self.stacked_convs = stacked_convs
        self.inter_convs = nn.ModuleList(
            nn.ModuleList(
                Conv(channels, channels, 3, stride=1, act=act)
                for _ in range(stacked_convs)
            )
            for _ in in_channels
        )
        self.cls_decomposition = nn.ModuleList(
            TaskDecomposition(channels, stacked_convs, act=act) for _ in in_channels
        )
        self.reg_decomposition = nn.ModuleList(
            TaskDecomposition(channels, stacked_convs, act=act) for _ in in_channels
        )
        # The replaced stacks must not linger as unused parameters, or they
        # would still be optimized and saved into every checkpoint.
        self.cls_convs = nn.ModuleList(nn.Identity() for _ in in_channels)
        self.reg_convs = nn.ModuleList(nn.Identity() for _ in in_channels)

    def _task_features(self, k, x):
        """Return ``(cls_feat, reg_feat)`` for level ``k`` from a stem output."""
        if not self.task_decomposition:
            return self.cls_convs[k](x), self.reg_convs[k](x)

        stacked = []
        for conv in self.inter_convs[k]:
            x = conv(x)
            stacked.append(x)
        stacked = torch.cat(stacked, dim=1)
        return self.cls_decomposition[k](stacked), self.reg_decomposition[k](stacked)

    def initialize_biases(self, prior_prob):
        super().initialize_biases(prior_prob)
        # A sigmoid cannot represent exactly zero; initialize near the
        # horizontal orientation instead of the unhelpful default of 90°.
        angle_prior = 0.01
        angle_bias = -torch.log(torch.tensor((1.0 - angle_prior) / angle_prior))
        for conv in self.angle_preds:
            conv.bias.data.fill_(float(angle_bias))

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stride_this_level, x) in enumerate(zip(self.strides, xin)):
            x = self.stems[k](x)
            cls_feat, reg_feat = self._task_features(k, x)

            cls_output = self.cls_preds[k](cls_feat)
            reg_output = self.reg_preds[k](reg_feat)
            angle_output = self.angle_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat(
                    [reg_output, angle_output, obj_output, cls_output], dim=1
                )
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.full(
                        (1, grid.shape[1]),
                        stride_this_level,
                        device=xin[0].device,
                        dtype=xin[0].dtype,
                    )
                )
                if self.use_l1:
                    batch_size, _, hsize, wsize = reg_output.shape
                    origin_preds.append(
                        reg_output.view(batch_size, 1, 4, hsize, wsize)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(batch_size, -1, 4)
                        .clone()
                    )
            else:
                output = torch.cat(
                    [
                        reg_output,
                        angle_output,
                        obj_output.sigmoid(),
                        cls_output.sigmoid(),
                    ],
                    dim=1,
                )
            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, dim=1),
                origin_preds,
                dtype=xin[0].dtype,
            )

        self.hw = [x.shape[-2:] for x in outputs]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].type())
        return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 6 + self.num_classes
        hsize, wsize = output.shape[-2:]

        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .type(dtype)
            )
            self.grids[k] = grid

        output = (
            output.view(batch_size, 1, n_ch, hsize, wsize)
            .permute(0, 1, 3, 4, 2)
            .reshape(batch_size, hsize * wsize, n_ch)
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        output[..., 4] = output[..., 4].sigmoid() * 180.0
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        return torch.cat(
            [
                (outputs[..., :2] + grids) * strides,
                torch.exp(outputs[..., 2:4]) * strides,
                outputs[..., 4:5].sigmoid() * 180.0,
                outputs[..., 5:],
            ],
            dim=-1,
        )

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        box_preds = outputs[:, :, :5]
        obj_preds = outputs[:, :, 5:6]
        cls_preds = outputs[:, :, 6:]

        nlabel = (labels[:, :, 1:5].sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, dim=1)
        y_shifts = torch.cat(y_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, dim=1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 5))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_boxes = labels[batch_idx, :num_gt, 1:6].clone()
                gt_boxes[:, 4].remainder_(180.0)
                gt_classes = labels[batch_idx, :num_gt, 0]
                try:
                    assignment = self.get_assignments(
                        batch_idx,
                        num_gt,
                        gt_boxes,
                        gt_classes,
                        box_preds[batch_idx],
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as error:
                    if "CUDA out of memory" not in str(error):
                        raise
                    logger.error(
                        "CUDA OOM during oriented label assignment; retrying on CPU."
                    )
                    torch.cuda.empty_cache()
                    assignment = self.get_assignments(
                        batch_idx,
                        num_gt,
                        gt_boxes,
                        gt_classes,
                        box_preds[batch_idx],
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        mode="cpu",
                    )

                (
                    gt_matched_classes,
                    fg_mask,
                    matched_similarity,
                    matched_gt_inds,
                    num_fg_img,
                ) = assignment
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * matched_similarity.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_boxes[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_boxes[matched_gt_inds, :4],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, dim=0)
        reg_targets = torch.cat(reg_targets, dim=0)
        obj_targets = torch.cat(obj_targets, dim=0)
        fg_masks = torch.cat(fg_masks, dim=0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, dim=0)

        num_fg = max(num_fg, 1)
        # Covariance inversion is intentionally kept in fp32 for AMP stability.
        loss_kld = self.kld_loss(
            box_preds.reshape(-1, 5)[fg_masks].float(),
            reg_targets.float(),
        ).sum() / num_fg
        loss_obj = self.bcewithlog_loss(
            obj_preds.reshape(-1, 1), obj_targets
        ).sum() / num_fg
        loss_cls = self.bcewithlog_loss(
            cls_preds.reshape(-1, self.num_classes)[fg_masks], cls_targets
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(
                origin_preds.reshape(-1, 4)[fg_masks], l1_targets
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_kld + loss_obj + loss_cls + loss_l1
        return (
            loss,
            reg_weight * loss_kld,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_boxes,
        gt_classes,
        box_preds,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        original_device = gt_boxes.device
        if mode == "cpu":
            gt_boxes = gt_boxes.cpu().float()
            box_preds = box_preds.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_boxes[:, :4], expanded_strides, x_shifts, y_shifts
        )
        candidate_boxes = box_preds[fg_mask]
        if mode == "cpu":
            cls_preds_ = cls_preds[batch_idx].cpu()[fg_mask]
            obj_preds_ = obj_preds[batch_idx].cpu()[fg_mask]
        else:
            cls_preds_ = cls_preds[batch_idx][fg_mask]
            obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_candidates = candidate_boxes.shape[0]

        similarities = pairwise_kld_similarity(
            gt_boxes.float(), candidate_boxes.float()
        )
        gt_cls = F.one_hot(
            gt_classes.to(torch.int64), self.num_classes
        ).float()
        similarity_loss = -torch.log(similarities.clamp_min(1e-8))

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_context = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=False)
        with autocast_context:
            class_scores = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            cls_cost = F.binary_cross_entropy(
                class_scores.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls.unsqueeze(1).repeat(1, num_candidates, 1),
                reduction="none",
            ).sum(-1)

        cost = (
            cls_cost
            + 3.0 * similarity_loss
            + float(1e6) * (~geometry_relation)
        )
        (
            num_fg,
            matched_classes,
            matched_similarity,
            matched_gt_inds,
        ) = self.simota_matching(
            cost, similarities, gt_classes, num_gt, fg_mask
        )

        if mode == "cpu":
            matched_classes = matched_classes.to(original_device)
            fg_mask = fg_mask.to(original_device)
            matched_similarity = matched_similarity.to(original_device)
            matched_gt_inds = matched_gt_inds.to(original_device)
        return (
            matched_classes,
            fg_mask,
            matched_similarity,
            matched_gt_inds,
            num_fg,
        )

    def simota_matching(
        self, cost, pairwise_similarity, gt_classes, num_gt, fg_mask
    ):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        n_candidate_k = min(10, pairwise_similarity.size(1))
        topk_similarity = torch.topk(
            pairwise_similarity, n_candidate_k, dim=1
        ).values
        dynamic_ks = torch.clamp(topk_similarity.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            pos_idx = torch.topk(
                cost[gt_idx],
                k=dynamic_ks[gt_idx].item(),
                largest=False,
            ).indices
            matching_matrix[gt_idx, pos_idx] = 1

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0).indices
            matching_matrix[:, multiple_match_mask] = 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        matched_classes = gt_classes[matched_gt_inds]
        matched_similarity = (
            matching_matrix * pairwise_similarity
        ).sum(0)[fg_mask_inboxes]
        return num_fg, matched_classes, matched_similarity, matched_gt_inds

    @torch.no_grad()
    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        """Visualize oriented SimOTA assignments using enclosing boxes."""
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        for k, (stride, feature) in enumerate(zip(self.strides, xin)):
            stem = self.stems[k](feature)
            cls_feat, reg_feat = self._task_features(k, stem)
            cls_output = self.cls_preds[k](cls_feat)
            output = torch.cat(
                [
                    self.reg_preds[k](reg_feat),
                    self.angle_preds[k](reg_feat),
                    self.obj_preds[k](reg_feat),
                    cls_output,
                ],
                dim=1,
            )
            output, grid = self.get_output_and_grid(output, k, stride, xin[0].type())
            outputs.append(output)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full(
                    (1, grid.shape[1]), stride,
                    device=xin[0].device, dtype=xin[0].dtype,
                )
            )

        outputs = torch.cat(outputs, dim=1)
        box_preds = outputs[:, :, :5]
        obj_preds = outputs[:, :, 5:6]
        cls_preds = outputs[:, :, 6:]
        x_shifts = torch.cat(x_shifts, dim=1)
        y_shifts = torch.cat(y_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)
        nlabel = (labels[:, :, 1:5].sum(dim=2) > 0).sum(dim=1)

        for batch_idx, (image, count, label) in enumerate(zip(imgs, nlabel, labels)):
            count = int(count)
            if count == 0:
                fg_mask = outputs.new_zeros(outputs.shape[1]).bool()
                matched_gt_inds = outputs.new_zeros(0, dtype=torch.long)
                gt_boxes = label.new_zeros((0, 5))
            else:
                gt_boxes = label[:count, 1:6]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(
                    batch_idx, count, gt_boxes, label[:count, 0],
                    box_preds[batch_idx], expanded_strides, x_shifts, y_shifts,
                    cls_preds, obj_preds,
                )
            image = image.permute(1, 2, 0).to(torch.uint8).cpu().numpy().copy()
            coords = torch.stack(
                [
                    ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                    ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ],
                dim=1,
            )
            boxes_xyxy = cxcywh2xyxy(gt_boxes[:, :4].clone())
            visualize_assign(
                image, boxes_xyxy, coords, matched_gt_inds,
                save_prefix + str(batch_idx) + ".png",
            )
