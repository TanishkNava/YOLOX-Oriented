#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch.nn as nn

from .network_blocks import ImprovedECA, MultiScaleDilatedConv
from .yolo_pafpn import YOLOPAFPN

__all__ = ["OrientedYOLOPAFPN"]


class OrientedYOLOPAFPN(YOLOPAFPN):
    """YOLOX PAFPN carrying the Oriented-YOLOX neck modules.

    The multi-scale dilated convolution is applied to the highest resolution
    backbone feature (C3) before fusion, which is where the paper's ablation
    found it useful; the enlarged receptive field there is also what small
    rotated objects need, since they only occupy a few stride-8 cells.  The
    improved ECA blocks re-weight each fused PAN output.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        multiscale_dilated=True,
        improved_eca=True,
    ):
        super().__init__(
            depth=depth,
            width=width,
            in_features=in_features,
            in_channels=in_channels,
            depthwise=depthwise,
            act=act,
        )
        self.dilated_c3 = (
            MultiScaleDilatedConv(int(in_channels[0] * width), act=act)
            if multiscale_dilated
            else None
        )
        self.eca_blocks = (
            nn.ModuleList(
                [ImprovedECA(int(channels * width)) for channels in in_channels]
            )
            if improved_eca
            else None
        )

    def forward(self, input):
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        if self.dilated_c3 is not None:
            features[0] = self.dilated_c3(features[0])

        outputs = self.fuse_features(features)

        if self.eca_blocks is not None:
            outputs = tuple(
                block(feature) for block, feature in zip(self.eca_blocks, outputs)
            )
        return outputs
