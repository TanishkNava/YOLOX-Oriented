#!/usr/bin/env python3

import torch
import torch.distributed as dist
import torch.nn as nn

from .yolox_base import Exp as YOLOXBaseExp

__all__ = ["Exp"]


class Exp(YOLOXBaseExp):
    """Base experiment for cxcywha YOLOX models."""

    def __init__(self):
        super().__init__()
        self.oriented = True
        self.angle_period = 180.0
        self.train_name = "train2017"
        self.val_name = "val2017"
        self.test_name = "test2017"

        # Oriented-YOLOX paper modules, off by default so the plain KLD
        # baseline stays reproducible and each one can be ablated alone.
        self.use_multiscale_dilated = False
        self.use_improved_eca = False
        self.use_task_decomposition = False
        self.head_stacked_convs = 2

    def get_model(self):
        from yolox.models import (
            YOLOX,
            YOLOPAFPN,
            OrientedYOLOPAFPN,
            YOLOXHeadOriented,
        )

        def init_yolo(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            if self.use_multiscale_dilated or self.use_improved_eca:
                backbone = OrientedYOLOPAFPN(
                    self.depth, self.width, in_channels=in_channels, act=self.act,
                    multiscale_dilated=self.use_multiscale_dilated,
                    improved_eca=self.use_improved_eca,
                )
            else:
                backbone = YOLOPAFPN(
                    self.depth, self.width, in_channels=in_channels, act=self.act
                )
            head = YOLOXHeadOriented(
                self.num_classes, self.width,
                in_channels=in_channels, act=self.act,
                task_decomposition=self.use_task_decomposition,
                stacked_convs=self.head_stacked_convs,
            )
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import OrientedCOCODataset, OrientedTrainTransform
        return OrientedCOCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=OrientedTrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False,
                        cache_img=None):
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            OrientedTrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        if self.dataset is None:
            with wait_for_the_master():
                self.dataset = self.get_dataset(
                    cache=cache_img is not None,
                    cache_type=cache_img or "ram",
                )
        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=OrientedTrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        if is_distributed:
            batch_size //= dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )
        return DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_reset_seed,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import OrientedCOCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        return OrientedCOCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.test_name if testdev else self.val_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False,
                      legacy=False):
        from yolox.evaluators import OrientedCOCOEvaluator
        return OrientedCOCOEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1] *= scale_x
            targets[..., 2] *= scale_y
            targets[..., 3] *= scale_x
            targets[..., 4] *= scale_y
        return inputs, targets
