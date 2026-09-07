#!/usr/bin/env python3

import os

from yolox.exp.oriented_yolox_base import Exp as OrientedExp


class Exp(OrientedExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 6
        self.data_dir = "/home/navavision/Tanishk!Workspace/rotational data/oriented_coco"

        # Preserve more detail from the 2560x1440 source images for small curtains.
        self.input_size = (960, 960)
        self.test_size = (960, 960)
        self.multiscale_range = 3

        self.max_epoch = 150
        self.warmup_epochs = 3
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.save_history_ckpt = False
        self.data_num_workers = 8
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
