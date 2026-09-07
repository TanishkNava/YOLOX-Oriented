#!/usr/bin/env python3

"""Full Oriented-YOLOX-S: KLD baseline plus the three paper modules."""

import os

from exps.custom.yolox_s_oriented import Exp as OrientedBaselineExp


class Exp(OrientedBaselineExp):
    def __init__(self):
        super().__init__()
        self.use_multiscale_dilated = True
        self.use_improved_eca = True
        self.use_task_decomposition = True
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
