from typing import Any

import numpy as np

from avstack.config import MODELS
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks


@MODELS.register_module()
class ConcaveHullLidarFOVEstimator(BaseModule):
    def __init__(
        self,
        concavity: int = 1,
        length_threshold: float = 4,
        max_height: float = np.inf,
        name="fov_estimator",
        *args,
        **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.concavity = concavity
        self.length_threshold = length_threshold
        self.max_height = max_height

    @apply_hooks
    def __call__(self, pc, in_global: bool = False, *args: Any, **kwds: Any) -> Any:
        fov = pc.concave_hull_bev(
            concavity=self.concavity,
            length_threshold=self.length_threshold,
            in_global=in_global,
            max_height=self.max_height,
        )
        return fov
