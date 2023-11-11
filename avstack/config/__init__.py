# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict
from .registry import Registry
from .root import ALGORITHMS, DATASETS


__all__ = ["ALGORITHMS", "Config", "ConfigDict", "DATASETS", "Registry"]
