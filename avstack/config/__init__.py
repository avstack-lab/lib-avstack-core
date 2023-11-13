# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict
from .registry import Registry
from .root import AGENTS, ALGORITHMS, DATASETS, MODELS, PIPELINE, REFERENCE


__all__ = [
    "AGENTS",
    "ALGORITHMS",
    "Config",
    "ConfigDict",
    "DATASETS",
    "MODELS",
    "PIPELINE",
    "REFERENCE",
    "Registry",
]
