# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict
from .registry import Registry
from .root import AGENTS, DATASETS, HOOKS, MODELS, PIPELINE, REFERENCE


__all__ = [
    "AGENTS",
    "Config",
    "ConfigDict",
    "DATASETS",
    "HOOKS",
    "MODELS",
    "PIPELINE",
    "REFERENCE",
    "Registry",
]
