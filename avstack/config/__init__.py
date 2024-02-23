# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict
from .registry import Registry
from .root import AGENTS, DATASETS, GEOMETRY, HOOKS, MODELS, PIPELINE, REFERENCE


__all__ = [
    "AGENTS",
    "Config",
    "ConfigDict",
    "DATASETS",
    "GEOMETRY",
    "HOOKS",
    "MODELS",
    "PIPELINE",
    "REFERENCE",
    "Registry",
]
