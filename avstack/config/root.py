from .registry import Registry


# data-related
DATASETS = Registry("datasets")
REFERENCE = Registry("reference")

# algorithm-related
AGENTS = Registry("agents")
MODELS = Registry("models")
PIPELINE = Registry("pipeline")

# other registries
HOOKS = Registry("hooks")
