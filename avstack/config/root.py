from .registry import Registry


# data-related
DATASETS = Registry("datasets")
REFERENCE = Registry("reference")

# algorithm-related
AGENTS = Registry("agents")
ALGORITHMS = Registry("algorithms")
MODELS = Registry("models")
PIPELINE = Registry("pipeline")
