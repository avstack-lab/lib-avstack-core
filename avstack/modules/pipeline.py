from typing import Any, Dict, List

from avstack.config import MODELS, PIPELINE, ConfigDict
from avstack.utils.decorators import apply_hooks

from .base import BaseModule


@PIPELINE.register_module()
class SerialPipeline(BaseModule):
    def __init__(self, modules: List[ConfigDict], *args: Any, **kwargs: Any) -> None:
        super().__init__(name="pipeline", *args, **kwargs)
        self.modules = [MODELS.build(mod) for mod in modules]

    @apply_hooks
    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        for module in self.modules:
            data = module(data, *args, **kwargs)
        return data

    def initialize(self, *args, **kwargs):
        for module in self.modules:
            module.initialize(*args, **kwargs)


@PIPELINE.register_module()
class MappedPipeline(BaseModule):
    def __init__(
        self,
        modules: Dict[str, ConfigDict],
        mapping: Dict[str, List[str]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A non-serial pipeline of modules mapping data to algorithms

        Can support routing data between multiple modules
        Gets tricky if the ordering of inputs to modules is important

        Arguments:
        :modules - dictionary of names to algorithms
        :mapping - dictionary of names of algorithms to names of algorithms whose outputs form inputs

        Example:
        ---
        modules = {"percep1": ALG1, "percep2": ALG2, "tracking": ALG3}
        mapping = {"percep1": ["sensor1"], "percep2": ["sensor2"], "tracking": ["percep1", "percep2"]}
        pipeline = MappedPipeline(modules, pipeline)
        data_in = {"sensor1": DATA1, "sensor2": DATA2}
        tracks = pipeline(data_in)
        """
        super().__init__(name="pipeline", *args, **kwargs)
        self.modules = {name: MODELS.build(mod) for name, mod in modules.items()}
        self.mapping = mapping

    @apply_hooks
    def __call__(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Runs modules one-by-one in order mapping data between them"""
        for name, module in self.modules.items():
            this_in = [data[in_name] for in_name in self.mapping[name]]
            last_data = module(*this_in)
            data[name] = last_data
        return last_data  # only return last module data?

    def initialize(self, *args, **kwargs):
        for module in self.modules.items():
            module.initialize(*args, **kwargs)


@PIPELINE.register_module()
class CustomPipeline(BaseModule):
    def __init__(self, modules: List[ConfigDict], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.modules = [MODELS.build(mod) for mod in modules]

    @apply_hooks
    def __call__(self):
        raise NotImplementedError("Implement a custom pipeline in a subclass")

    def initialize(self, *args, **kwargs):
        for module in self.modules:
            module.initialize(*args, **kwargs)
