from avstack.config import MODELS, PIPELINE


@MODELS.register_module()
class Module1():
    def __call__(self, data: int):
        return data + 5


@MODELS.register_module()
class Module2():
    def __call__(self, data: int):
        return data * 3


@MODELS.register_module()
class Module3():
    def __call__(self, data1: int, data2: int):
        return data1 + data2


def test_serial_pipeline():
    modules = [{"type": "Module1"}, {"type": "Module2"}]
    pipe_cfg = {"type": "SerialPipeline", "modules": modules}
    pipeline = PIPELINE.build(pipe_cfg)
    i_in = 1
    i_out = pipeline(i_in)
    assert i_out == 18


def test_mapped_pipeline():
    modules = {"module1": {"type": "Module1"}, "module2":{"type": "Module2"}, "module3":{"type": "Module3"}}
    mapping = {"module1": ["input1"], "module2": ["module1"], "module3": ["module1", "module2"]}
    pipe_cfg = {"type": "MappedPipeline", "modules": modules, "mapping": mapping}
    pipeline = PIPELINE.build(pipe_cfg)
    i_in = 1
    i_out = pipeline(i_in)
    assert i_out == 24