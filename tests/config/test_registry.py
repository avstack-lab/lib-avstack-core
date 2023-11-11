from typing import Dict

from avstack.config import ALGORITHMS


@ALGORITHMS.register_module()
class MyAlgorithm:
    def __init__(self, arg_1: str, arg_2: Dict[str, int]) -> None:
        self.arg_1 = arg_1
        self.arg_2 = arg_2


def test_registry():
    alg = ALGORITHMS.build(dict(type="MyAlgorithm", arg_1="test", arg_2=dict(a=1)))
    assert isinstance(alg, MyAlgorithm)
    assert alg.arg_1 == "test"
