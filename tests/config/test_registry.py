from typing import Dict

from avstack.config import ALGORITHMS, ConfigDict


@ALGORITHMS.register_module()
class MyAlgorithm:
    def __init__(self, arg_1: str, arg_2: Dict[str, int]) -> None:
        self.arg_1 = arg_1
        self.arg_2 = arg_2


@ALGORITHMS.register_module()
class MyAlgorithmSecondaryBuild:
    def __init__(self, arg_1: str, arg_2: ConfigDict) -> None:
        self.arg_1 = arg_1
        self.arg_2 = ALGORITHMS.build(arg_2)


def test_registry_1():
    alg = ALGORITHMS.build(dict(type="MyAlgorithm", arg_1="test", arg_2=dict(a=1)))
    assert isinstance(alg, MyAlgorithm)
    assert alg.arg_1 == "test"


def test_registry_2():
    alg = ALGORITHMS.build(dict(type="BasicXyTracker"))
    assert type(alg).__name__ == "BasicXyTracker"


def test_registry_two_level():
    alg = ALGORITHMS.build(
        dict(
            type="MyAlgorithmSecondaryBuild",
            arg_1="test",
            arg_2=dict(type="BasicXyTracker"),
        )
    )
    assert isinstance(alg, MyAlgorithmSecondaryBuild)
    assert alg.arg_1 == "test"
    assert type(alg.arg_2).__name__ == "BasicXyTracker"
