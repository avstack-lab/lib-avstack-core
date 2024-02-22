from typing import Dict

from avstack.config import MODELS, ConfigDict


@MODELS.register_module()
class MyAlgorithm:
    def __init__(self, arg_1: str, arg_2: Dict[str, int]) -> None:
        self.arg_1 = arg_1
        self.arg_2 = arg_2


@MODELS.register_module()
class MyMODELSecondaryBuild:
    def __init__(self, arg_1: str, arg_2: ConfigDict, arg_3: str = "default") -> None:
        self.arg_1 = arg_1
        self.arg_2 = MODELS.build(arg_2)
        self.arg_3 = arg_3


def test_registry_1():
    alg = MODELS.build(dict(type="MyAlgorithm", arg_1="test", arg_2=dict(a=1)))
    assert isinstance(alg, MyAlgorithm)
    assert alg.arg_1 == "test"


def test_registry_2():
    alg = MODELS.build(dict(type="BasicXyTracker"))
    assert type(alg).__name__ == "BasicXyTracker"


def test_registry_two_level():
    alg = MODELS.build(
        dict(
            type="MyMODELSecondaryBuild",
            arg_1="test",
            arg_2=dict(type="BasicXyTracker"),
        )
    )
    assert isinstance(alg, MyMODELSecondaryBuild)
    assert alg.arg_1 == "test"
    assert type(alg.arg_2).__name__ == "BasicXyTracker"


def test_build_with_default():
    alg_1 = MODELS.build(
        dict(
            type="MyMODELSecondaryBuild",
            arg_1="test",
            arg_2=dict(type="BasicXyTracker"),
        ),
    )

    alg_2 = MODELS.build(
        dict(
            type="MyMODELSecondaryBuild",
            arg_1="test",
            arg_2=dict(type="BasicXyTracker"),
        ),
        default_args={"arg_3": "new value!"},
    )
    assert alg_1.arg_3 == "default"
    assert alg_2.arg_3 == "new value!"
