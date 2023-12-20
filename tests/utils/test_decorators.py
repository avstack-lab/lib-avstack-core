import os
import tempfile
import time

import numpy as np

from avstack.modules.base import BaseModule
from avstack.utils import decorators


def test_profileit():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name + ".prof"

        @decorators.profileit(tmp_file)
        def func_to_test():
            A = np.random.randn(10, 20)
            b = np.random.randn(20)
            c = A @ b

        for _ in range(10):
            func_to_test()
        assert os.path.exists(tmp_file)


def test_iterationmonitor():
    @decorators.FunctionTriggerIterationMonitor(print_rate=2)
    def func_to_test():
        time.sleep(0.05)

    for _ in range(20):
        func_to_test()


def test_apply_hooks():
    class TestClass(BaseModule):
        def __init__(self):
            self.a = 0
            super().__init__()

        @decorators.apply_hooks
        def __call__(self):
            self.a += 1

    def pre_hook(aclass, *args, **kwargs):
        aclass.a += 10
        return args, kwargs

    def post_hook(aclass, *args):
        aclass.a += 100
        return args

    atest = TestClass()
    atest.register_pre_hook(pre_hook)
    atest.register_post_hook(post_hook)

    assert atest.a == 0
    atest()
    assert atest.a == 111
