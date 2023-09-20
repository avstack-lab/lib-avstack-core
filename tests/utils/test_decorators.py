import os
import tempfile
import time

import numpy as np

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
