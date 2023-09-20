import time

import numpy as np


def check_xor_for_none(a, b):
    assert check_xor(a is None, b is None), "Can only pass in one of these inputs"


def check_xor(a, b):
    return (a or b) and (not a or not b)


class IterationMonitor:
    def __init__(self, print_method="real_time", print_rate=1 / 2) -> None:
        self.iteration = 0
        self.print_method = print_method
        self.t0 = time.time()
        self.tf = time.time()
        self.t_last_print = -np.inf
        self.print_rate = print_rate
        self.print_interval = 1.0 / print_rate

    def tick(self):
        self.iteration += 1
        self.tf = time.time()
        self.print()

    def print(self):
        if self.print_method == "real_time":
            if (self.tf - self.t_last_print + 1e-6) >= self.print_interval:
                self.t_last_print = self.tf
                print(
                    f"Iteration: {self.iteration:5d}, {self.tf-self.t0:5.2f} Seconds Elapsed"
                )
        elif self.print_method == "sim_time":
            raise NotImplementedError
        elif self.print_method == "iteration":
            raise NotImplementedError
        else:
            raise NotImplementedError(self.print_method)
