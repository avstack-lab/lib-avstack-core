import cProfile
import os

from .other import IterationMonitor


def profileit(name, folder="./"):
    """A decorator to profile a function

    Use it as:

    @profileit("profile_for_func1_001")
    def func1(...)
    ...

    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    name = os.path.join(folder, name)
    # if os.path.exists(name):
    #     if '.' in name:
    #         name_parts = name.split('.')
    #         name_parts[-2] = name_parts[-2] + '_1'
    #         name = '.'.join(name_parts)
    #     else:
    #         name = name + '_1'
    def inner(func):
        """Decorator wrapper that takes func as input"""

        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            # Note use of name from outer scope
            prof.dump_stats(name)
            return retval

        return wrapper

    return inner


class FunctionTriggerIterationMonitor:
    """A decorator to monitor iteration timing

    Use it as:

    @FunctionTriggerIterationMonitor(print_rate=1/2)
    def func1(...)
    ...

    """

    def __init__(self, print_rate) -> None:
        if callable(print_rate):
            raise TypeError("Did you pass the `print_rate` parameter to the decorator?")
        self.timer = IterationMonitor(print_rate=print_rate, print_method="real_time")

    def __call__(self, func):
        decorator_self = self  # why need this?

        def wrapper(*args, **kwargs):
            decorator_self.timer.tick()
            return func(*args, **kwargs)

        return wrapper
