from typing import List

from avstack.config import HOOKS, ConfigDict


class BaseModule:
    def __init__(
        self,
        name: str,
        output_folder: str = "",
        pre_hooks: List[ConfigDict] = [],
        post_hooks: List[ConfigDict] = [],
    ) -> None:
        self.name = name
        self.output_folder = output_folder
        default_args = {}
        default_args["output_folder"] = output_folder
        self.pre_hooks = [
            HOOKS.build(hook, default_args=default_args) for hook in pre_hooks
        ]
        self.post_hooks = [
            HOOKS.build(hook, default_args=default_args) for hook in post_hooks
        ]

    def _apply_pre_hooks(self, *args, **kwargs):
        for hook in self.pre_hooks:
            args, kwargs = hook(*args, **kwargs)
        return args, kwargs

    def _apply_post_hooks(self, *args):
        for hook in self.post_hooks:
            args = hook(*args)
        if isinstance(args, tuple) and (len(args) == 1):
            args = args[0]  # do not love this at all...
        return args

    def register_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def register_post_hook(self, hook):
        self.post_hooks.append(hook)
