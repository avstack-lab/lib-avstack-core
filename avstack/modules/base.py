class BaseModule:
    def __init__(self) -> None:
        self.pre_hooks = []
        self.post_hooks = []

    def _apply_pre_hooks(self, *args, **kwargs):
        for hook in self.pre_hooks:
            args, kwargs = hook(self, *args, **kwargs)
        return args, kwargs

    def _apply_post_hooks(self, *args):
        for hook in self.post_hooks:
            args = hook(self, *args)
        if len(args) == 1:
            args = args[0]  # do not love this at all...
        return args

    def register_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def register_post_hook(self, hook):
        self.post_hooks.append(hook)
