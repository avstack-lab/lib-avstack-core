import logging
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Type, Union

from rich.console import Console
from rich.table import Table

from .build_functions import build_from_cfg
from .utils import is_seq_of


MODULE2PACKAGE = []


class Registry:
    """A registry to map strings to classes or functions


    Args:
        name (str): Registry name
        parent (:obj:`Registry`, optional): Parent registry. The class
            registered in children registry could be built from parent.
            Defaults to None.

    Example of hierarchical registry
    ALGORITHMS = Registry('algorithms')
    TRACKERS = Registry('trackers', parent=ALGORITHMS)

    # register model
    @TRACKERS.register_module()
    class AnExampleTracker:
        pass

    # build model
    tracker = TRACKERS.build(dict(type='AnExampleTracker'))
    """

    def __init__(self, name: str, parent: Optional["Registry"] = None) -> None:
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, "Registry"] = dict()
        self._locations = []
        self.scope = None
        self._imported = False
        self.parent: Optional["Registry"]
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None
        self.build_func = build_from_cfg

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._get_root_registry()

    def _get_root_registry(self) -> "Registry":
        """Return the root registry."""
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def import_from_location(self) -> None:
        """import modules from the pre-defined locations in self._location."""
        if not self._imported:
            # Avoid circular import
            from .logging import print_log

            # avoid BC breaking
            if len(self._locations) == 0 and self.scope in MODULE2PACKAGE:
                print_log(
                    f'The "{self.name}" registry in {self.scope} did not '
                    "set import location. Fallback to call "
                    f"`{self.scope}.utils.register_all_modules` "
                    "instead.",
                    logger="current",
                    level=logging.DEBUG,
                )
                try:
                    module = import_module(f"{self.scope}.utils")
                except (ImportError, AttributeError, ModuleNotFoundError):
                    if self.scope in MODULE2PACKAGE:
                        print_log(
                            f"{self.scope} is not installed and its "
                            "modules will not be registered. If you "
                            "want to use modules defined in "
                            f"{self.scope}, Please install {self.scope} by "
                            f"`pip install {MODULE2PACKAGE[self.scope]}.",
                            logger="current",
                            level=logging.WARNING,
                        )
                    else:
                        print_log(
                            f"Failed to import {self.scope} and register "
                            "its modules, please make sure you "
                            "have registered the module manually.",
                            logger="current",
                            level=logging.WARNING,
                        )
                else:
                    # The import errors triggered during the registration
                    # may be more complex, here just throwing
                    # the error to avoid causing more implicit registry errors
                    # like `xxx`` not found in `yyy` registry.
                    module.register_all_modules(False)  # type: ignore

            for loc in self._locations:
                import_module(loc)
                print_log(
                    f"Modules of {self.scope}'s {self.name} registry have "
                    f"been automatically imported from {loc}",
                    logger="current",
                    level=logging.DEBUG,
                )
            self._imported = True

    def get(self, key: str) -> Optional[Type]:
        if not isinstance(key, str):
            raise TypeError(
                "The key argument of `Registry.get` must be a str, " f"got {type(key)}"
            )

        obj_cls = None

        # lazy import the modules to register them into the registry
        self.import_from_location()

        if key in self._module_dict:
            obj_cls = self._module_dict[key]
        else:
            # try to get the target from its parent or ancestors
            parent = self.parent
            while parent is not None:
                if key in parent._module_dict:
                    obj_cls = parent._module_dict[key]
                    registry_name = parent.name
                    scope_name = parent.scope
                    break
                parent = parent.parent

        return obj_cls

    def build(self, cfg: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.

        Examples:
            >>> from mmengine import Registry
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     def __init__(self, depth, stages=4):
            >>>         self.depth = depth
            >>>         self.stages = stages
            >>> cfg = dict(type='ResNet', depth=50)
            >>> model = MODELS.build(cfg)
        """
        return self.build_func(cfg, *args, **kwargs, registry=self)

    def _add_child(self, registry: "Registry") -> None:
        """Add a child for a registry.

        Args:
            registry (:obj:`Registry`): The ``registry`` will be added as a
                child of the ``self``.
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(
        self,
        module: Type,
        module_name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
    ) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f"module must be Callable, but got {type(module)}")

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(
                    f"{name} is already registered in {self.name} "
                    f"at {existed_module.__module__}"
                )
            self._module_dict[name] = module

    def register_module(
        self,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        module: Optional[Type] = None,
    ) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.

        Examples:
            >>> backbones = Registry('backbone')
            >>> # as a decorator
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> # as a normal function
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register
