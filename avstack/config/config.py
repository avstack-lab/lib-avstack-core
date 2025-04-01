import ast
import os.path as osp
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from addict import Dict

from .io import check_file_exist, dump


BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]


class Config:
    """Base class for configuration of AVstack projects/modules

    Example:

    from avstack.config import Config

    cfg = Config.fromfile('myconfigfile.py')
    x = cfg.some_config_value
    """

    def __init__(self, cfg_dict: dict, filename: str = None) -> None:
        """Initialize a configuration object"""
        super().__setattr__("_cfg_dict", cfg_dict)
        super().__setattr__("_filename", filename)

    @property
    def filename(self) -> str:
        """get file name of config."""
        return self._filename

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    @property
    def pretty_text(self) -> str:
        """Get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list_tuple(k, v, use_mapping=False):
            if isinstance(v, list):
                left = "["
                right = "]"
            else:
                left = "("
                right = ")"

            v_str = f"{left}\n"
            # check if all items in the list are dict
            for item in v:
                if isinstance(item, dict):
                    v_str += f"dict({_indent(_format_dict(item), indent)}),\n"
                elif isinstance(item, tuple):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: 501
                elif isinstance(item, list):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: 501
                elif isinstance(item, str):
                    v_str += f"{_indent(repr(item), indent)},\n"
                else:
                    v_str += str(item) + ",\n"
            if k is None:
                return _indent(v_str, indent) + right
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent) + right
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(
                sorted(input_dict.items(), key=lambda x: str(x[0]))
            ):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _format_list_tuple(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # if self._format_python_code:
        #     # copied from setup.cfg
        #     yapf_style = dict(
        #         based_on_style='pep8',
        #         blank_line_before_nested_class_or_def=True,
        #         split_before_expression_after_opening_paren=True)
        #     try:
        #         if digit_version(yapf.__version__) >= digit_version('0.40.2'):
        #             text, _ = FormatCode(text, style_config=yapf_style)
        #         else:
        #             text, _ = FormatCode(
        #                 text, style_config=yapf_style, verify=True)
        #     except:  # noqa: E722
        #         raise SyntaxError('Failed to format the config file, please '
        #                           f'check the syntax of: \n{text}')
        return text

    @staticmethod
    def fromfile(filename: str):
        return Config(cfg_dict=Config._file_to_config_dict(filename), filename=filename)

    def dump(self, file: Optional[Union[str, Path]] = None):
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.

        Returns:
            str or None: Config text.
        """
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = self.to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
            else:
                file_format = self.filename.split(".")[-1]
                return dump(cfg_dict, file_format=file_format)
        elif file.endswith(".py"):
            with open(file, "w", encoding="utf-8") as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split(".")[-1]
            return dump(cfg_dict, file=file, file_format=file_format)

    @staticmethod
    def _dict_to_config_dict(cfg: dict):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(value)
        elif isinstance(cfg, tuple):
            cfg = tuple(Config._dict_to_config_dict(_cfg) for _cfg in cfg)
        elif isinstance(cfg, list):
            cfg = [Config._dict_to_config_dict(_cfg) for _cfg in cfg]
        return cfg

    @staticmethod
    def _file_to_dict(filename: str):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        file_ext_name = osp.splitext(filename)[1]
        if file_ext_name not in [".py", ".json", ".yaml", ".yml"]:
            raise OSError("Only py/yml/yaml/json type are supported now!")

        # populate base fields
        base_cfg_dict = ConfigDict()
        for base_cfg_path in Config._get_base_files(filename):
            base_cfg_path = Config._get_cfg_path(base_cfg_path, filename)
            _cfg_dict = Config._file_to_config_dict(
                filename=base_cfg_path,
            )
            duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
            if len(duplicate_keys) > 0:
                raise KeyError(
                    "Duplicate key is not allowed among bases. "
                    f"Duplicate keys: {duplicate_keys}"
                )
            _cfg_dict = Config._dict_to_config_dict(_cfg_dict)
            base_cfg_dict.update(_cfg_dict)

        if filename.endswith(".py"):
            with open(filename, encoding="utf-8") as f:
                parsed_codes = ast.parse(f.read())
                parsed_codes = RemoveAssignFromAST(BASE_KEY).visit(parsed_codes)
            codeobj = compile(parsed_codes, filename, mode="exec")
            # Support load global variable in nested function of the
            # config.
            global_locals_var = {BASE_KEY: base_cfg_dict}
            ori_keys = set(global_locals_var.keys())
            eval(codeobj, global_locals_var, global_locals_var)
            cfg_dict = {
                key: value
                for key, value in global_locals_var.items()
                if (key not in ori_keys and not key.startswith("__"))
            }

        else:
            raise NotImplementedError(file_ext_name)

        cfg_dict.pop(BASE_KEY, None)
        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("__")}

        return cfg_dict

    @staticmethod
    def _get_cfg_path(cfg_path: str, filename: str) -> Tuple[str, Optional[str]]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            Tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        if "::" in cfg_path:
            # `cfg_path` startswith '::' means an external config path.
            # Get package name and relative config path.
            raise NotImplementedError
            # scope = cfg_path.partition('::')[0]
            # package, cfg_path = _get_package_and_cfg_path(cfg_path)

            # if not is_installed(package):
            #     raise ModuleNotFoundError(
            #         f'{package} is not installed, please install {package} '
            #         f'manually')

            # # Get installed package path.
            # package_path = get_installed_path(package)
            # try:
            #     # Get config path from meta file.
            #     cfg_path = _get_external_cfg_path(package_path, cfg_path)
            # except ValueError:
            #     # Since base config does not have a metafile, it should be
            #     # concatenated with package path and relative config path.
            #     cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
            # except FileNotFoundError as e:
            #     raise e
            # return cfg_path, scope
        else:
            # Get local config path.
            cfg_dir = osp.dirname(filename)
            cfg_path = osp.join(cfg_dir, cfg_path)
            return cfg_path

    @staticmethod
    def _file_to_config_dict(filename: str):
        return Config._dict_to_config_dict(Config._file_to_dict(filename))

    @staticmethod
    def _get_base_files(filename: str):
        file_format = osp.splitext(filename)[1]
        if file_format == ".py":
            Config._validate_py_syntax(filename)
            with open(filename, encoding="utf-8") as f:
                parsed_codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (
                        isinstance(c, ast.Assign)
                        and isinstance(c.targets[0], ast.Name)
                        and c.targets[0].id == BASE_KEY
                    )

                base_code = next((c for c in parsed_codes if is_base_line(c)), None)
                if base_code is not None:
                    base_code = ast.Expression(  # type: ignore
                        body=base_code.value
                    )  # type: ignore
                    base_files = eval(compile(base_code, "", mode="eval"))
                else:
                    base_files = []
        else:
            raise NotImplementedError(file_format)
        if isinstance(base_files, str):
            base_files = [base_files]
        return base_files

    @staticmethod
    def _validate_py_syntax(filename: str):
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding="utf-8") as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(
                "There are syntax errors in config " f"file {filename}: {e}"
            )

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False) -> dict:
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        (dict, list) if allow_list_keys else dict
                    )
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b


class ConfigDict(Dict):
    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, "__parent", kwargs.pop("__parent", None))
        object.__setattr__(__self, "__key", kwargs.pop("__key", None))
        object.__setattr__(__self, "__frozen", False)
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    @classmethod
    def _hook(cls, item):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == self.key:
            return None
        else:
            return node
