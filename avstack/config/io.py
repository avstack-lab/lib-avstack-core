import os.path as osp
import warnings
from io import BytesIO, StringIO
from pathlib import Path

from .handlers import file_handlers  # noqa: F401


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def check_file_exist(file):
    if not osp.exists(file):
        raise FileNotFoundError(file)


def dump(
    obj, file=None, file_format=None, file_client_args=None, backend_args=None, **kwargs
):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello word', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                "same time."
            )

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if file_client_args is not None:
            from mmengine.fileio.file_client import FileClient

            file_client = FileClient.infer_client(file_client_args, file)
            file_backend = file_client
        else:
            from mmengine.fileio.io import get_file_backend

            file_backend = get_file_backend(file, backend_args=backend_args)

        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                file_backend.put(f.getvalue(), file)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
