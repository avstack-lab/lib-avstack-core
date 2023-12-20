import os
import sys
import tempfile

from avstack.datastructs import DataContainer
from avstack.modules.base import BaseModule
from avstack.utils.decorators import apply_hooks
from avstack.utils.logging import ObjectLogger


sys.path.append("tests/")
from utilities import get_object_global


class MyTestModule(BaseModule):
    @apply_hooks
    def __call__(self, frame, n_objects):
        objs = DataContainer(
            frame=frame,
            timestamp=0,
            source_identifier="",
            data=[get_object_global(i) for i in range(n_objects)],
        )
        return objs


def test_object_logger():
    n_files = 10
    n_objects = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ObjectLogger(save_folder=tmpdir)
        for i in range(n_files):
            objs = DataContainer(
                frame=i,
                timestamp=0,
                source_identifier="",
                data=[get_object_global(i) for i in range(n_objects)],
            )
            logger(objs)
        _, _, files = next(os.walk(tmpdir))
        assert len(files) == n_files


def test_object_logger_as_hook():
    module = MyTestModule()
    n_files = 10
    n_objects = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        module.register_post_hook(ObjectLogger(save_folder=tmpdir))
        for i in range(n_files):
            module(frame=i, n_objects=n_objects)
        _, _, files = next(os.walk(tmpdir))
        assert len(files) == n_files
