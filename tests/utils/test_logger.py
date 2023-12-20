import os
import shutil
import sys
import tempfile

from avstack.config import ALGORITHMS, Config
from avstack.datastructs import DataContainer
from avstack.modules.base import BaseModule
from avstack.utils.decorators import apply_hooks
from avstack.utils.logging import ObjectStateLogger


sys.path.append("tests/")
from utilities import get_object_global


@ALGORITHMS.register_module()
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
        logger = ObjectStateLogger(save_folder=tmpdir)
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
        module.register_post_hook(ObjectStateLogger(save_folder=tmpdir))
        for i in range(n_files):
            module(frame=i, n_objects=n_objects)
        _, _, files = next(os.walk(tmpdir))
        assert len(files) == n_files


def test_logger_form_config():
    try:
        fname = "tests/utils/logger_cfg.py"
        cfg = Config.fromfile(fname)
        save_dir = cfg.alg.post_hooks[0].save_folder
        module = ALGORITHMS.build(cfg.alg)
        module(frame=10, n_objects=4)
        _, _, files = next(os.walk(save_dir))
        assert len(files) == 1
    except Exception as e:
        shutil.rmtree(save_dir)
        raise e
    finally:
        shutil.rmtree(save_dir)
