import os

from avstack.config import HOOKS
from avstack.datastructs import DataContainer


class Logger:
    def __init__(self, save_folder: str) -> None:
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        """Log objects to a folder, then return them"""
        raise NotImplementedError


@HOOKS.register_module()
class ObjectLogger(Logger):
    def __call__(self, objects: DataContainer, *args, **kwargs):
        file = os.path.join(
            self.save_folder,
            f"objects-{objects.frame:04d}-{objects.timestamp:5.2f}.txt",
        )
        with open(file, "w") as f:
            f.write(objects.encode())
        return objects
