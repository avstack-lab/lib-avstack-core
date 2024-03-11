import os

from avstack.config import HOOKS
from avstack.datastructs import DataContainer
from avstack.sensors import SensorData


# TODO: put this on its own thread


class Logger:
    def __init__(self, output_folder: str) -> None:
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        """Log objects to a folder, then return them"""
        raise NotImplementedError


####################################################
# things in DataContainers
####################################################


class _DataContainerLogger(Logger):
    def __call__(self, objects: DataContainer, *args, **kwargs):
        file = os.path.join(
            self.output_folder,
            f"{self.prefix}-{objects.source_identifier}-{objects.frame:010d}-{objects.timestamp:012.2f}.txt",
        )
        with open(file, "w") as f:
            f.write(objects.encode())
        return objects


@HOOKS.register_module()
class ObjectStateLogger(_DataContainerLogger):
    prefix = "objectstate"


@HOOKS.register_module()
class DetectionsLogger(_DataContainerLogger):
    prefix = "detections"


@HOOKS.register_module()
class TracksLogger(_DataContainerLogger):
    prefix = "tracks"


####################################################
# others
####################################################


@HOOKS.register_module()
class SensorDataLogger(Logger):
    def __call__(self, data: SensorData, *args, **kwargs):
        data.save_to_folder(self.output_folder)
        return [data]  # need this based on the extraction in post-hooks for now...
