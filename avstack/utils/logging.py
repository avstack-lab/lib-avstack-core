import os
import pickle

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
            f"{self.prefix}-{objects.source_identifier}-{objects.frame:010d}-{objects.timestamp:012.2f}.{self.file_ending}",
        )
        self._write_to_file(objects, file)
        return objects
    
    def _encode(self, objects):
        return objects.encode()

    def _write_to_file(self, objects, file):
        with open(file, "w") as f:
            f.write(self._encode(objects))


@HOOKS.register_module()
class ObjectStateLogger(_DataContainerLogger):
    prefix = "objectstate"
    file_ending = "txt"


@HOOKS.register_module()
class DetectionsLogger(_DataContainerLogger):
    prefix = "detections"
    file_ending = "txt"


@HOOKS.register_module()
class TracksLogger(_DataContainerLogger):
    prefix = "tracks"
    file_ending = "txt"
    

@HOOKS.register_module()
class StoneSoupTracksLogger(_DataContainerLogger):
    prefix = "tracks"
    file_ending = "pickle"
    def _write_to_file(self, objects, file):
        with open(file, "wb") as f:
            pickle.dump(objects, f)



####################################################
# others
####################################################


@HOOKS.register_module()
class SensorDataLogger(Logger):
    def __call__(self, data: SensorData, *args, **kwargs):
        data.save_to_folder(self.output_folder, *args, **kwargs)
        return [data]  # need this based on the extraction in post-hooks for now...
