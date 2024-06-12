import os
import pickle
from typing import TYPE_CHECKING

from avstack.config import HOOKS


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.environment.objects import ObjectState
    from avstack.geometry import Shape
    from avstack.sensors import SensorData


# TODO: put this on its own thread


class Logger:
    def __init__(self, output_folder: str) -> None:
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def __call__(self, objects, *args, **kwargs):
        """Log objects to a folder, then return them"""
        file = self._get_file_name(objects, *args, **kwargs)
        self._write_to_file(objects, *args, file=file, **kwargs)
        return objects

    def _get_file_name(self):
        raise NotImplementedError

    def _encode(self, objects, *args, **kwargs):
        return objects.encode()

    def _write_to_file(self, objects, file, *args, **kwargs):
        with open(file, "w") as f:
            f.write(self._encode(objects))


####################################################
# agent poses
####################################################


@HOOKS.register_module()
class AgentPoseLogger(Logger):
    prefix = "agent"
    file_ending = "txt"

    def _get_file_name(self, agent: "ObjectState", *args, **kwargs):
        try:
            frame = agent.frame
        except AttributeError:
            frame = 0
        file = os.path.join(
            self.output_folder,
            f"{self.prefix}-{agent.ID}-{frame:010d}-{agent.timestamp:012.2f}.{self.file_ending}",
        )
        return file


####################################################
# things in DataContainers
####################################################


class _DataContainerLogger(Logger):
    def _get_file_name(self, objects: "DataContainer", *args, **kwargs):
        file = os.path.join(
            self.output_folder,
            f"{self.prefix}-{objects.source_identifier}-{objects.frame:010d}-{objects.timestamp:012.2f}.{self.file_ending}",
        )
        return file


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
    def __call__(self, data: "SensorData", *args, **kwargs):
        data.save_to_folder(self.output_folder, *args, **kwargs)
        return [data]  # need this based on the extraction in post-hooks for now...


@HOOKS.register_module()
class FieldOfViewLogger(Logger):
    prefix = "fov"
    file_ending = "txt"

    def _get_file_name(self, fov: "Shape", *args, **kwargs):
        file = os.path.join(
            self.output_folder,
            f"{self.prefix}-{fov.frame:010d}-{fov.timestamp:012.2f}.{self.file_ending}",
        )
        return file
