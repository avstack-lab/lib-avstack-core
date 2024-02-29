from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame, Shape

from avstack.config import MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


# ==============================================================
# MEASUREMENT-BASED MULTI-SENSOR TRACKER
# ==============================================================


@MODELS.register_module()
class MeasurementBasedMultiTracker(BaseModule):
    def __init__(self, tracker, name="multitracker", platform=GlobalOrigin3D, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tracker = (
            MODELS.build(tracker) if isinstance(tracker, ConfigDict) else tracker
        )
        self.platform = platform

    @apply_hooks
    def __call__(
        self,
        detections: Dict[int, DataContainer],
        fovs: Dict[int, "Shape"],
        platforms: Dict[int, "ReferenceFrame"],
        *args,
        **kwargs,
    ) -> DataContainer:

        for ID in detections:
            # use the FOV model to filter for observable tracks
            if detections[ID] is None:
                continue
            trks_observable = [
                trk
                for trk in self.tracker.tracks_active
                if fovs[ID].check_point(
                    trk.position.change_reference(platforms[ID], inplace=False).x
                )
            ]
            # update the tracks with the new detections
            self.tracker(
                detections=detections[ID],
                platform=self.platform,
                trks_observable=trks_observable,
            )

        # format as data container
        tracks_out = DataContainer(
            frame=self.tracker.frame,
            timestamp=self.tracker.timestamp,
            data=self.tracker.tracks_confirmed,
            source_identifier=self.name,
        )
        return tracks_out


# ==============================================================
# TRACK-BASED MULTI-SENSOR TRACKER
# ==============================================================


class TrackBasedMultiTracker(BaseModule):
    pass
