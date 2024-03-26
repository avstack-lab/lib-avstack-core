from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame, Shape

from avstack.config import MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import WorldFrame
from avstack.modules import BaseModule
from avstack.utils.decorators import apply_hooks


# ==============================================================
# MEASUREMENT-BASED MULTI-SENSOR TRACKER
# ==============================================================


@MODELS.register_module()
class MeasurementBasedMultiTracker(BaseModule):
    def __init__(self, tracker, name="multitracker", frame=WorldFrame, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tracker = (
            MODELS.build(tracker) if isinstance(tracker, ConfigDict) else tracker
        )
        self.reference = frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tracker, name)

    @apply_hooks
    def __call__(
        self,
        detections: Dict[int, DataContainer],
        fovs: Dict[int, "Shape"],
        frames: Dict[int, "ReferenceFrame"],
        check_reference: bool = True,
        *args,
        **kwargs,
    ) -> DataContainer:

        for ID in detections:
            # align reference frame for checking observables
            pts_ref = [
                trk.position.change_reference(frames[ID])
                if trk.position.frame != frames[ID]
                else trk.position
                for trk in self.tracker.tracks_active
            ]

            # use the FOV model to filter for observable tracks
            if detections[ID] is None:
                continue
            trks_observable = [
                trk
                for trk, pos in zip(self.tracker.tracks_active, pts_ref)
                if fovs[ID].check_point(pos.x)
            ]
            # update the tracks with the new detections
            self.tracker(
                detections=detections[ID],
                platform=self.platform,
                trks_observable=trks_observable,
                *args,
                **kwargs,
            )

        # format as data container
        tracks_out = DataContainer(
            frame=self.tracker.frame,
            stamp=self.tracker.stamp,
            data=self.tracker.tracks_confirmed,
            source_identifier=self.name,
        )
        return tracks_out


# ==============================================================
# TRACK-BASED MULTI-SENSOR TRACKER
# ==============================================================


class TrackBasedMultiTracker(BaseModule):
    pass
