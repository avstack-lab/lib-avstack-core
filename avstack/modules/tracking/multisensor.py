from typing import TYPE_CHECKING, Any, Dict, Union

import numpy as np


if TYPE_CHECKING:
    from avstack.geometry import ReferenceFrame, Shape

from avstack.config import MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D
from avstack.geometry.utils import parallel_in_polygon
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
            MODELS.build(tracker, default_args={"name": "trackerformulti"})
            if isinstance(tracker, (dict, ConfigDict))
            else tracker
        )
        self.platform = platform

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tracker, name)

    @apply_hooks
    def __call__(
        self,
        detections: Dict[int, DataContainer],
        fovs: Dict[int, Union["Shape", np.ndarray]],
        platforms: Dict[int, "ReferenceFrame"],
        check_reference: bool = True,
        *args,
        **kwargs,
    ) -> DataContainer:
        """Run the multi-platform, multi-target tracking

        NOTE: all inputs must be in common reference frame
        """

        for ID in detections:
            # NOTE: no need to align frames since all inputs are global
            pts_ref = [trk.position for trk in self.tracker.tracks_active]
            # align reference frame for checking observables
            # pts_ref = [
            #     trk.position.change_reference(platforms[ID], inplace=False)
            #     if check_reference
            #     else trk.position
            #     for trk in self.tracker.tracks_active
            # ]

            # use the FOV model to filter for observable tracks
            if detections[ID] is None:
                continue
            try:
                # for fov shape that has a check point method
                trks_observable = []
                for trk, pos in zip(self.tracker.tracks_active, pts_ref):
                    if fovs[ID].check_point(pos.x):
                        trks_observable.append(trk)
            except AttributeError:
                # for an array of points that needs a polygon method
                pos_to_check = [pos.x[:2] for pos in pts_ref]
                trks_observable = (
                    []
                    if len(pos_to_check) == 0
                    else [
                        trk
                        for trk, in_h in zip(
                            self.tracker.tracks_active,
                            parallel_in_polygon(pos_to_check, fovs[ID]),
                        )
                        if in_h
                    ]
                )

            # if (detections[ID].timestamp == 0.74) and (ID == 2):
            #     tracks_out_tmp = DataContainer(
            #         frame=self.tracker.frame,
            #         timestamp=self.tracker.timestamp,
            #         data=self.tracker.tracks_confirmed,
            #         source_identifier=self.name,
            #     )
            #     dists = []
            #     for i, track_i in enumerate(tracks_out_tmp):
            #         for j, track_j in enumerate(tracks_out_tmp):
            #             if j <= i:
            #                 continue
            #             dists.append(track_i.box.distance(track_j.box))
            #     import pdb; pdb.set_trace()

            # update the tracks with the new detections
            self.tracker(
                detections=detections[ID],
                platform=self.platform,
                trks_observable=trks_observable,
                check_reference=check_reference,
                *args,
                **kwargs,
            )

            # check if two tracks are too close to each other
            tracks_out_tmp = DataContainer(
                frame=self.tracker.frame,
                timestamp=self.tracker.timestamp,
                data=self.tracker.tracks_confirmed,
                source_identifier=self.name,
            )
            dists = []
            for i, track_i in enumerate(tracks_out_tmp):
                for j, track_j in enumerate(tracks_out_tmp):
                    if j <= i:
                        continue
                    dists.append(track_i.box.distance(track_j.box))
            if any([d < 1 for d in dists]):
                import pdb

                pdb.set_trace()

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
