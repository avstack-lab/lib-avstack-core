# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os
import shutil

import numpy as np

from avstack.environment.objects import VehicleState
from avstack.modules.perception.detections import BoxDetection


class _TrackingAlgorithm:
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        save_output=False,
        save_folder="",
        **kwargs
    ):
        self.iframe = -1
        self.assign_metric = assign_metric
        self.assign_radius = assign_radius
        self.save = save_output
        self.save_folder = save_folder
        self.save = save_output
        self.save_folder = os.path.join(save_folder, "tracking")
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)

    @property
    def confirmed_tracks(self):
        return self.tracks_confirmed

    def assign(self, dets, tracks):
        A = np.zeros((len(dets), len(tracks)))
        for i, b1 in enumerate(dets):
            boxa = b1.box if isinstance(b1, VehicleState) else b1
            box1 = boxa.box if isinstance(b1, BoxDetection) else boxa
            for j, b2 in enumerate(tracks):
                box2 = b2.as_object().box
                # -- either way, change origin and use radius to filter coarsely
                if box1.origin != box2.origin:
                    box1.change_origin(box2.origin)
                if self.assign_radius is not None:
                    dist = box1.t.distance(box2.t)
                    if dist > self.assign_radius:
                        continue
                # -- use the metric of choice
                if self.assign_metric == "IoU":
                    cost = -box1.IoU(box2)  # lower is better
                elif self.assign_metric == "center_dist":
                    cost = dist - self.assign_radius  # lower is better
                else:
                    raise NotImplementedError(self.assign_metric)
                # -- store result
                A[i, j] = cost
        return A

    def __call__(self, *args, **kwargs):
        self.frame = kwargs.get('frame')
        self.iframe += 1
        tracks = self.track(*args, **kwargs)
        if self.save:
            trk_str = "\n".join([trk.format_as("avstack") for trk in tracks])
            fname = os.path.join(self.save_folder, "%06d.txt" % self.frame)
            with open(fname, "w") as f:
                f.write(trk_str)
        return tracks


class PassthroughTracker(_TrackingAlgorithm):
    def __init__(self, output, **kwargs):
        super().__init__("PassthroughTracker")
        self.output = output

    def __call__(self, detections, *args, **kwargs):
        if isinstance(detections, dict):
            if len(detections) == 1:
                return detections[list(detections.keys())[0]]
            else:
                raise NotImplementedError(detections)
        else:
            return detections
