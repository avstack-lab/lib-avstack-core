from datetime import datetime, timedelta

import numpy as np
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    RandomWalk,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.geometry import ReferenceFrame
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


@MODELS.register_module()
class StoneSoupKalmanTrackerBase(BaseModule):
    category_index = {"car": {"id": 1, "name": "car"}}
    # self.category_index = {1: {"id": 1, "name": "car"}}

    def __init__(self, **kwargs):
        super().__init__(name="tracking", **kwargs)
        self.iframe = -1
        self.frame = 0
        self.timestamp = 0
        # initial time
        self.t0 = datetime.now()

    @property
    def tracks_confirmed(self):
        return self.tracks

    @property
    def tracks_active(self):
        return self.tracks

    @apply_hooks
    def __call__(self, detections, platform: ReferenceFrame, **kwargs):
        if not isinstance(detections, DataContainer):
            raise ValueError(
                f"Detections are {type(detections)}, must be DataContainer"
            )
        self.timestamp = float(detections.timestamp)
        self.frame = int(detections.frame)
        self.iframe += 1
        tracks = self.track(detections, platform, **kwargs)
        track_data = DataContainer(self.frame, self.timestamp, tracks, self.name)
        return track_data

    def track(self, detections_in, platform, **kwargs):
        # wrap AVstack detections to SS detections
        detections = set()
        timestamp = self.t0 + timedelta(seconds=detections_in.timestamp)
        for det in detections_in:
            state_vector, metadata = self._convert_detection(det)
            detection = Detection(
                state_vector=state_vector,
                timestamp=timestamp,
                metadata=metadata,
            )
            detections.add(detection)

        # run tracker
        self.tracker.detector_iter = iter([(timestamp, detections)])
        timestamp, self.tracks = next(self.tracker)
        return self.tracks


@MODELS.register_module()
class StoneSoupKalmanTracker2DBox(StoneSoupKalmanTrackerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # track filtering
        # state is: [x, xdot, y, ydot, width, height] where x, y are top left, coordinates
        t_models = [
            ConstantVelocity(20**2),  # x
            ConstantVelocity(20**2),  # y
            RandomWalk(20**2),  # width
            RandomWalk(20**2),  # height
        ]
        transition_model = CombinedLinearGaussianTransitionModel(t_models)
        # detection is [x, y, width, height]
        measurement_model = LinearGaussian(
            ndim_state=6,
            mapping=[0, 2, 4, 5],
            noise_covar=np.diag([5**2, 5**2, 3**2, 3**2]),
        )
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)

        # data association
        hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis())
        data_associator = GNNWith2DAssignment(hypothesiser)

        # track management
        prior_state = GaussianState(
            StateVector(np.zeros((6, 1))),
            CovarianceMatrix(
                np.diag([100**2, 30**2, 100**2, 30**2, 500**2, 500**2])
            ),
        )
        deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
        initiator = MultiMeasurementInitiator(
            prior_state,
            deleter_init,
            data_associator,
            updater,
            measurement_model,
            min_points=3,
        )
        deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)

        # attributes
        self.tracker = MultiTargetTracker(
            initiator=initiator,
            detector=None,
            deleter=deleter,
            data_associator=data_associator,
            updater=updater,
        )

    @classmethod
    def _convert_detection(cls, det):
        box = det.box.box2d
        class_ = det.obj_type
        score = det.score
        metadata = {
            "raw_box": box,
            "class": cls.category_index[class_],
            "score": score,
        }
        # Transform box to be in format (x, y, w, h)
        state_vector = StateVector(
            [box[0], box[1], (box[2] - box[0]), (box[3] - box[1])]
        )
        return state_vector, metadata


@MODELS.register_module()
class StoneSoupKalmanTracker3DBox(StoneSoupKalmanTrackerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # track filtering
        # state is: [x, xdot, y, ydot, z, zdot, height, width, length, yaw]
        t_models = [
            ConstantVelocity(10**2),  # x
            ConstantVelocity(10**2),  # y
            ConstantVelocity(10**2),  # z
            RandomWalk(1**2),  # height
            RandomWalk(1**2),  # width
            RandomWalk(1**2),  # length
            RandomWalk(0.1**2),  # yaw
        ]
        transition_model = CombinedLinearGaussianTransitionModel(t_models)
        # detection is [x, y, z, height, width, length, yaw]
        measurement_model = LinearGaussian(
            ndim_state=10,
            mapping=[0, 2, 4, 6, 7, 8, 9],
            noise_covar=np.diag(
                [2**2, 2**2, 2**2, 0.5**2, 0.5**2, 0.5**2, 0.1**2]
            ),
        )
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)

        # data association
        hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis())
        data_associator = GNNWith2DAssignment(hypothesiser)

        # track management
        prior_state = GaussianState(
            StateVector(np.zeros((10, 1))),
            CovarianceMatrix(
                np.diag(
                    [
                        20**2,
                        10**2,
                        20**2,
                        10**2,
                        20**2,
                        10**2,
                        5**2,
                        5**2,
                        5**2,
                        1**2,
                    ]
                )
            ),
        )
        deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
        initiator = MultiMeasurementInitiator(
            prior_state,
            deleter_init,
            data_associator,
            updater,
            measurement_model,
            min_points=3,
        )
        deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)

        # attributes
        self.tracker = MultiTargetTracker(
            initiator=initiator,
            detector=None,
            deleter=deleter,
            data_associator=data_associator,
            updater=updater,
        )

    @classmethod
    def _convert_detection(cls, det):
        box = det.box.box3d
        class_ = det.obj_type
        score = det.score
        metadata = {
            "raw_box": box,
            "class": cls.category_index[class_],
            "score": score,
        }
        # Transform box to be in format (x, y, z, h, w, l, yaw)
        state_vector = StateVector(box)
        return state_vector, metadata
