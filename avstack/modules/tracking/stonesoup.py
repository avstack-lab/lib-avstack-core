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
from avstack.geometry import (
    Attitude,
    Box2D,
    Box3D,
    Position,
    ReferenceFrame,
    transform_orientation,
)
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


@MODELS.register_module()
class StoneSoupKalmanTrackerBase(BaseModule):
    category_index = {
        "car": {"id": 1, "name": "car"},
        "truck": {"id": 2, "name": "truck"},
        "bicycle": {"id": 3, "name": "bicycle"},
        "motorcycle": {"id": 4, "name": "motorcycle"},
    }

    def __init__(self, t0=datetime.now(), **kwargs):
        super().__init__(name="tracking", **kwargs)
        self.iframe = -1
        self.frame = 0
        self.timestamp = 0
        # initial time
        self.t0 = t0

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

    def track(self, detections_in, platform, calibration, **kwargs):
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
        self._last_detections = detections
        self.tracker.detector_iter = iter([(timestamp, detections)])
        timestamp, self.tracks = next(self.tracker)
        for track in self.tracks:
            self._augment_track(track, calibration)
        return self.tracks

    @classmethod
    def _convert_detection(cls, det):
        raise NotImplementedError

    @staticmethod
    def _augment_track(track):
        raise NotImplementedError


@MODELS.register_module()
class StoneSoupKalmanTracker2DBox(StoneSoupKalmanTrackerBase):
    def __init__(
        self,
        missed_distance=30,
        qx=20**2,
        qb=20**2,
        px=100**2,
        pv=30**2,
        pb=500**2,
        rx=1**2,
        rb=0.25**2,
        init_deleter_steps=3,
        init_min_points=3,
        deleter_steps=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # track filtering
        # state is: [x, xdot, y, ydot, width, height] where x, y are top left, coordinates
        t_models = [
            ConstantVelocity(qx),  # x
            ConstantVelocity(qx),  # y
            RandomWalk(qb),  # width
            RandomWalk(qb),  # height
        ]
        transition_model = CombinedLinearGaussianTransitionModel(t_models)
        # detection is [x, y, width, height]
        measurement_model = LinearGaussian(
            ndim_state=6,
            mapping=[0, 2, 4, 5],
            noise_covar=np.diag([*[rx] * 2, *[rb] * 2]),
        )
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)

        # data association - missed_distance is gate
        hypothesiser = DistanceHypothesiser(
            predictor, updater, Mahalanobis(), missed_distance=missed_distance
        )
        data_associator = GNNWith2DAssignment(hypothesiser)

        # track management
        prior_state = GaussianState(
            StateVector(np.zeros((6, 1))),
            CovarianceMatrix(np.diag([*[px, pv] * 2, *[pb] * 2])),
        )
        deleter_init = UpdateTimeStepsDeleter(
            time_steps_since_update=init_deleter_steps
        )
        initiator = MultiMeasurementInitiator(
            prior_state,
            deleter_init,
            data_associator,
            updater,
            measurement_model,
            min_points=init_min_points,
        )
        deleter = UpdateTimeStepsDeleter(time_steps_since_update=deleter_steps)

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

    @staticmethod
    def _augment_track(track, calibration):
        x0, y0, w, h = np.array(track.state_vector[[0, 2, 4, 5]])
        box2d = [x0, y0, x0 + w, y0 + h]
        ID = track.id
        obj_type = track.metadata["class"]["name"]
        track.box = Box2D(box2d, calibration, ID=ID, obj_type=obj_type)


@MODELS.register_module()
class StoneSoupKalmanTracker3DBox(StoneSoupKalmanTrackerBase):
    def __init__(
        self,
        missed_distance=5,
        qx=0.1,
        qb=0.01,
        qy=0.1,
        px=5**2,
        pv=10**2,
        pb=2**2,
        py=0.5**2,
        rx=1**2,
        rb=0.25**2,
        ry=0.1**2,
        init_deleter_steps=3,
        init_min_points=3,
        deleter_steps=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # track filtering
        # state is: [x, xdot, y, ydot, z, zdot, height, width, length, yaw]
        # CV noise parameter is 2*sigma_m^2*tau_m according to Blackman
        t_models = [
            ConstantVelocity(qx),  # x
            ConstantVelocity(qx),  # y
            ConstantVelocity(qx),  # z
            RandomWalk(qb),  # height
            RandomWalk(qb),  # width
            RandomWalk(qb),  # length
            RandomWalk(qy),  # yaw
        ]
        transition_model = CombinedLinearGaussianTransitionModel(t_models)
        # detection is [x, y, z, height, width, length, yaw]
        measurement_model = LinearGaussian(
            ndim_state=10,
            mapping=[0, 2, 4, 6, 7, 8, 9],
            noise_covar=np.diag([*[rx] * 3, *[rb] * 3, ry]),
        )
        predictor = KalmanPredictor(transition_model)
        updater = KalmanUpdater(measurement_model)

        # data association - missed dist is gate
        hypothesiser = DistanceHypothesiser(
            predictor, updater, Mahalanobis(), missed_distance=missed_distance
        )
        data_associator = GNNWith2DAssignment(hypothesiser)

        # track management
        prior_state = GaussianState(
            StateVector(np.zeros((10, 1))),
            CovarianceMatrix(np.diag([*[px, pv] * 3, *[pb] * 3, py])),
        )
        deleter_init = UpdateTimeStepsDeleter(
            time_steps_since_update=init_deleter_steps
        )
        initiator = MultiMeasurementInitiator(
            prior_state,
            deleter_init,
            data_associator,
            updater,
            measurement_model,
            min_points=init_min_points,
        )
        deleter = UpdateTimeStepsDeleter(time_steps_since_update=deleter_steps)

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

    @staticmethod
    def _augment_track(track, calibration):
        # TODO: fix global origin usage
        position = Position(
            np.array(track.state_vector[[0, 2, 4]]).reshape(3),
            calibration.reference,
        )
        attitude = Attitude(
            transform_orientation([0, 0, track.state_vector[9]], "euler", "quat"),
            calibration.reference,
        )
        hwl = np.array(track.state_vector[[6, 7, 8]]).reshape(3)
        track.box3d = Box3D(position, attitude, hwl, where_is_t="bottom")
