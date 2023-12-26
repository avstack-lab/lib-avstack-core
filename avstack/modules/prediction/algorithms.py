import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


n_procs_max = max(1, os.cpu_count() // 2)


class _PredictionAlgorithm(BaseModule):
    def __init__(self, name="prediction", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    @apply_hooks
    def __call__(self, objects, frame, *args, **kwargs):
        """
        TODO: could make saving faster with multiproc too
        """
        predictions = self._predict_objects(objects, *args, **kwargs)
        return predictions


class KinematicPrediction(_PredictionAlgorithm):
    def __init__(self, dt_pred, t_pred_forward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_forward = t_pred_forward
        self.dt = dt_pred
        self.dt_predicts = np.arange(0 + self.dt, self.t_forward + self.dt, self.dt)

    def _predict_objects(self, objects, *args, use_pool=False, **kwargs):
        pred_objs = {}
        part_func = partial(self._predict_per_object, self.dt_predicts)
        if use_pool:
            with Pool(max(1, min(n_procs_max, len(objects)))) as p:
                pred_objs = p.map(part_func, objects)
        else:
            pred_objs = [part_func(obj) for obj in objects]
        return {obj.ID: pred for obj, pred in zip(objects, pred_objs)}

    @staticmethod
    def _predict_per_object(dt_predicts, obj):
        pred_obj = {}
        try:
            obj = obj.as_object()
        except AttributeError as e:
            pass
        for dt in dt_predicts:
            pred_obj[dt] = obj.predict(dt)
        return pred_obj
