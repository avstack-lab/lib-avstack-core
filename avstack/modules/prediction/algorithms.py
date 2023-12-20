import os
import shutil
from functools import partial
from multiprocessing import Pool

import numpy as np

from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


n_procs_max = max(1, os.cpu_count() // 2)


class _PredictionAlgorithm(BaseModule):
    def __init__(self, save_output=False, save_folder="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save = save_output
        self.save_folder = os.path.join(save_folder, "prediction")
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)

    @apply_hooks
    def __call__(self, objects, frame, *args, **kwargs):
        """
        TODO: could make saving faster with multiproc too
        """
        predictions = self._predict_objects(objects, *args, **kwargs)
        if self.save:
            preds = []
            for obj_ID in predictions:
                for dt in predictions[obj_ID]:
                    preds.append(predictions[obj_ID][dt].encode())
            fname = os.path.join(self.save_folder, "%06i.txt" % frame)
            with open(fname, "w") as f:
                f.write("\n".join(preds))
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
