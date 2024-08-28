from typing import TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:
    from .instantaneous import ConfusionMatrix

import numpy as np


class ConfusionMatrixArray:
    def __init__(self, confusions: List["ConfusionMatrix"]):
        self.confusions = confusions

    @property
    def precisions(self):
        return [conf.precision for conf in self.confusions]

    @property
    def recalls(self):
        return [conf.recall for conf in self.confusions]

    def average_precision(self) -> Tuple[float, float, float]:
        """Average precision computation"""
        prec = self.precisions
        rec = self.recalls
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)
        ap = 0.0
        for i in i_list:
            try:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
            except IndexError:
                pass
        return ap, mpre, mrec

    def log_average_miss_rate(self) -> Tuple[float, float, float]:
        """Log-average miss rate computation

        Calculated by averaging miss rates at 9 evenly spaced FPPI points
        between 10e-2 and 10e0, in log-space.

        output:
            lamr | log-average miss rate
            mr | miss rate
            fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
            State of the Art." Pattern Analysis and Machine Intelligence, IEEE
            Transactions on 34.4 (2012): 743 - 761.
        """
        prec = self.precisions
        rec = self.recalls
        prec = np.asarray(prec)
        rec = np.asarray(rec)

        # if there were no detections of that class
        if prec.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        fppi = 1 - prec
        mr = 1 - rec

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num=9)
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = np.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi


def get_timeseries_metrics():
    raise NotImplementedError
