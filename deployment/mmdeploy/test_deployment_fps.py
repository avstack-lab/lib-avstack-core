from mmdeploy_runtime import Detector
import os
import cv2
import time
import numpy as np


def run_with_timing(detector, img, n_inferences):
    timing = []
    for _ in range(n_inferences):
        t1 = time.time()
        bboxes, labels, _ = detector(img)
        t2 = time.time()
        timing.append((t2 - t1)/1000)
    print(f'Timing Results:\n   mean: {np.mean(timing):4.2f} ms,  std: {np.mean(timing):4.2f}')


def main_mmdeploy(img, n_inferences=10):
    detector = Detector(
        model_path='mmdeploy_models/cascade_rcnn_coco',
        device_name='cuda',
        device_id=0
    )
    run_with_timing(detector, img, n_inferences)


def main_mmdet(img, n_inferences=10):
    pass


if __name__ == "__main__":
    MMDET_BASE = "../../third_party/mmdetection"
    img = cv2.imread(os.path.join(MMDET_BASE, "demo", "demo.jpg"))
    n_inferences = 20
    main_mmdeploy(img, n_inferences=n_inferences)
    main_mmdet(img, n_inferences=n_inferences)