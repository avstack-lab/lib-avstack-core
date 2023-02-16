# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""

import os
import shutil


class _FusionAlgorithm:
    def __init__(self, save_output=False, save_folder="", **kwargs):
        self.iframe = -1
        self.save = save_output
        self.save_folder = save_folder
        self.save = save_output
        self.save_folder = os.path.join(save_folder, "fusion")
        if save_output:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder)

    def __call__(self, *args, **kwargs):
        self.iframe += 1
        self.frame = kwargs.get("frame")
        tracks = self.fuse(*args, **kwargs)
        if self.save:
            trk_str = "\n".join([trk.format_as("avstack") for trk in tracks])
            fname = os.path.join(self.save_folder, "%06d.txt" % kwargs.get("frame"))
            with open(fname, "w") as f:
                f.write(trk_str)
        return tracks
