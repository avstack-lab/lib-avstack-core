# @Author: Spencer Hallyburton <spencer>
# @Date:   2021-02-24
# @Filename: __init__.py
# @Last modified by:   spencer
# @Last modified time: 2021-02-24


import avstack.modules
import avstack.ego
import avstack.environment
import avstack.geometry
import avstack.utils
import avstack.exceptions


class GroundTruthInformation():
    """Standardize the representation of ground truth data"""
    def __init__(self, frame, timestamp, ego_state, objects=[], lane_lines=[],
            environment=avstack.environment.EnvironmentState(), lane_id=None):
        self.frame = frame
        self.timestamp = timestamp
        self.ego_state = ego_state
        self.objects = objects
        self.lane_lines = lane_lines
        self.lane_id = lane_id
        self.environment = environment
