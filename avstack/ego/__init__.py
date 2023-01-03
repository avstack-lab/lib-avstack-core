# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-30
# @Description:
"""

"""

import avstack.ego.vehicle



def get_ego(obj_class, obj_stack):
    """Get uninitialized stack"""
    if obj_class.lower() == 'vehicle':
        if obj_stack.lower() == 'level2lidarbasedvehicle':
            stack = avstack.ego.vehicle.Level2LidarBasedVehicle
        elif obj_stack.lower() == 'level2groundtruthperception':
            stack = avstack.ego.vehicle.Level2GroundTruthPerception
        elif obj_stack.lower() == 'level2gtperceptiongtlocalization':
            stack = avstack.ego.vehicle.Level2GtPerceptionGtLocalization
        elif obj_stack.lower() == 'groundtruthmapplanner':
            stack = avstack.ego.vehicle.GroundTruthMapPlanner
        elif obj_stack.lower() == 'passthroughautopilotvehicle':
            stack = avstack.ego.vehicle.PassthroughAutopilotVehicle
        else:
            raise NotImplementedError(obj_stack.lower())
    else:
        raise NotImplementedError
    return stack