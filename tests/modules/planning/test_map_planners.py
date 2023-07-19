# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-26
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""

import sys


sys.path.append("tests/")


# def test_carla_map_based_planner():
#     found_egg, msg = avstack.utils.add_carla_eggs(raise_on_error=False)
#     if found_egg:
#         try:
#             import carla
#             client = carla.Client('localhost', 2000)
#             client.set_timeout(0.5)
#             try:
#                 world = client.get_world()
#             except RuntimeError as e:
#                 print('Could not find carla simulator running')
#                 return
#             map_data = world.get_map()
#             ego_init = get_ego(seed=1, frame=GlobalOrigin3D)
#             planner = avstack.modules.planning.vehicle.MapBasedPlanningAndControl(ego_init, map_data)
#             destination = ego_init.position + 100*ego_init.attitude.forward_vector
#             planner.set_destination(destination)
#             environment = avstack.environment.EnvironmentState()
#             for i in range(20):
#                 ego_state = ego_init
#                 objects_3d = []
#                 objects_2d = []
#                 ctrl = planner(ego_state, environment, objects_3d, objects_2d)
#                 assert isinstance(ctrl, avstack.modules.control.vehicle.VehicleControlSignal)
#                 assert ctrl.throttle > 0.5
#         except ImportError as e:
#             print('Could not import carla')
#     else:
#         print('Could not find carla egg...here is message:{}'.format(msg))
