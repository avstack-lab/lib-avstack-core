# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""

import sys
import avstack.datastructs as ds

sys.path.append('tests/')
from utilities import get_lidar_data


def test_data_manager():
    data_manager = ds.DataManager()
    pc1 = get_lidar_data(0.0, 1)
    data_manager.push(pc1)
    assert data_manager.has_data(pc1.source_identifier)
    pc2 = get_lidar_data(1.0, 100)
    data_manager.push(pc2)
    pc_got = data_manager.pop(pc1.source_identifier)
    assert pc_got == pc1
