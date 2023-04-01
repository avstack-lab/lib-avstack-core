# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-27
# @Description:
"""

"""

import sys
import time

import numpy as np

import avstack.datastructs as ds


sys.path.append("tests/")
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


def generate_data(dt_interval=0.05, n_data=100):
    data = {
        dt_interval
        * frame: ds.DataContainer(
            frame,
            dt_interval * frame,
            np.array([dt_interval * frame]),
            source_identifier="0",
        )
        for frame in range(n_data)
    }
    return data


def test_data_buffer():
    max_size = 30
    buffer = ds.BasicDataBuffer(max_size=max_size)
    data = generate_data()
    for i, dc in enumerate(data.values()):
        buffer.push(dc)
    assert len(buffer.data[buffer.data_ids()[0]]) == max_size


def test_delay_managed_buffer_event_driven_emit_one():
    max_size = 30
    dt_delay = 0.1
    dt_interval = 0.05
    buffer = ds.DelayManagedDataBuffer(
        max_size=max_size, dt_delay=dt_delay, method="event-driven"
    )
    data = generate_data(dt_interval=dt_interval)
    for i, dc in enumerate(data.values()):
        buffer.push(dc)
        elements = buffer.emit_one()
        if i > 1:
            assert len(elements) == 1, len(elements)
            assert len(elements[list(elements.keys())[0]]) == 1
            assert len(buffer[buffer.keys[0]]) == 2
        else:
            assert len(elements) == 0


def test_delay_managed_buffer_real_time_emit_one():
    max_size = 30
    dt_delay = 0.01
    dt_interval = 0.005
    buffer = ds.DelayManagedDataBuffer(
        max_size=max_size, dt_delay=dt_delay, method="real-time"
    )
    data = generate_data(dt_interval=dt_interval, n_data=20)
    for i, dc in enumerate(data.values()):
        time.sleep(dt_interval + 1e-4)
        buffer.push(dc)
        elements = buffer.emit_one()
        if i > 1:
            assert len(elements) == 1, len(elements)
            assert len(elements[list(elements.keys())[0]]) == 1
            assert len(buffer[buffer.keys[0]]) == 2
        else:
            assert len(elements) == 0


def test_delay_managed_buffer_event_driven_emit_all():
    max_size = 30
    dt_delay = 0.2
    dt_interval = 0.05
    n_data = 20
    buffer = ds.DelayManagedDataBuffer(
        max_size=max_size, dt_delay=dt_delay, method="event-driven"
    )
    data = generate_data(dt_interval=dt_interval, n_data=n_data)
    for i, dc in enumerate(data.values()):
        buffer.push(dc)

    assert len(buffer[buffer.keys[0]]) == n_data
    elements = buffer.emit_all()
    assert len(elements) == 1, len(elements)
    n_popped = int((n_data * dt_interval - dt_delay) / dt_interval)
    assert len(elements[list(elements.keys())[0]]) == min(n_data, n_popped)
