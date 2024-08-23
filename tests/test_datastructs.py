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
from avstack.geometry import GlobalOrigin3D, ReferenceFrame


sys.path.append("tests/")
from utilities import get_lidar_data, get_object_global


def test_priority_min_heap_full():
    # min heap should give the lowest value first
    # non-circular means lowest popped if full
    max_size = 10
    queue = ds.PriorityQueue(max_size=max_size, max_heap=False, circular=False)
    for i in range(20):
        queue.push(priority=i, item=None)
        assert queue.top()[0] == max(0, i - max_size + 1)


def test_priority_max_heap_full():
    # max heap should give the highest value first
    # non-circular means highest popped if full
    max_size = 10
    queue = ds.PriorityQueue(max_size=max_size, max_heap=True, circular=False)
    for i in range(20):
        queue.push(priority=i, item=None)
        assert queue.top()[0] == min(i, max_size - 1)


def test_priority_min_heap_full_circular():
    # min heap should give the lowest value first
    # circular means highest popped if full
    max_size = 10
    queue = ds.PriorityQueue(max_size=max_size, max_heap=False, circular=True)
    for i in range(20):
        queue.push(priority=i, item=None)
        assert queue.top()[0] == 0


def test_priority_max_heap_full_circular():
    # max heap should give the highest value first
    # circular means lowest popped if full
    max_size = 10
    queue = ds.PriorityQueue(max_size=max_size, max_heap=True, circular=True)
    for i in range(20):
        queue.push(priority=i, item=None)
        assert queue.top()[0] == i


def get_object_dc():
    frame = timestamp = source_identifier = 0
    data = [get_object_global(seed=i, reference=GlobalOrigin3D) for i in range(4)]
    dc = ds.DataContainer(
        frame=frame, timestamp=timestamp, data=data, source_identifier=source_identifier
    )
    return dc


def test_datacontainer_apply():
    dc1 = get_object_dc()
    ref2 = ReferenceFrame(
        x=np.array([1, 2, 3]), q=np.quaternion(1), reference=GlobalOrigin3D
    )
    ref3 = ReferenceFrame(
        x=np.array([1, 1, 1]), q=np.quaternion(1), reference=GlobalOrigin3D
    )

    # check the initial
    ids = []
    id_dc1 = id(dc1)
    for item in dc1:
        assert item.reference == GlobalOrigin3D
        ids.append(id(item))

    # use the apply version -- inplace
    dc1.apply("change_reference", reference=ref2, inplace=True)
    for idx, item in enumerate(dc1):
        assert item.reference == ref2
        assert id(item) == ids[idx]
    assert id(dc1) == id_dc1

    # use the apply version -- not inplace
    dc1.apply("change_reference", reference=ref3, inplace=False)
    for idx, item in enumerate(dc1):
        assert item.reference == ref3
        assert id(item) != ids[idx]
    assert id(dc1) == id_dc1


def test_datacontainer_apply_and_return():
    dc1 = get_object_dc()
    ref2 = ReferenceFrame(
        x=np.array([1, 2, 3]), q=np.quaternion(1), reference=GlobalOrigin3D
    )
    dc2 = dc1.apply_and_return("change_reference", reference=ref2, inplace=False)
    assert id(dc1) != id(dc2)
    for el1, el2 in zip(dc1, dc2):
        assert id(el1) != id(el2)


def test_data_manager():
    data_manager = ds.DataManager()
    pc1 = get_lidar_data(0.0, 1)
    data_manager.push(pc1)
    assert data_manager.has_data(pc1.source_identifier)
    pc2 = get_lidar_data(1.0, 100)
    data_manager.push(pc2)
    pc_got = data_manager.pop(pc1.source_identifier, with_priority=False)
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
        elements = buffer.emit_one(with_priority=False)
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
