from typing import List
import numpy as np


# ==================================================
# PSEUDO-PACKETS ARE PURE-PYTHON OBJECTS
# ==================================================

def get_velodyne_firing_time(sensor: str) -> float:
    """Returns the firing timing for a velodyne sensor"""
    if sensor is None:
        raise NotImplementedError
    elif sensor.lower() == 'vlp-16':
        firing_time = 55.296e-6  # for all 16 lasers
    elif sensor.lower() == 'hdl-32e':
        firing_time = 46.080e-6 # for all 32 lasers
    elif sensor.lower() == 'hdl-64e':
        firing_time = 50.0e-6
    elif sensor.lower() in ['hdl-64e-s2', 'hdl-64e-s2.1']:
        firing_time = 48.0e-6  # 32 of upper synchedwith 32 of lower
    else:
        raise NotImplementedError(sensor)
    return firing_time


def get_velodyne_elevation_table(sensor: str) -> dict:
    """Returns the discrete elevation angles of the sensor"""
    if sensor.lower() == 'vlp-16':
        angs = [-15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        assert len(angs) == 16
    elif sensor.lower() == 'hdl-32e':
        angs = [-30.67, -9.33, -29.33, -8.00, -28.00, -6.67, -26.67, -5.33, -25.33,
            -4.00, -24.00, -2.67, -22.67, -1.33, -21.33, 0.00, -20.00, 1.33, -18.67,
            2.67, -17.33, 4.00, -16.00, 5.33, -14.67, 6.67, -13.33, 8.00, -12.00, 9.33,
            -10.67, 10.67]
        assert len(angs) == 32
    elif sensor.lower() == 'hdl-64e':
        angs = list(np.linspace(2, -24.9, 64))
    elif sensor.lower() == 'hdl-64e-s2':
        angs = list(np.arange(2, -8.33, -1/3)) + list(np.arange(-8.83, -24.33-0.5-1e-5, -0.5))
    elif sensor.lower() == 'hdl-64e-s2.1':
        angs = list(np.linspace(2, -29.5, 64))
    else:
        raise NotImplementedError('Sensor {} not implemented yet'.format(sensor))
    return {i:a for i, a in enumerate(angs)}


class VelodyneHeader():
    def __init__(self) -> None:
        pass


class VelodyneDataBlock():
    def __init__(self, azimuth: float, data: np.ndarray) -> None:
        self.azimuth = azimuth
        self.data = data


class VelodyneFactory():
    def __init__(self, mode, sensor) -> None:
        self.mode = mode
        self.sensor = sensor


class VelodynePseudoPacket():
    def __init__(self, header: VelodyneHeader, data_blocks: List[VelodyneDataBlock], 
            timestamp: float, factory: VelodyneFactory) -> None:
        self.header = header
        self.data_blocks = data_blocks
        self.timestamp = timestamp
        self.factor= factory