import sys

from avstack.geometry import GlobalOrigin3D
from avstack.modules.tracking import tracks


sys.path.append("tests/")


def test_score_tracks():
    reference = GlobalOrigin3D
    razel = [100, 0, 0]
    track = tracks.XyzFromRazelTrack(
        t0=0.0, razel=razel, reference=reference, obj_type="car"
    )
    assert not track.confirmed
    for _ in range(20):
        track.update(razel)
    assert track.confirmed
    assert track.probability > 0.99


def test_coast():
    reference = GlobalOrigin3D
    razel = [100, 0, 0]
    track = tracks.XyzFromRazelTrack(
        t0=0.0, razel=razel, reference=reference, obj_type="car"
    )
    assert not track.confirmed
    dt = 0.5
    for i in range(20):
        track.predict(t=i * dt)
    assert not track.active
