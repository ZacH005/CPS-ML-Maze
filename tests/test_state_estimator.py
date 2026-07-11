import numpy as np

from cps_maze.vision.state_estimator import LowPassVelocityEstimator


def test_velocity_estimator_computes_smoothed_velocity():
    estimator = LowPassVelocityEstimator(alpha=1.0)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.0)
    state = estimator.update(np.array([10.0, 0.0]), timestamp_s=2.0)

    assert np.allclose(state.velocity_mm_s, [5.0, 0.0])


def test_velocity_estimator_reset_clears_previous_sample():
    estimator = LowPassVelocityEstimator(alpha=1.0)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.0)
    estimator.update(np.array([10.0, 0.0]), timestamp_s=2.0)
    estimator.reset()
    state = estimator.update(np.array([100.0, 0.0]), timestamp_s=3.0)

    assert np.allclose(state.velocity_mm_s, [0.0, 0.0])


def test_burst_frame_does_not_manufacture_velocity():
    # A frame ~0 ms after the previous one (camera burst) with a tiny position
    # jitter must NOT produce a huge velocity, and must not poison the estimate:
    # the next well-spaced frame should measure velocity over the true interval.
    estimator = LowPassVelocityEstimator(alpha=1.0, min_dt_s=0.006)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.000)
    burst = estimator.update(np.array([0.3, 0.0]), timestamp_s=0.001)  # +1 ms
    assert np.allclose(burst.velocity_mm_s, [0.0, 0.0])  # not 300 mm/s
    # fresh position is still served for control
    assert np.allclose(burst.position_mm, [0.3, 0.0])

    # next real frame measures over the true interval from the last good sample
    state = estimator.update(np.array([2.0, 0.0]), timestamp_s=0.020)
    assert np.allclose(state.velocity_mm_s, [100.0, 0.0])  # 2 mm / 0.02 s


def test_detection_jump_is_clamped():
    # A several-mm position jump at a normal dt (tracker mislock) is clamped in
    # magnitude before smoothing, so it cannot spike the speed.
    estimator = LowPassVelocityEstimator(alpha=1.0, max_speed_mm_s=250.0)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.0)
    state = estimator.update(np.array([10.0, 0.0]), timestamp_s=0.016)  # 625 mm/s raw
    assert np.isclose(np.linalg.norm(state.velocity_mm_s), 250.0)
