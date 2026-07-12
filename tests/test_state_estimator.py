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


def test_smoothing_is_frame_rate_invariant():
    # The same jittery position signal, sampled at 10 fps and at 120 fps for the
    # same real duration, must produce similar smoothed speed. A fixed per-frame
    # alpha fails this (it smooths far less at high fps); the time-constant filter
    # passes it. This is the regression that stopped the ball moving at 120 fps.
    rng_amp = 0.5  # mm of position jitter around a parked ball
    tau = 0.10

    def parked_speed(fps: float) -> float:
        est = LowPassVelocityEstimator(tau_s=tau)
        dt = 1.0 / fps
        speeds = []
        t = 0.0
        # deterministic alternating jitter (worst case for a difference filter)
        for i in range(int(2.0 * fps)):
            x = rng_amp if i % 2 == 0 else -rng_amp
            s = est.update(np.array([x, 0.0]), timestamp_s=t)
            speeds.append(float(np.linalg.norm(s.velocity_mm_s)))
            t += dt
        return float(np.mean(speeds[-int(fps):]))  # last ~1 s

    slow = parked_speed(10.0)
    fast = parked_speed(120.0)
    # Both should read a small parked speed, and the fast loop must not be wildly
    # noisier than the slow one (the old fixed-alpha filter made it ~10x worse).
    assert fast < 3.0 * slow + 5.0


def test_detection_jump_is_clamped():
    # A several-mm position jump at a normal dt (tracker mislock) is clamped in
    # magnitude before smoothing, so it cannot spike the speed.
    estimator = LowPassVelocityEstimator(alpha=1.0, max_speed_mm_s=250.0)

    estimator.update(np.array([0.0, 0.0]), timestamp_s=0.0)
    state = estimator.update(np.array([10.0, 0.0]), timestamp_s=0.016)  # 625 mm/s raw
    assert np.isclose(np.linalg.norm(state.velocity_mm_s), 250.0)
