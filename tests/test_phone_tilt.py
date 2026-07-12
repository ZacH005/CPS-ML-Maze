from cps_maze.control.phone_tilt import PhoneTiltConfig, is_stale, map_orientation


def test_inside_deadzone_maps_to_neutral():
    cfg = PhoneTiltConfig(deadzone_deg=3.0, max_tilt_deg=25.0, max_tilt=0.9)
    yaw, pitch = map_orientation(2.0, -2.0, cfg)
    assert yaw == 0.0
    assert pitch == 0.0


def test_beyond_max_tilt_deg_saturates():
    cfg = PhoneTiltConfig(deadzone_deg=3.0, max_tilt_deg=25.0, max_tilt=0.9)
    yaw, pitch = map_orientation(90.0, -90.0, cfg)
    assert yaw == 0.9
    assert pitch == -0.9


def test_invert_flags_flip_sign():
    cfg = PhoneTiltConfig(deadzone_deg=3.0, max_tilt_deg=25.0, max_tilt=0.9,
                          yaw_sign=-1.0, pitch_sign=-1.0)
    yaw, pitch = map_orientation(10.0, 10.0, cfg)
    assert yaw < 0.0
    assert pitch < 0.0


def test_swap_axes_swaps_which_input_drives_which_channel():
    cfg = PhoneTiltConfig(deadzone_deg=3.0, max_tilt_deg=25.0, max_tilt=0.9)
    swapped_cfg = PhoneTiltConfig(deadzone_deg=3.0, max_tilt_deg=25.0, max_tilt=0.9,
                                  swap_axes=True)
    yaw, pitch = map_orientation(10.0, 20.0, cfg)
    swapped_yaw, swapped_pitch = map_orientation(10.0, 20.0, swapped_cfg)
    assert swapped_yaw == pitch
    assert swapped_pitch == yaw


def test_is_stale():
    assert is_stale(age_s=0.5, timeout_s=0.35) is True
    assert is_stale(age_s=0.1, timeout_s=0.35) is False
