from fpi_crane_hw.keyboard_joy_node import build_joy_state


def test_build_joy_state_maps_keyboard_inputs():
    axes, buttons = build_joy_state({"w", "d", "q", "i", "e", "l"})

    assert axes[0] == 1.0
    assert axes[1] == -1.0
    assert axes[3] == 1.0
    assert axes[4] == -1.0
    assert buttons[4] == 1
    assert buttons[5] == 1


def test_build_joy_state_clears_state_when_keys_not_pressed():
    axes, buttons = build_joy_state(set())

    assert axes == [0.0] * 6
    assert buttons == [0] * 6
