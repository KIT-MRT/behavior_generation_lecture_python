import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils.vizard.vizard as vz


class FakeButtonPressedEvent:
    def __init__(self, key):
        self.key = key


def test_buttons_up_down():
    matplotlib.use("Agg")

    fig = plt.figure()
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(326)
    t = np.linspace(0, 6 * np.pi, num=100)
    sin_t = np.sin(t)

    ax1.plot(t, sin_t)
    ax2.plot(t, sin_t)
    (point1,) = ax1.plot([], [], marker="o", color="crimson", ms=15)
    (point2,) = ax2.plot([], [], marker="x", color="crimson", ms=15)

    def update(i, *fargs):
        point1.set_data(t[i], sin_t[i])
        point2.set_data(t[i], sin_t[i])
        for farg in fargs:
            print(farg)

    viz = vz.Vizard(figure=fig, update_func=update, time_vec=t)
    plt.show()

    button_up_pressed_event = FakeButtonPressedEvent(u"up")
    viz.key_press_event(button_up_pressed_event)
    assert viz.i == 10, "Animation should be at frame 10 after pressing button 'up'"

    button_down_pressed_event = FakeButtonPressedEvent(u"down")
    viz.key_press_event(button_down_pressed_event)
    viz.key_press_event(button_down_pressed_event)
    assert viz.i == 0, "Animation frame should be 0 at least"
