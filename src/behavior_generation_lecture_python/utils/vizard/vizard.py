import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.widgets
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


class Vizard(object):
    """Class around matplotlib's FuncAnimation for managing
    and controlling plot animations via GUI elements.
    """

    def __init__(self, figure, update_func, time_vec):
        """Constructor

        Args:
            interval: Update interval of animation's event source
                (timer).
        """
        self.plots = []
        self.func_animations = []
        self.event_source = None

        self.i = 0
        self.min = 0
        self.max = 0

        self.runs = True
        self.measure = False
        self.measuring = False
        self.measure_data = None

        self.register_plot(figure=figure, update_func=update_func, time_vec=time_vec)

        self.go_to(1)

    class VizardPlot(object):
        """Helper class for storing plot fixtures."""

        def __init__(self):
            self.figure = None
            self.init_func = None
            self.update_func = None
            self.time_vec = None
            self.anim = None
            self.controls = None

    class VizardControls(object):
        """Helper class for storing animation controls."""

        def __init__(self):
            self.button_stop = None
            self.button_playpause = None
            self.button_oneback = None
            self.button_oneforward = None
            self.slider = None
            self.button_measure = None
            self.text_measure = None

    def register_plot(self, figure, update_func, time_vec, init_func=None, blit=False):
        """Create new FuncAnimation instance
        with given plot class.
        """
        if not isinstance(figure, matplotlib.figure.Figure):
            raise RuntimeError("Figure is not a matplotlib figure!")
        if not callable(update_func):
            raise RuntimeError("Update function is not callable!")
        if len(time_vec) < 1:
            raise RuntimeError("Timestamps vector must not be empty!")

        self.min = 0
        self.max = len(time_vec) - 1

        # Draw controls below figure's axes
        controls = self.setup(figure)

        # Augment update function (bind with controls)
        update_func_aug = lambda *args: self.update_plot(controls, update_func, *args)

        # Create instance of FuncAnimation
        anim = FuncAnimation(
            figure,
            update_func_aug,
            init_func=init_func,
            frames=self.frames,
            event_source=self.event_source,
            blit=blit,
        )

        # Store plot supplies for future modification
        plot = self.VizardPlot()
        plot.figure = figure
        plot.controls = controls
        plot.anim = anim
        plot.init_func = init_func
        plot.update_func = update_func_aug
        plot.time_vec = time_vec
        self.plots.append(plot)

        # Get event source from Animation,
        # which is also used as event source for
        # all future Animation class instances
        self.event_source = plot.anim.event_source

    def frames(self):
        """Generator that returns the current plot index.

        Returns:
            Index.
        """
        while self.runs:
            self.i += 1
            if self.min < self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def play(self):
        """Start the function animation."""
        self.runs = True
        self.stop_measurement()
        self.event_source.start()
        self.set_play_pause_label("pause")

    def pause(self):
        """Stop the function animation."""
        self.runs = False
        self.stop_measurement()
        self.event_source.stop()
        self.set_play_pause_label("run")

    def playpause(self, *args):
        """Toggle the play/pause state
        of function animation.
        """
        del args
        if self.runs:
            self.pause()
        else:
            self.play()

    def stop(self, *args):
        """Stop the function animation
        and go to first index.
        """
        del args
        self.pause()
        self.stop_measurement()
        self.i = 0
        self.update_plots()

    def oneforward(self, event=None):
        """Step one step forward and pause."""
        self.step(1)

    def onebackward(self, event=None):
        """Step one step backward and pause."""
        self.step(-1)

    def step(self, stepsize):
        """Step with given stepsize and
        pause the function animation.

        Args:
            stepsize: The stepsize to jump.
        """
        self.stop_measurement()
        self.pause()
        self.i = max(min(self.i + stepsize, self.max), self.min)
        self.update_plots()

    def toggle_measure(self, event):
        self.measure = not self.measure
        for plot in self.plots:
            button = plot.controls.button_measure
            button.color = "0.95" if self.measure else "0.85"
            text = plot.controls.text_measure
            text.set_text(
                "Draw line by holding left mouse button in a plot"
                if self.measure
                else ""
            )
        if not self.measure and self.measure_data:
            self.measure_data[3].remove()
            self.measure_data = None
        if event:
            event.canvas.draw()

    def stop_measurement(self, event=None):
        if self.measure:
            self.toggle_measure(event)

    def move_axes(self, event):
        if self.measuring:
            x, y, axis, line = self.measure_data
            if axis != event.inaxes:
                return
            dx = event.xdata - x
            dy = event.ydata - y
            for plot in self.plots:
                text = plot.controls.text_measure
                text.set_text(
                    "dx:{:.3} dy:{:.3} dist:{:.3}".format(
                        dx, dy, math.sqrt(dx ** 2 + dy ** 2)
                    )
                )
            line.set_data([[x, event.xdata], [y, event.ydata]])
            event.canvas.draw()

    def start_measurement(self, event):
        ax = event.inaxes
        if not self.measure:
            return
        self.measuring = True
        if self.measure_data:
            self.measure_data[3].remove()
        line = plt.Line2D([], [], linewidth=2, color="blue")
        ax.add_line(line)
        self.measure_data = (event.xdata, event.ydata, event.inaxes, line)

    def end_measurement(self, event):
        self.measuring = False

    def go_to(self, index):
        """Go to given index and update plots.

        Args:
            index: Index.
        """
        if self.i == int(index):
            return
        self.stop_measurement()
        self.i = int(index)
        self.update_plots()

    def setup(self, figure):
        """Draw controls in additional
        axes of the plot's figure.

        Args:
            figure: Handle to figure to add controls to.

        Returns:
            Control class holding all controls.
        """
        # Define control dimensions
        MARGIN_LEFT = 0.2
        MARGIN_BOTTOM = 0.2
        CONTROLS_HEIGHT = 0.2
        SLIDER_WIDTH = 2.5
        BUTTON_WIDTH = 0.5
        SLIDER_VAL_LABEL_WIDTH = 0.5
        SLIDER_MAX_VAL_LABEL_WIDTH = 0.5
        CONTROLS_SPACING = 0.05

        NR_BUTTONS = 4

        # Move all axes up to make room for the controls
        figure.subplots_adjust(bottom=CONTROLS_HEIGHT)

        button_widths = [
            Size.Fixed(BUTTON_WIDTH),
            Size.Fixed(CONTROLS_SPACING),
        ] * NR_BUTTONS

        h_grid = (
            [Size.Fixed(MARGIN_LEFT)]
            + button_widths
            + [
                Size.Fixed(SLIDER_VAL_LABEL_WIDTH),
                Size.Fixed(SLIDER_WIDTH),
                Size.Fixed(SLIDER_MAX_VAL_LABEL_WIDTH),
            ]
        )

        v_grid = [Size.Fixed(MARGIN_BOTTOM), Size.Fixed(CONTROLS_HEIGHT)]

        complete_figure = (0.0, 0.0, 1.0, 1.0)
        player_divider = Divider(figure, complete_figure, h_grid, v_grid, aspect=False)

        # Create axes for buttons
        ax_buttons = []
        for i in range(NR_BUTTONS):
            ax = Axes(figure, player_divider.get_position())
            ax.set_axes_locator(player_divider.new_locator(nx=2 * i + 1, ny=1))
            ax_buttons.append(ax)
            figure.add_axes(ax)

        # Create axes for slider
        ax_slider = Axes(figure, player_divider.get_position())
        ax_slider.set_axes_locator(
            player_divider.new_locator(nx=1 + 2 * NR_BUTTONS + 1, ny=1)
        )  # 2 accounts for the spacing between buttons
        figure.add_axes(ax_slider)

        # Create controls
        controls = self.VizardControls()
        controls.button_stop = matplotlib.widgets.Button(
            ax_buttons[0], label="\U000025A0"
        )
        controls.button_playpause = matplotlib.widgets.Button(
            ax_buttons[1], label="\U000025B6"
        )
        controls.button_oneback = matplotlib.widgets.Button(
            ax_buttons[2], label="\U000029CF"
        )
        controls.button_oneforward = matplotlib.widgets.Button(
            ax_buttons[3], label="\U000029D0"
        )
        controls.slider = matplotlib.widgets.Slider(
            ax_slider,
            label=self.max,
            valmin=self.min,
            valmax=self.max,
            valinit=self.i,
            valfmt="%i",
            valstep=1,
        )
        ax_measure_btn = figure.add_axes([0.75, 0.01, 0.05, 0.02])
        controls.button_measure = matplotlib.widgets.Button(
            ax_measure_btn, label="Measure"
        )

        controls.text_measure = figure.text(0.81, 0.015, "")

        self.set_play_pause_label("pause", controls=controls)
        controls.slider.valtext.set_position((-0.02, 0.5))
        controls.slider.valtext.set_horizontalalignment("right")
        controls.slider.label.set_position((1.02, 0.5))
        controls.slider.label.set_horizontalalignment("left")
        # Connect to key press events
        figure.canvas.mpl_connect("key_press_event", self.key_press_event)
        figure.canvas.mpl_connect("motion_notify_event", self.move_axes)
        figure.canvas.mpl_connect("button_press_event", self.start_measurement)
        figure.canvas.mpl_connect("button_release_event", self.end_measurement)

        # Set control's callbacks
        controls.button_stop.on_clicked(self.stop)
        controls.button_playpause.on_clicked(self.playpause)
        controls.button_oneforward.on_clicked(lambda *args: self.step(1))
        controls.button_oneback.on_clicked(lambda *args: self.step(-1))
        controls.slider.on_changed(self.go_to)
        controls.button_measure.on_clicked(self.toggle_measure)

        return controls

    def key_press_event(self, event):
        """Handle key press event.

        Args:
            event: Information about the key event. currently only space
                and the four arrow keys are handled
        """
        if event.key == " ":
            self.playpause()
        elif event.key == "right":
            self.oneforward()
        elif event.key == "left":
            self.onebackward()
        elif event.key == "up":
            self.step(10)
        elif event.key == "down":
            self.step(-10)
        else:
            warnings.warn(f"No event for '{event.key}' configured!")

    def set_play_pause_label(self, label, controls=None):
        """Set the label of the play/pause button.

        Args:
            label: Label text to set.
            controls: (Optional) Handle to single controls class.
        """
        label_text = label
        if label == "run":
            label_text = "\U000025B6"
        elif label == "pause":
            label_text = "\U0000275A\U0000275A"

        if controls:
            # Set label only for button in given controls class
            controls.button_playpause.label.set_text(label_text)
        else:
            # Set label for button in all controls classes
            for plot in self.plots:
                plot.controls.button_playpause.label.set_text(label_text)

    def update_plot(self, controls, update_func, *args):
        """Augmentation of update function
        of plot class that additionally
        updates the slider position.

        Args:
            controls: Handle to control class of figure.
            update_func: Handle to update function of plot class.
            *args: Arguments to update_func.
        :return Pass through return of update function of plot class (for blit).
        """
        if controls:
            controls.slider.set_val(self.i)
        return update_func(*args)

    def update_plots(self):
        """Call update function of all plots
        with current index and update slider.
        """
        for plot in self.plots:
            plot.update_func(self.i)
            plot.figure.canvas.draw_idle()
