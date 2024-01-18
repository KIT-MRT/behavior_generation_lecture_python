from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def make_plot_grid_step_function(columns, rows, U_over_time, show=True):
    """ipywidgets interactive function supports single parameter as input.

    This function creates and return such a function by taking as input
    other parameters.

    from https://github.com/aimacode/aima-python

    The MIT License (MIT)

    Copyright (c) 2016 aima-python contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def plot_grid_step(iteration):
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse()  # output like book
        grid = [[-200 if y is None else y for y in x] for x in grid]
        fig = plt.imshow(grid, cmap=plt.cm.bwr, interpolation="nearest")

        plt.axis("off")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        for col in range(len(grid)):
            for row in range(len(grid[0])):
                magic = grid[col][row]
                fig.axes.text(
                    row, col, "{0:.2f}".format(magic), va="center", ha="center"
                )
        if show:
            plt.show()

    return plot_grid_step


def make_plot_policy_step_function(columns, rows, policy_over_time, show=True):
    """Create a function that allows plotting a policy over time."""

    def plot_grid_step(iteration):
        data = policy_over_time[iteration]
        for row in range(rows):
            for col in range(columns):
                if not (col, row) in data:
                    continue
                x = col + 0.5
                y = row + 0.5
                if data[(col, row)] is None:
                    plt.scatter([x], [y], color="black")
                    continue
                dx = data[(col, row)][0]
                dy = data[(col, row)][1]
                scaling = np.sqrt(dx**2.0 + dy**2.0) * 2.5
                dx /= scaling
                dy /= scaling
                plt.arrow(x, y, dx, dy)
        plt.axis("equal")
        plt.xlim([0, columns])
        plt.ylim([0, rows])
        if show:
            plt.show()

    return plot_grid_step
