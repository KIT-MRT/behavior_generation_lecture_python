import tkinter as tk


class VisualizeTrafficLights:
    def __init__(self):
        self.windows = tk.Tk()
        self.windows.title("Traffic Light FSM")
        self.canvas = tk.Canvas(self.windows)

        self.vehicle_red = self.__create_circle(55, 55, 50)
        self.vehicle_yellow = self.__create_circle(55, 165, 50)
        self.vehicle_green = self.__create_circle(55, 275, 50)
        self.pedestrian_red = self.__create_circle(180, 55, 50)
        self.pedestrian_green = self.__create_circle(180, 165, 50)
        self.pedestrian_press_button = self.canvas.create_rectangle(140, 260, 220, 290)
        self.pedestrian_press_label = self.canvas.create_text(180, 275, text="Press")
        # self.pedestrian_press = tk.Button(text="Press", command=pedestrian_press_fun)
        # self.canvas.create_window(180, 265, window=self.pedestrian_press)

        self.__pedestrian_press_white()

        self.canvas.config(width=400, height=400)
        self.canvas.pack()

        self.__update()

    def __create_circle(self, x, y, r, fill=None):
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return self.canvas.create_oval(x0, y0, x1, y1, fill=fill)

    def __update(self):
        self.canvas.update_idletasks()
        self.canvas.update()
        self.windows.update_idletasks()
        self.windows.update()

    def vehicle_go(self):
        self.canvas.itemconfig(self.vehicle_red, fill="black")
        self.canvas.itemconfig(self.vehicle_yellow, fill="black")
        self.canvas.itemconfig(self.vehicle_green, fill="green")
        self.canvas.itemconfig(self.pedestrian_red, fill="red")
        self.canvas.itemconfig(self.pedestrian_green, fill="black")

        self.__update()

    def vehicle_prepare_to_stop(self):
        self.canvas.itemconfig(self.vehicle_red, fill="black")
        self.canvas.itemconfig(self.vehicle_yellow, fill="yellow")
        self.canvas.itemconfig(self.vehicle_green, fill="black")
        self.canvas.itemconfig(self.pedestrian_red, fill="red")
        self.canvas.itemconfig(self.pedestrian_green, fill="black")

        self.__update()

    def vehicle_stop(self):
        self.canvas.itemconfig(self.vehicle_red, fill="red")
        self.canvas.itemconfig(self.vehicle_yellow, fill="black")
        self.canvas.itemconfig(self.vehicle_green, fill="black")
        self.canvas.itemconfig(self.pedestrian_red, fill="red")
        self.canvas.itemconfig(self.pedestrian_green, fill="black")

        self.__update()

    def pedestrian_go(self):
        self.__pedestrian_press_white()

        self.canvas.itemconfig(self.vehicle_red, fill="red")
        self.canvas.itemconfig(self.vehicle_yellow, fill="black")
        self.canvas.itemconfig(self.vehicle_green, fill="black")
        self.canvas.itemconfig(self.pedestrian_red, fill="black")
        self.canvas.itemconfig(self.pedestrian_green, fill="green")

        self.__update()

    def pedestrian_stop(self):
        self.canvas.itemconfig(self.vehicle_red, fill="red")
        self.canvas.itemconfig(self.vehicle_yellow, fill="black")
        self.canvas.itemconfig(self.vehicle_green, fill="black")
        self.canvas.itemconfig(self.pedestrian_red, fill="red")
        self.canvas.itemconfig(self.pedestrian_green, fill="black")

        self.__update()

    def vehicle_prepare_to_go(self):
        self.canvas.itemconfig(self.vehicle_red, fill="red")
        self.canvas.itemconfig(self.vehicle_yellow, fill="yellow")
        self.canvas.itemconfig(self.vehicle_green, fill="black")
        self.canvas.itemconfig(self.pedestrian_red, fill="red")
        self.canvas.itemconfig(self.pedestrian_green, fill="black")

        self.__update()

    def pedestrian_press_red(self):
        self.canvas.itemconfig(self.pedestrian_press_button, fill="red")

        self.__update()

    def __pedestrian_press_white(self):
        self.canvas.itemconfig(self.pedestrian_press_button, fill="white")

        self.__update()

    def mainloop(self):
        self.windows.mainloop()

    def register_button_event(self, button_press_function):
        self.canvas.tag_bind(
            self.pedestrian_press_button, "<Button-1>", button_press_function
        )
        self.canvas.tag_bind(
            self.pedestrian_press_label, "<Button-1>", button_press_function
        )
