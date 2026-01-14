import tkinter as tk

class VisualizeTrafficLights:
    def __init__(self):
        self.windows = tk.Tk()
        self.windows.title("Traffic Light FSM")
        self.windows.configure(bg="#2c3e50")  # Dark background for window

        # Canvas settings
        self.width = 400
        self.height = 400
        self.canvas = tk.Canvas(
            self.windows, 
            width=self.width, 
            height=self.height, 
            bg="#2c3e50", 
            highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=20)

        # Traffic Light Colors
        self.colors = {
            "off_red": "#4a0000", "on_red": "#ff3b30",
            "off_yellow": "#4a4a00", "on_yellow": "#ffcc00",
            "off_green": "#002a00", "on_green": "#4cd964",
            "housing": "#1a1a1a",
            "text": "#ecf0f1"
        }

        # Draw Housing (Background Boxes)
        # Vehicle Light Housing
        self._draw_housing(50, 20, 160, 360)
        # Pedestrian Light Housing
        self._draw_housing(240, 20, 350, 240)

        # Labels
        self.canvas.create_text(105, 375, text="Vehicle", fill=self.colors["text"], font=("Helvetica", 12, "bold"))
        self.canvas.create_text(295, 255, text="Pedestrian", fill=self.colors["text"], font=("Helvetica", 12, "bold"))

        # Create Lights
        self.vehicle_red = self.__create_circle(105, 80, 40, fill=self.colors["off_red"])
        self.vehicle_yellow = self.__create_circle(105, 190, 40, fill=self.colors["off_yellow"])
        self.vehicle_green = self.__create_circle(105, 300, 40, fill=self.colors["off_green"])

        self.pedestrian_red = self.__create_circle(295, 80, 40, fill=self.colors["off_red"])
        self.pedestrian_green = self.__create_circle(295, 190, 40, fill=self.colors["off_green"])

        # Pedestrian Button
        self.pedestrian_press_button = self.canvas.create_rectangle(
            240, 280, 350, 320, 
            fill="#95a5a6", outline="", width=0
        )
        self.pedestrian_press_label = self.canvas.create_text(
            295, 300, 
            text="PRESS BUTTON", 
            fill="#2c3e50", 
            font=("Helvetica", 10, "bold")
        )

        self.__update()

    def _draw_housing(self, x1, y1, x2, y2):
        r = 15 # radius for rounded corners effect
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors["housing"], outline="black", width=2)

    def __create_circle(self, x, y, r, fill=None):
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="black", width=2)

    def __update(self):
        self.canvas.update_idletasks()
        self.canvas.update()

    # Helper to set lights easily
    def _set_lights(self, v_red, v_yel, v_grn, p_red, p_grn):
        self.canvas.itemconfig(self.vehicle_red, fill=self.colors["on_red"] if v_red else self.colors["off_red"])
        self.canvas.itemconfig(self.vehicle_yellow, fill=self.colors["on_yellow"] if v_yel else self.colors["off_yellow"])
        self.canvas.itemconfig(self.vehicle_green, fill=self.colors["on_green"] if v_grn else self.colors["off_green"])
        
        self.canvas.itemconfig(self.pedestrian_red, fill=self.colors["on_red"] if p_red else self.colors["off_red"])
        self.canvas.itemconfig(self.pedestrian_green, fill=self.colors["on_green"] if p_grn else self.colors["off_green"])
        self.__update()

    def vehicle_go(self):
        self._set_lights(False, False, True, True, False)

    def vehicle_prepare_to_stop(self):
        self._set_lights(False, True, False, True, False)

    def vehicle_stop(self):
        self._set_lights(True, False, False, True, False)

    def pedestrian_go(self):
        self.__pedestrian_press_reset()
        self._set_lights(True, False, False, False, True)

    def pedestrian_stop(self):
        self._set_lights(True, False, False, True, False)

    def vehicle_prepare_to_go(self):
        self._set_lights(True, True, False, True, False)

    def pedestrian_press_red(self):
        self.canvas.itemconfig(self.pedestrian_press_button, fill="#e74c3c") # Red button
        self.canvas.itemconfig(self.pedestrian_press_label, fill="white")
        self.__update()

    def __pedestrian_press_reset(self):
        self.canvas.itemconfig(self.pedestrian_press_button, fill="#95a5a6") # Grey button
        self.canvas.itemconfig(self.pedestrian_press_label, fill="#2c3e50")
        self.__update()

    def mainloop(self):
        self.windows.mainloop()

    def register_button_event(self, button_press_function):
        self.canvas.tag_bind(self.pedestrian_press_button, "<Button-1>", button_press_function)
        self.canvas.tag_bind(self.pedestrian_press_label, "<Button-1>", button_press_function)
