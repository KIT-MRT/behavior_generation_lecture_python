from statemachine import StateMachine, State
import time
import warnings

from finite_state_machine.traffic_light_visualization import VisualizeTrafficLights


class TrafficLightStateMachine(StateMachine):
    """A traffic light machine"""

    veh_green = State("veh_green", initial=True)
    veh_yellow = State("veh_yellow")
    veh_red_ped_red_1 = State("veh_red_ped_red_1")
    veh_red_ped_red_2 = State("veh_red_ped_red_2")
    veh_red_ped_green = State("veh_red_ped_green")
    veh_red_yellow = State("veh_red_yellow")

    veh_prepare_to_stop = veh_green.to(veh_yellow)
    veh_stop = veh_yellow.to(veh_red_ped_red_1)
    ped_go = veh_red_ped_red_1.to(veh_red_ped_green)
    ped_stop = veh_red_ped_green.to(veh_red_ped_red_2)
    veh_prepare_to_go = veh_red_ped_red_2.to(veh_red_yellow)
    veh_go = veh_red_yellow.to(veh_green)

    def __init__(self, start_mainloop=True):
        StateMachine.__init__(self)
        self.visu = VisualizeTrafficLights()
        self.visu.register_button_event(self.button_press)
        self.visu.vehicle_go()
        if start_mainloop:
            self.visu.mainloop()

    def on_enter_veh_yellow(self):
        self.visu.pedestrian_press_red()
        time.sleep(3)
        self.visu.vehicle_prepare_to_stop()
        time.sleep(3)
        self.run("veh_stop")

    def on_enter_veh_red_ped_red_1(self):
        self.visu.vehicle_stop()
        time.sleep(2)
        self.run("ped_go")

    def on_enter_veh_red_ped_green(self):
        self.visu.pedestrian_go()
        time.sleep(5)
        self.run("ped_stop")

    def on_enter_veh_red_ped_red_2(self):
        self.visu.pedestrian_stop()
        time.sleep(5)
        self.run("veh_prepare_to_go")

    def on_enter_veh_red_yellow(self):
        self.visu.vehicle_prepare_to_go()
        time.sleep(2)
        self.run("veh_go")

    def on_enter_veh_green(self):
        self.visu.vehicle_go()

    def button_press(self, event):
        if self.is_veh_green:
            self.run("veh_prepare_to_stop")
        else:
            warnings.warn(
                "The button only has effect if currently vehicles have green."
            )
