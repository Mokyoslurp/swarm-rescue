"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the
keyboard
"""

import pathlib
import sys
from typing import List, Type
import matplotlib.pyplot as plt
import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.elements.return_area import ReturnArea
from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.my_utils.data_logger import DataLogger
from swarm_rescue.simulation.utils.my_utils.data_plotter import DataPlotter

from swarm_rescue.solutions.my_solution.estimators.kalman_filter_1 import EKF1

PATH = "./src/swarm_rescue/solutions/my_solution/estimators/data/"


# Insert the 'src' directory, located two levels up from the current script,
# into sys.path. This ensures Python can find project-specific modules
# (e.g., 'swarm_rescue') when the script is run from a subfolder like 'examples/'.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


class MyDroneEstimator(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.last_inside = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self) -> CommandsDict:
        command: CommandsDict = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0,
        }

        if self.is_inside_return_area != self.last_inside:
            print("is_inside_return_area : ", self.is_inside_return_area)
            self.last_inside = self.is_inside_return_area

        return command


class MyMapKeyboard(MapAbstract):
    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (600, 600)

        self._rescue_center = RescueCenter(size=(100, 100))
        self._rescue_center_pos = ((0, 100), 0)

        self._return_area = ReturnArea(size=(150, 100))
        self._return_area_pos = ((0, -20), 0)

        self._wounded_persons_pos = [(200, 0), (-200, 0), (200, -200), (-200, -200)]

        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        self._number_drones = 1
        self._drones_pos = [((0, 0), 0)]
        self._drones = []

        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._rescue_center, self._rescue_center_pos)

        self._playground.add(self._return_area, self._return_area_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            self._playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(
            size_area=self._size_area,
            number_drones=self._number_drones,
            max_timestep_limit=self._max_timestep_limit,
            max_walltime_limit=self._max_walltime_limit,
        )
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            self._playground.add(drone, self._drones_pos[i])


def print_keyboard_man():
    print("How to use the keyboard to direct the drone?")
    print("\t- up / down key : forward and backward")
    print("\t- left / right key : turn left / right")
    print("\t- shift + left/right key : left/right lateral movement")
    print("\t- W key : grasp wounded person")
    print("\t- L key : display (or not) the lidar sensor")
    print("\t- S key : display (or not) the semantic sensor")
    print("\t- P key : draw position from GPS sensor")
    print("\t- C key : draw communication between drones")
    print("\t- M key : print messages between drones")
    print("\t- Q key : exit the program")
    print("\t- R key : reset")


def main():
    logger = DataLogger()

    # print_keyboard_man()
    the_map = MyMapKeyboard(drone_type=MyDroneEstimator)

    gui = GuiSR(
        the_map=the_map,
        draw_lidar_rays=True,
        draw_semantic_rays=True,
        use_keyboard=True,
    )

    gui._playground.window.on_update = logger.on_update_log(gui)

    gui.run()

    score_health_returned = the_map.compute_score_health_returned()
    print("score_health_returned = ", score_health_returned)

    logger.save_data(PATH + "example.csv")

    ekf = EKF1()

    estimated_x = []
    estimated_y = []
    estimated_theta = []
    for _, data in logger.data.iterrows():
        ekf.step(
            np.array(data[["command_forward", "command_lateral", "command_rotation"]]),
            np.array(data[["measured_x", "measured_y", "measured_theta"]]),
        )

        estimated_x.append(ekf.x)
        estimated_y.append(ekf.y)
        estimated_theta.append(ekf.theta)

    plotter = DataPlotter(logger.load_data(PATH + "example.csv"))

    plotter.plot_position(estimated_x, estimated_y)
    plotter.plot_angle(estimated_theta)
    plt.show()


if __name__ == "__main__":
    main()
