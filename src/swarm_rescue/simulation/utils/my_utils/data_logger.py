"""
Logger decorator support for on_update function of GuiSR
"""

from typing import Dict

import pandas as pd
from swarm_rescue.simulation.drone.controller import CommandsDict

from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract

COLUMNS = [
    "time",
    "true_x",
    "true_y",
    "true_theta",
    "true_vx",
    "true_vy",
    "true_omega",
    "measured_x",
    "measured_y",
    "measured_theta",
    "measured_vx",
    "measured_vy",
    "measured_omega",
    "command_forward",
    "command_lateral",
    "command_rotation",
]


class DataLogger:
    """
    Provides methods to save simulation parameters and make a data file with it
    """

    def __init__(self) -> None:
        self.data = pd.DataFrame([], columns=COLUMNS)

    def log_data(
        self,
        elapsed_time: float,
        playground_agents: list[DroneAbstract],
        drones_commands: Dict[DroneAbstract, CommandsDict],
    ) -> None:
        """Logs all first drone data into a new line in a pandas dataframe

        Args:
            elapsed_time (float): time elapsed from the bigninning of the simulation
            playground_agents (list[DroneAbstract]): list of all agents in the simulation
            drones_commands (Dict[DroneAbstract, CommandsDict]): dictionnary of commands
        """
        drone = playground_agents[0]

        new_data = {}

        new_data["time"] = elapsed_time

        position = drone.true_position()
        velocity = drone.true_velocity()
        new_data["true_x"] = position[0]
        new_data["true_y"] = position[1]
        new_data["true_theta"] = drone.true_angle()
        new_data["true_vx"] = velocity[0]
        new_data["true_vy"] = velocity[1]
        new_data["true_omega"] = drone.true_angular_velocity()

        position = drone.measured_gps_position()
        velocity = drone.measured_velocity()
        new_data["measured_x"] = position[0]
        new_data["measured_y"] = position[1]
        new_data["measured_theta"] = drone.measured_compass_angle()
        new_data["measured_vx"] = velocity[0]
        new_data["measured_vy"] = velocity[1]
        new_data["measured_omega"] = drone.measured_angular_velocity()

        commands = drones_commands.get(drone, {})
        if "forward" in commands.keys():
            new_data["command_forward"] = commands["forward"]
        else:
            new_data["command_lateral"] = 0
        if "lateral" in commands.keys():
            new_data["command_lateral"] = commands["lateral"]
        else:
            new_data["command_lateral"] = 0
        if "rotation" in commands.keys():
            new_data["command_rotation"] = commands["rotation"]
        else:
            new_data["command_rotation"] = 0

        self.data.loc[len(self.data)] = new_data

    def save_data(self, path: str = "example.csv"):
        """Saves the pandas DataFrame into a csv file

        Args:
            path (str, optional): csv file path. Defaults to "example.csv".
        """
        self.data[self.data.isna()] = 0
        self.data.to_csv(path)

    def load_data(self, path: str = "example.csv"):
        """Loads a DataFrame from a csv file

        Args:
            path (str, optional): csv file path. Defaults to "example.csv".
        """

        self.data = pd.read_csv(path)
        return self.data

    def on_update_log(self, gui: GuiSR):
        """Defines a wrapper for the GuiSR on_update method, to log the data at each time step

        Args:
            gui (GuiSR): the instance of the gui
        """

        def wrapper(delta_time: float):
            result = gui.on_update(delta_time)

            self.log_data(
                gui._elapsed_walltime,  # pylint: disable=protected-access
                gui._playground.agents,  # pylint: disable=protected-access
                gui._drones_commands,  # pylint: disable=protected-access
            )
            return result

        return wrapper
