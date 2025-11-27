from abc import ABC, abstractmethod
from typing import Union

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.solutions.my_solution.drone_pose import DronePose


class AbstractController(ABC):
    @abstractmethod
    def __init__(self):
        self.setpoint: Union[DronePose, None] = None

    @abstractmethod
    def get_command(self, current_pose: DronePose) -> CommandsDict: ...
