from abc import ABC, abstractmethod
from typing import Union

from swarm_rescue.simulation.drone.controller import CommandsDict


class AbstractController(ABC):
    @abstractmethod
    def __init__(self):
        self.setpoint: Union[tuple[float, float, float], None] = None

    @abstractmethod
    def get_command(self, current_pose: tuple[float, float, float]) -> CommandsDict: ...
