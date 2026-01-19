from typing import Self

from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.solutions.my_solution.planning.graph import Vertex


class DronePose:
    """Represents a 2D pose with position and orientation"""

    def __init__(self, x: float, y: float, yaw: float = 0.0):
        """Creates a new pose

        Args:
            x (float): x position
            y (float): y position
            yaw (float, optional): yaw value. Defaults to 0.0.
        """
        self.x = x
        self.y = y
        self.yaw = yaw

    def distance_squared(self, other: Self) -> float:
        """Calculates the squared distance to an other pose using x and y

        Args:
            other (Self): the other pose

        Returns:
            float: the squared distance
        """
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    @classmethod
    def from_vertex(cls, vertex: Vertex) -> Self:
        """Generates a drone pose from a map vertex

        Args:
            vertex (Vertex): the vertex

        Returns:
            Self: a new pose
        """
        return DronePose(x=vertex.x, y=vertex.y)

    @classmethod
    def from_vertices(cls, vertices: list[Vertex]) -> list[Pose]:
        """Generates a list of drone poses from a list of map vertices

        Args:
            vertices (list[Vertex]): the vertices

        Returns:
            list[Pose]: the new poses
        """
        poses = []
        for vertex in vertices:
            poses.append(DronePose.from_vertex(vertex))

        return poses

    def __add__(self, other: object) -> Self:
        """Adds to poses x, y and yaw values

        Args:
            other (object): the other pose

        Returns:
            Self: the addition of the two poses
        """
        if isinstance(other, DronePose):
            return DronePose(self.x + other.x, self.y + other.y, self.yaw + other.yaw)
        return self

    def __eq__(self, other: object) -> bool:
        """Verifies if two poses are the same in x, y and yaw

        Args:
            other (object): the other pose

        Returns:
            bool: True if the poses have the same x, y and yaw, False either
        """
        if isinstance(other, DronePose):
            return self.x == other.x and self.y == other.y and self.yaw == other.yaw
        return False

    def __str__(self) -> str:
        """Generates a string representing the pose by its x, y and yaw values

        Returns:
            str: the string
        """
        return f"X = {self.x} ; Y = {self.y} ; Yaw = {self.yaw}"


class DroneState(DronePose):
    """Respresents a drone state by its pose and angular and linear speeds"""

    def __init__(
        self, x: float, y: float, yaw: float, vx: float, vy: float, yaw_rate: float
    ):
        """Generates the drone state

        Args:
            x (float): x position
            y (float): y position
            yaw (float): yaw value
            vx (float): x velocity
            vy (float): y velocity
            yaw_rate (float): yaw rate value
        """
        super().__init__(x, y, yaw)

        self.vx = vx
        self.vy = vy
        self.yaw_rate = yaw_rate
