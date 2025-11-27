from typing import Self

import numpy as np
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.solutions.my_solution.planning.graph import Vertex


class DronePose(Pose):
    """
    Describes a drone Pose with a few extra methods
    """

    @property
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def distance_squared(self, other: Self) -> float:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    @classmethod
    def from_vertex(cls, vertex: Vertex) -> Self:
        return DronePose(position=np.array([vertex.x, vertex.y]))

    @classmethod
    def from_vertices(cls, vertices: list[Vertex]) -> list[Pose]:
        poses = []
        for vertex in vertices:
            poses.append(DronePose.from_vertex(vertex))

        return poses

    def __add__(self, other: object) -> Self:
        if isinstance(other, DronePose):
            return DronePose(
                self.position + other.position, self.orientation + other.orientation
            )
        return self

    def __str__(self) -> str:
        return f"X = {self.x} ; Y = {self.y}"
