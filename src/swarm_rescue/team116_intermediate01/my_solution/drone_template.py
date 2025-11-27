from abc import ABC


from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.my_solution.drone_pose import DronePose
from swarm_rescue.solutions.my_solution.mapping.occupancy_grid import OccupancyGrid
from swarm_rescue.solutions.my_solution.planning.planner import Planner


class DroneTemplate(DroneAbstract, ABC):
    def __init__(
        self,
        identifier: int | None = None,
        misc_data: MiscData | None = None,
        display_lidar_graph: bool = False,
        **kwargs,
    ):
        super().__init__(identifier, misc_data, display_lidar_graph, **kwargs)

        resolution = 18
        self.occupancy_grid = OccupancyGrid(
            size_area_world=self.size_area, resolution=resolution, lidar=self.lidar()
        )

        self.path_planner = Planner(distance_threshold=18)

        self.pose: DronePose
