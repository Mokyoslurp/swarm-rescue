from typing import Optional
import numpy as np

from swarm_rescue.solutions.my_solution.drone_pose import DronePose
from swarm_rescue.solutions.my_solution.mapping.frontier import Frontier
from swarm_rescue.solutions.my_solution.mapping.occupancy_grid import OccupancyGrid
from swarm_rescue.solutions.my_solution.planning.graph import Vertex
from swarm_rescue.solutions.my_solution.planning.a_star import a_star


class Planner:
    """Defines a basic planner that can compute a path in an occupancy grid to explore it"""

    def __init__(self, distance_threshold: float = 10) -> None:
        self.goal_pose: DronePose = None
        self.current_path: list[DronePose] = []

        self.path_end: DronePose = None

        self.distance_threshold = distance_threshold

        self.frontiers: list[Frontier] = []

        self.counter = 0
        self.counter_limit = 50

        self.current_pose_rounded: DronePose = DronePose(np.array([0, 0]), 0)

    def get_next_pose(
        self,
        current_pose: DronePose,
        grid: OccupancyGrid,
        goal: Optional[DronePose] = None,
    ) -> DronePose:
        """Gets the next pose the drone should go to

        Args:
            current_pose (DronePose): current pose of the drone
            grid (OccupancyGrid): the current occupancy grid

        Returns:
            DronePose: the pose the drone should aim to
        """

        x_g, y_g = grid._conv_world_to_grid(current_pose.x, current_pose.y)  # pylint: disable=protected-access
        x_w, y_w = grid._conv_grid_to_world(x_g, y_g)  # pylint: disable=protected-access
        self.current_pose_rounded = Vertex(x_w, y_w)

        recomputed_path = False

        if len(self.current_path) == 0 or (goal is not None and goal != self.path_end):
            self.compute_new_path(current_pose, grid, goal)

        if (
            self.path_end is not None
            and current_pose.distance_squared(self.path_end)
            <= self.distance_threshold**2
        ):
            self.compute_new_path(current_pose, grid, goal)

        if self.counter == self.counter_limit:
            self.counter = 0

            self.compute_new_path(current_pose, grid, self.path_end)
            recomputed_path = True

        if len(self.current_path) != 0:
            if (
                recomputed_path
                or self.goal_pose is None
                or current_pose.distance_squared(self.goal_pose)
                <= self.distance_threshold**2
            ):
                self.goal_pose = self.current_path.pop(0)

        if self.goal_pose is not None:
            self.counter += 1
            return self.goal_pose
        return current_pose

    def compute_new_path(
        self,
        current_pose: DronePose,
        grid: OccupancyGrid,
        goal: Optional[DronePose] = None,
    ) -> list[DronePose]:
        """Computes a path to a chosen frontier

        Args:
            current_pose (DronePose): current pose of the drone
            grid (OccupancyGrid): current occupancy grid

        Returns:
            list[DronePose]: a path to the frontier in world coodinates
        """
        path = None

        x, y = grid._conv_world_to_grid(current_pose.x, current_pose.y)  # pylint: disable=protected-access

        # 1. Determine the frontier to explore

        if goal is None:
            frontiers = grid.get_frontiers()
            self.frontiers = frontiers

            # Closest frontier
            min_i = 0

            max_score = 0
            for i, frontier in enumerate(frontiers):
                barycenter = frontier.barycenter()
                distance = (barycenter.x - x) ** 2 + (barycenter.y - y) ** 2
                size = len(frontier.points)
                score = size - distance / 100
                if score > max_score:
                    min_i = i
                    max_score = score

            next_frontier = frontiers[min_i]
            end = next_frontier.barycenter()
        else:
            x_goal, y_goal = grid._conv_world_to_grid(goal.x, goal.y)  # pylint: disable=protected-access
            end = Vertex(x_goal, y_goal)

        # 2. Define planning problem

        start = Vertex(x, y)

        # If the barycenter is occupied, search for closest empty point
        if grid.is_occupied(end.x, end.y):
            end = grid.get_nearest_freepoint(end)

            if end is None:
                return []

        # 3. Compute the actual path in a graph

        adt = grid.to_graph(start, end)
        _, path = a_star(adt, start, end)

        # 4. Manage no solution cases

        if path is None or len(path) == 0:
            return []

        else:
            # Conversion to poses in world coordinates
            pose_path = []
            for point in path:
                i, j = grid._conv_grid_to_world(point.x, point.y)  # pylint: disable=protected-access

                pose_path.append(DronePose(position=np.array([i, j])))
            self.current_path = pose_path
            self.path_end = self.current_path[-1]
            return pose_path
