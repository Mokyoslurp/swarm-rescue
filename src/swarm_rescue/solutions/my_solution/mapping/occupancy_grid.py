import math
import cv2
import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_lidar import DroneLidar
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.solutions.my_solution.mapping.frontier import Frontier
from swarm_rescue.solutions.my_solution.mapping.vertex import Vertex
from swarm_rescue.solutions.my_solution.planning.graph import GraphADT


EVERY_N = 3
LIDAR_DIST_CLIP = 20.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 5.0
FREE_ZONE_VALUE = -4.0
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40


class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self, size_area_world, resolution: float, lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar: DroneLidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """Bayesian map update with new observation

        Args:
            pose (Pose): corrected pose in world coordinates
        """

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(
                pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE
            )

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (
            int(self.size_area_world[1] * 0.5),
            int(self.size_area_world[0] * 0.5),
        )
        self.zoomed_grid = cv2.resize(  # pylint: disable=no-member
            self.zoomed_grid,
            new_zoomed_size,
            interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
        )

    def get_frontiers(self) -> list[Frontier]:
        """Computes a list of frontiers in the grid

        Returns:
            list[Frontier]: Tthe frontiers
        """

        points: set[Vertex] = set([])
        frontiers: list[Frontier] = []

        n_row = len(self.grid)
        n_col = len(self.grid[0, :])
        for x in range(n_row):
            for y in range(n_col):
                if self.is_empty(x, y):
                    delta = [-1, 0, 1]
                    for i in delta:
                        for j in delta:
                            try:
                                if self.is_free(x + i, y + j):
                                    points.add(Vertex(x + i, y + j))
                            except IndexError:
                                pass

        def get_closest_frontier(frontiers: list[Frontier], point: Vertex):
            for frontier in frontiers:
                if frontier.is_close_to_point(point):
                    return frontier
            return None

        for point in points:
            frontier = get_closest_frontier(frontiers, point)
            if frontier is not None:
                frontier.add_point(point)
            else:
                frontiers.append(Frontier([point]))

        max_size = 0
        for frontier in frontiers:
            size = len(frontier.points)
            if size > max_size:
                max_size = size

        frontiers_cleaned = frontiers.copy()
        for frontier in frontiers:
            if len(frontier.points) < max_size - 5:
                frontiers_cleaned.remove(frontier)

        return frontiers_cleaned

    def get_obstacles(self) -> list[tuple[int, int, float]]:
        """Gets the obstacles of the grid

        Returns:
            list[tuple[int, int, float]]: a list of tuples contianing x and y coordinates of
                the obstacles and the associated value in the grid
        """
        matrix = self.grid

        obstacle_list = []
        for row in range(len(matrix)):
            for col in range(len(matrix[row, :])):
                if self.is_occupied(row, col):
                    # y_obs, x_obs = col, row
                    y_obs, x_obs = row, col
                    obstacle_list.append((x_obs, y_obs, 1.0))  # [x, y, obstacle_radius]
        return obstacle_list

    def to_graph(self, start: Vertex, end: Vertex) -> GraphADT:
        """Creates a Graph from the grid

        Args:
            start (Vertex): the start position in the grid
            end (Vertex): the end position in the grid

        Returns:
            GraphADT: the graph created
        """

        map_graph = GraphADT()
        map_graph = self.add_edges_around(start.x, start.y, map_graph)

        map_graph = self.add_edges_around(end.x, end.y, map_graph)

        n_row = len(self.grid)
        n_col = len(self.grid[0, :])
        for row in range(n_row):
            for col in range(n_col):
                if self.is_free(row, col):
                    map_graph = self.add_edges_around(row, col, map_graph)
                else:
                    pass

        return map_graph

    def is_empty(self, x: int, y: int) -> bool:
        """Checks if the value at the coordinates is not yet explored in the grid

        Args:
            x (int): x coordinate of the point
            y (int): y coordinate of the point

        Returns:
            bool: True if the value is 0, False either
        """
        return self.grid[x, y] == 0

    def is_occupied(self, x: int, y: int) -> bool:
        """Checks if the value at the coordinates is occupied in the grid

        Args:
            x (int): x coordinate of the point
            y (int): y coordinate of the point

        Returns:
            bool: True if the value is positive, False either
        """
        return self.grid[x, y] > 0

    def is_free(self, x: int, y: int) -> bool:
        """Checks if the value at the coordinates is free in the grid

        Args:
            x (int): x coordinate of the point
            y (int): y coordinate of the point

        Returns:
            bool: True if the value is negative, False either
        """
        return self.grid[x, y] < 0

    def add_edges_around(self, x: int, y: int, map_graph: GraphADT) -> GraphADT:
        """Adds all possible edges to the graph, departing from coordinates in the grid

        Args:
            x (int): x coordinate of the departure point
            y (int): y coordinate of the departure point
            map_graph (GraphADT): the graph to add the points to

        Returns:
            GraphADT: the graph with the points added
        """
        # Checks all points around the specified coordinates
        delta = [-1, 0, 1]

        for i in delta:
            for j in delta:
                try:
                    if self.is_free(x + i, y + j):
                        # Avoid putting an edge with 0 value in the graph
                        if i != 0 or j != 0:
                            # Connect (x, y) and (x+i, y+j) in the graph
                            map_graph.append(
                                Vertex(x, y),
                                Vertex(x + i, y + j),
                                1,
                            )

                # Accounts for boundary indices
                except IndexError:
                    pass

        return map_graph

    def get_nearest_freepoint(self, point: Vertex, delta: int = 3) -> Vertex:
        """Returns a point in the grid that is free and the closest possible to the specified point

        Args:
            point (Vertex): the original point, normally not free
            delta (int, optional): the delta of coordinates to check. Defaults to 3.

        Returns:
            Vertex: the nearest free point
        """
        freepoints: list[Vertex] = []
        distances = np.array([])

        indices = np.linspace(-delta, delta, 2 * delta + 1).astype(int)
        for dr in indices:
            for dc in indices:
                try:
                    if self.is_free(point.x + dr, point.y + dc):
                        if dr != 0 or dc != 0:
                            freepoints.append(Vertex(point.x + dc, point.y + dr))
                            distances = np.append(
                                distances, [math.hypot(abs(dr), abs(dc))]
                            )
                except IndexError:
                    pass

        if freepoints:
            index = np.argmin(distances)
            new_end = freepoints[index]
            return new_end
        return None
