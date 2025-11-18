import cv2
import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_lidar import DroneLidar
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.pose import Pose


EVERY_N = 3
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2.0
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

        alpha = 0.8
        points_x = pose.position[0] + np.multiply(alpha * lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(alpha * lidar_dist, sin_rays)

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
