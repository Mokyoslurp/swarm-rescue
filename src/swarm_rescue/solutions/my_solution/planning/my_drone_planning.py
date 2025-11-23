from typing import Tuple, Type

import arcade
import numpy as np


from swarm_rescue.maps.walls_medium_02 import add_boxes, add_walls
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.path import Path
from swarm_rescue.simulation.utils.pose import Pose

from swarm_rescue.solutions.my_solution.drone_pose import DronePose
from swarm_rescue.solutions.my_solution.mapping.occupancy_grid import OccupancyGrid

from swarm_rescue.solutions.my_solution.planning.planner import Planner


class MyDroneMapping(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.iteration: int = 0

        resolution = 12
        self.grid = OccupancyGrid(
            size_area_world=self.size_area, resolution=resolution, lidar=self.lidar()
        )

        self.planner = Planner(distance_threshold=20)

        self.path_done = Path()

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """

    def control(self) -> CommandsDict:
        """
        We only send a command to do nothing
        """

        command: CommandsDict = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0,
        }

        # estimated_pose = Pose(
        #     np.asarray(self.measured_gps_position()), self.measured_compass_angle()
        # )
        estimated_pose = DronePose(np.asarray(self.true_position()), self.true_angle())

        self.grid.update_grid(pose=estimated_pose)

        # Display
        self.iteration += 1

        if self.iteration % 3 == 0:
            position = np.array([self.true_position()[0], self.true_position()[1]])
            angle = self.true_angle()
            pose = Pose(position=position, orientation=angle)
            self.path_done.append(pose)

        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, estimated_pose, title="occupancy grid")
            self.grid.display(
                self.grid.zoomed_grid,
                estimated_pose,
                title="zoomed occupancy grid",
            )

        next_pose = self.planner.get_next_pose(estimated_pose, self.grid)

        if next_pose is not None:
            # P controller
            forward = 0.2 * np.clip(next_pose.x - estimated_pose.x, -1.0, 1.0)
            lateral = 0.2 * np.clip(next_pose.y - estimated_pose.y, -1.0, 1.0)
            rotation = 0.1 * np.clip(-self.true_angle(), -1, 1)

            command: CommandsDict = {
                "forward": forward,
                "lateral": lateral,
                "rotation": rotation,
                "grasper": 0,
            }

        return command

    def draw_bottom_layer(self):
        try:
            self.draw_point(self.planner.current_pose_rounded, arcade.color.GREEN)
            self.draw_point(self.planner.goal_pose, arcade.color.BLUE)
            self.draw_point(self.planner.path_end, arcade.color.RED)
            self.draw_path(path=self.path_done, color=(255, 0, 255))
            self.draw_goal_path()
        except Exception:
            pass

    def draw_point(self, pose: DronePose, color: arcade.Color = arcade.color.BLACK):
        x = pose.x + self._half_size_array[0]
        y = pose.y + self._half_size_array[1]
        arcade.draw_point(
            x,
            y,
            color=color,
            size=10,
        )

    def draw_all_frontiers(self):
        for frontier in self.planner.frontiers:
            barycenter = frontier.barycenter()
            x = barycenter.x + self._half_size_array[0]
            y = barycenter.y + self._half_size_array[1]
            arcade.draw_point(
                x,
                y,
                color=arcade.color.BLUE,
                size=10,
            )

    def draw_goal_path(self):
        pose_start = self.planner.goal_pose
        for pose_end in self.planner.current_path:
            x_start = pose_start.x + self._half_size_array[0]
            y_start = pose_start.y + self._half_size_array[1]
            x_end = pose_end.x + self._half_size_array[0]
            y_end = pose_end.y + self._half_size_array[1]
            arcade.draw_line(
                x_start,
                y_start,
                x_end,
                y_end,
                color=arcade.color.GREEN,
                line_width=3,
            )
            pose_start = pose_end

    def draw_path(self, path: Path, color: Tuple[int, int, int]):
        length = path.length()
        pt2 = None
        for ind_pt in range(length):
            pose = path.get(ind_pt)
            pt1 = pose.position + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(
                    float(pt2[0]), float(pt2[1]), float(pt1[0]), float(pt1[1]), color
                )
            pt2 = pt1


class MyMapMapping(MapAbstract):
    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (1113, 750)

        self._rescue_center = RescueCenter(size=(210, 90))
        self._rescue_center_pos = ((440, 315), 0)

        self._number_drones = 1
        self._drones_pos = [((-50, 0), 0)]
        self._drones = []

        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(self._playground)
        add_boxes(self._playground)

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


def main():
    the_map = MyMapMapping(drone_type=MyDroneMapping)

    gui = GuiSR(
        the_map=the_map,
        use_keyboard=False,
    )
    gui.run()


if __name__ == "__main__":
    main()
