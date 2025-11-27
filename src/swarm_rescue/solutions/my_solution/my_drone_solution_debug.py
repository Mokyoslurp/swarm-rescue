import arcade
import numpy as np
from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01


from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.utils.path import Path
from swarm_rescue.simulation.utils.utils import deg2rad

from swarm_rescue.solutions.my_solution.drone_pose import DronePose
from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate
from swarm_rescue.solutions.my_solution.estimators.kalman_filter_1 import EKF1
from swarm_rescue.solutions.my_solution.state_machine.state_machine import StateMachine


class MyDroneSolution(DroneTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_machine = StateMachine()
        self.kalman_filter = EKF1(
            R=np.square(np.diag([0.5, 0.5, deg2rad(4.0)])),
        )
        self.previous_command = np.array([0.0, 0.0, 0.0])

        self.iteration = 0
        self.path_done: Path = Path()

    def define_message_for_all(self): ...

    def control(self) -> CommandsDict:
        gps = self.measured_gps_position()
        if gps is None:
            gps = [None, None]
        measurements = np.array(
            [
                gps[0],
                gps[1],
                self.measured_compass_angle(),
            ]
        )
        self.kalman_filter.step(self.previous_command, measurement=measurements)

        self.pose = DronePose(
            np.array([self.kalman_filter.x, self.kalman_filter.y]),
            self.kalman_filter.theta,
        )

        self.occupancy_grid.update_grid(pose=self.pose)

        command = self.state_machine.get_command(self)
        self.previous_command = np.array(
            [command["forward"], command["lateral"], command["rotation"]]
        )

        self.iteration += 1

        if self.iteration % 3 == 0:
            position = np.array([self.true_position()[0], self.true_position()[1]])
            angle = self.true_angle()
            pose = DronePose(position=position, orientation=angle)
            self.path_done.append(pose)

        if self.iteration % 5 == 0:
            self.occupancy_grid.display(
                self.occupancy_grid.grid, self.pose, title="occupancy grid"
            )
            self.occupancy_grid.display(
                self.occupancy_grid.zoomed_grid,
                self.pose,
                title="zoomed occupancy grid",
            )

        return command

    def draw_bottom_layer(self):
        try:
            self.draw_goal_path()
            self.draw_point(self.path_planner.current_pose_rounded, arcade.color.GREEN)
            self.draw_point(self.path_planner.goal_pose, arcade.color.BLUE)
            self.draw_point(self.path_planner.path_end, arcade.color.RED)
            self.draw_path(path=self.path_done, color=(255, 0, 255))
            self.draw_all_frontiers()

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
        for frontier in self.path_planner.frontiers:
            barycenter = frontier.barycenter()
            b_x, b_y = self.occupancy_grid._conv_grid_to_world(
                barycenter.x, barycenter.y
            )
            x = b_x + self._half_size_array[0]
            y = b_y + self._half_size_array[1]
            arcade.draw_point(
                x,
                y,
                color=arcade.color.BLUE,
                size=10,
            )

    def draw_goal_path(self):
        pose_start = self.path_planner.goal_pose
        for pose_end in self.path_planner.current_path:
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

    def draw_path(self, path: Path, color: tuple[int, int, int]):
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


def main():
    the_map = MapIntermediate01(
        drone_type=MyDroneSolution,  # zones_config=(ZoneType.NO_GPS_ZONE,)
    )

    gui = GuiSR(
        the_map=the_map,
        use_keyboard=False,
    )
    gui.run()


if __name__ == "__main__":
    main()
