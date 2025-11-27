import arcade
import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.path import Path
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.solutions.my_solution.controllers.pid_controller import PIDController

SETPOINTS = [
    [100, 100, np.pi / 2],
    [-100, 100, np.pi],
    [100, -100, 3 * np.pi / 2],
    [-100, -100, 0],
]


class MyDrone1(DroneAbstract):
    def __init__(
        self,
        identifier: int | None = None,
        misc_data: MiscData | None = None,
        display_lidar_graph: bool = False,
        **kwargs,
    ):
        super().__init__(identifier, misc_data, display_lidar_graph, **kwargs)

        self.pid = PIDController(Kp=(1.6, 9.0), Ki=(2, 2), Kd=(8.0, 0.6))
        self.pid.setpoint = SETPOINTS[0]

        self.counter = 0
        self.trigger = 60
        self.point = 0

        self.path = Path()

    def define_message_for_all(self):
        return None

    def control(self) -> CommandsDict:
        self.counter += 1
        if self.counter % self.trigger == 0:
            self.counter = 0

            self.point += 1
            self.point = self.point % 4

            self.pid.setpoint = SETPOINTS[self.point]

            print(self.true_position(), self.true_angle())
            print("STEP")
            print(self.pid.setpoint)

        if self.counter % 3 == 0:
            position = np.array([self.true_position()[0], self.true_position()[1]])
            angle = self.true_angle()
            pose = Pose(position=position, orientation=angle)
            self.path.append(pose)

        position = self.true_position()
        command = self.pid.get_command((position[0], position[1], self.true_angle()))

        return command

    def draw_bottom_layer(self):
        self.draw_path(path=self.path, color=(255, 0, 255))
        self.draw_direction()

    def draw_path(self, path: Path, color: tuple[int, int, int]):
        length = path.length()
        # print(length)
        pt2 = [0, 0]
        for ind_pt in range(length):
            pose = path.get(ind_pt)
            pt1 = pose.position + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(
                    float(pt2[0]), float(pt2[1]), float(pt1[0]), float(pt1[1]), color
                )
            pt2 = pt1

    def draw_direction(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 250 * np.array(
            [np.cos(self.true_angle()), np.sin(self.true_angle())]
        )
        color = (255, 64, 0)
        arcade.draw_line(
            float(pt2[0]), float(pt2[1]), float(pt1[0]), float(pt1[1]), color
        )


class MyMap(MapAbstract):
    def __init__(self, drone_type: type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (400, 400)

        # POSITIONS OF THE DRONES
        self._number_drones = 1
        self._drones_pos = []
        for i in range(self._number_drones):
            pos = ((-100, -100), 0)
            self._drones_pos.append(pos)

        self._drones: list[DroneAbstract] = []

        self._playground = ClosedPlayground(size=self._size_area)

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
    the_map = MyMap(drone_type=MyDrone1)

    gui = GuiSR(
        the_map=the_map,
        use_keyboard=False,
        use_mouse_measure=True,
        enable_visu_noises=False,
    )

    gui.run()


if __name__ == "__main__":
    main()
