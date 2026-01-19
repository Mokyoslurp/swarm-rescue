import numpy as np
from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01


from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
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
            self.kalman_filter.x,
            self.kalman_filter.y,
            self.kalman_filter.theta,
        )

        self.occupancy_grid.update_grid(pose=self.pose)

        command = self.state_machine.get_command(self)
        self.previous_command = np.array(
            [command["forward"], command["lateral"], command["rotation"]]
        )

        return command


def main():
    the_map = MapIntermediate01(
        drone_type=MyDroneSolution, zones_config=(ZoneType.NO_GPS_ZONE,)
    )

    gui = GuiSR(
        the_map=the_map,
        use_keyboard=False,
    )
    gui.run()


if __name__ == "__main__":
    main()
