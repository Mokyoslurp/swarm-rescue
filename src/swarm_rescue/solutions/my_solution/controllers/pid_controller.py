import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import clamp, normalize_angle
from swarm_rescue.solutions.my_solution.controllers.abstract_controller import (
    AbstractController,
)


class PIDController(AbstractController):
    def __init__(
        self,
        Kp: tuple[float, float] = (0.0, 0.0),
        Ki: tuple[float, float] = (0.0, 0.0),
        Kd: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.cumulated_error = (0, 0, 0)
        self.previous_error = (0, 0, 0)

    def get_command(self, current_pose: tuple[float, float, float]) -> CommandsDict:
        # If no setpoint, return a static command
        if self.setpoint is not None:
            # Extract errors
            x_error = self.setpoint[0] - current_pose[0]
            y_error = self.setpoint[1] - current_pose[1]
            r_error = float(normalize_angle(self.setpoint[2] - current_pose[2]))

            # Commands computation
            x_command = (
                self.Kp[0] * x_error
                + self.Ki[0] * self.cumulated_error[0]
                + self.Kd[0] * (x_error - self.previous_error[0])
            )

            y_command = (
                self.Kp[0] * y_error
                + self.Ki[0] * self.cumulated_error[1]
                + self.Kd[0] * (y_error - self.previous_error[1])
            )

            r_command = (
                self.Kp[1] * r_error
                + self.Ki[1] * self.cumulated_error[2]
                + self.Kd[1] * (r_error - self.previous_error[2])
            )

            # Cumulated error for integral controller
            self.cumulated_error += (x_error, y_error, r_error)

            # Previous error for derivative controller
            self.previous_error = [x_error, y_error, r_error]

            # Transform commands in drone frame
            cos_r = np.cos(current_pose[2])
            sin_r = np.sin(current_pose[2])
            forward = cos_r * x_command + sin_r * y_command
            lateral = cos_r * y_command - sin_r * x_command

            # Clamp commands
            forward = clamp(forward, -1.0, 1.0)
            lateral = clamp(lateral, -1.0, 1.0)
            rotation = clamp(r_command, -1.0, 1.0)

            return {
                "forward": forward,
                "lateral": lateral,
                "rotation": rotation,
            }

        return {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
        }
