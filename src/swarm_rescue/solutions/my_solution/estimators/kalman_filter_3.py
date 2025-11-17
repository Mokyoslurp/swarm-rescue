import numpy as np

from swarm_rescue.simulation.utils.utils import deg2rad, normalize_angle
from swarm_rescue.solutions.my_solution.estimators.extended_kalman_filter import (
    ExtendedKalmanFilter,
)
from swarm_rescue.solutions.my_solution.estimators.filter_parameters import (
    FilterParameters,
)


class EKF3(ExtendedKalmanFilter):
    """EKF implementation 3

    State:
        x
        y
        theta
        x_dot
        y_dot
        theta_dot
        gps_biais_x
        gps_biais_y
        compass_biais

    Measures:
        x_gps
        y_gps
        theta_compass
        v_odometer
        alpha_odometer
        d_theta_odometer

    Command:
        linear force frontways
        linear force sideways
        angular speed

    """

    @property
    def x(self):
        return self.X[0]

    @property
    def y(self):
        return self.X[1]

    @property
    def theta(self):
        return self.X[2]

    @property
    def x_dot(self):
        return self.X[3]

    @property
    def y_dot(self):
        return self.X[4]

    @property
    def theta_dot(self):
        return self.X[5]

    @property
    def gps_biais_x(self):
        return self.X[6]

    @property
    def gps_biais_y(self):
        return self.X[7]

    @property
    def compass_biais(self):
        return self.X[8]

    def __init__(
        self,
        Q=np.square(
            np.diag([0.00001, 0.00001, 0.001, 1, 1, 1, 10 * 1, 2 * 1, 2 * 0.014])
        ),
        R=np.square(np.diag([5, 5, deg2rad(4.0), 0.2, deg2rad(8.0), deg2rad(1.0)])),
        X0=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        param=FilterParameters(),
    ):
        super().__init__(Q, R, X0, param)

        self.P[0, 0] = 0
        self.P[1, 1] = 0
        self.P[2, 2] = 0

        self.previous_x = self.x
        self.previous_y = self.y
        self.previous_theta = self.theta

    def displacement_distance(self):
        d = np.sqrt(self.x_dot**2 + self.x_dot**2)
        if d == 0:
            return 1
        else:
            return d

    def state_model(self, u: np.ndarray) -> np.ndarray:
        k = self.param.im * self.param.kf

        x = self.x_dot
        y = self.y_dot

        theta = self.param.kw * u[2]
        theta = normalize_angle(theta)

        x_dot = (
            k * (u[0] * np.cos(self.theta) - u[1] * np.sin(self.theta))
            - self.param.v * self.x_dot
        )
        y_dot = (
            k * (u[0] * np.sin(self.theta) + u[1] * np.cos(self.theta))
            - self.param.v * self.y_dot
        )

        f_x = self.X + self.param.dt * np.array(
            [
                x,
                y,
                theta,
                x_dot,
                y_dot,
                0,
                0,
                0,
                0,
            ]
        )

        return f_x

    def state_model_jacobian(self, u: np.ndarray) -> np.ndarray:
        k = self.param.im * self.param.kf

        F = np.eye(9) + self.param.dt * np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    0,
                    0,
                    -k * (u[0] * np.sin(self.theta) + u[1] * np.cos(self.theta)),
                    -self.param.v,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    k * (u[0] * np.cos(self.theta) - u[1] * np.sin(self.theta)),
                    0,
                    -self.param.v,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        return F

    def measure_model(self) -> np.ndarray:
        y_hat = np.array(
            [
                self.x + self.gps_biais_x,
                self.y + self.gps_biais_y,
                normalize_angle(self.theta + self.compass_biais),
                self.x_dot,
                self.y_dot,
                self.theta_dot,
            ]
        )
        return y_hat

    def measure_model_jacobian(self) -> np.ndarray:
        C = np.array(
            [
                [1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        return C

    def step(
        self,
        command: np.ndarray,
        measurement: np.ndarray,
    ):
        self.previous_x = self.x
        self.previous_y = self.y
        self.previous_theta = self.theta

        self.predict(command)
        self.correct(measurement)

        self.X[2] = normalize_angle(self.X[2])
