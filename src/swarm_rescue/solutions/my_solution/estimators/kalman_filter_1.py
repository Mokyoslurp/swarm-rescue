import numpy as np

from swarm_rescue.simulation.utils.utils import deg2rad, normalize_angle
from swarm_rescue.solutions.my_solution.estimators.extended_kalman_filter import (
    ExtendedKalmanFilter,
)
from swarm_rescue.solutions.my_solution.estimators.filter_parameters import (
    FilterParameters,
)


class EKF1(ExtendedKalmanFilter):
    """EKF implementation 1

    State:
        x
        y
        theta
        x_dot
        y_dot

    Measures:
        x
        y
        theta

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

    def __init__(
        self,
        Q=np.square(np.diag([0.0001, 0.0001, 0.001, 0.0001, 0.0001])),
        R=np.square(np.diag([5, 5, deg2rad(4.0)])),
        X0=np.array([0, 0, 0, 0, 0]),
        param=FilterParameters(),
    ):
        super().__init__(Q, R, X0, param)

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
            ]
        )

        return f_x

    def state_model_jacobian(self, u: np.ndarray) -> np.ndarray:
        k = self.param.im * self.param.kf

        F = np.eye(5) + self.param.dt * np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [
                    0,
                    0,
                    -k * (u[0] * np.sin(self.theta) + u[1] * np.cos(self.theta)),
                    -self.param.v,
                    0,
                ],
                [
                    0,
                    0,
                    k * (u[0] * np.cos(self.theta) - u[1] * np.sin(self.theta)),
                    0,
                    -self.param.v,
                ],
            ]
        )

        return F

    def measure_model(self) -> np.ndarray:
        return self.measure_model_jacobian() @ self.X

    def measure_model_jacobian(self) -> np.ndarray:
        C = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        return C

    def correct(self, y: np.ndarray):
        """Makes a corrective step and saves the new state

        :param y: the true measurements from the drone
        :type y: np.ndarray
        """
        if y[0] is not None and y[1] is not None and y[2] is not None:
            # Innovation
            z = y - self.measure_model()

            # Kalman gain
            H = self.measure_model_jacobian()
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)

            # State update
            self.X = self.X + K @ z

            # Covariance update
            self.P = self.P - K @ H @ self.P

    def step(
        self,
        command: np.ndarray,
        measurement: np.ndarray,
    ):
        self.predict(command)
        self.correct(measurement)

        self.X[2] = normalize_angle(self.X[2])
