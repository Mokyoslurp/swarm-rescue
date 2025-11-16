from abc import ABC, abstractmethod

import numpy as np

from swarm_rescue.solutions.my_solution.estimators.filter_parameters import (
    FilterParameters,
)


class ExtendedKalmanFilter(ABC):
    """EKF standard implmentation"""

    def __init__(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        X0: np.ndarray,
        param: FilterParameters,
    ):
        """Initializes an ExtendedKalmanFilter

        :param Q: Model covariance matrix
        :type Q: np.ndarray
        :param R: Measurement covariance matrix
        :type R: np.ndarray
        :param X0: Initial state vector
        :type X0: np.ndarray
        """
        # State vector
        self.X = X0
        # State covariance
        self.P = 1000 * np.eye(X0.size)

        # Model covariance
        self.Q = Q
        # Measure covariance
        self.R = R

        # EKF parameters
        self.param = param

    def step(
        self,
        command: np.ndarray,
        measurement: np.ndarray,
    ):
        """Makes one predictive step and one corrective step

        :param command: the input command of the drone, u
        :type command: np.ndarray
        :param measurement: the measured output of the real drone
        :type measurement: np.ndarray
        """
        self.predict(command)
        self.correct(measurement)

    @abstractmethod
    def state_model(self, u: np.ndarray) -> np.ndarray:
        """Generates new prediction for state model

        :param u: the input command for the prediction
        :type u: np.ndarray
        :return: the state prediction
        :rtype: np.ndarray
        """
        ...

    @abstractmethod
    def measure_model(self) -> np.ndarray:
        """Generates a predicted measurement from predicted state

        :return: the predicted measurement vector
        :rtype: np.ndarray
        """
        ...

    @abstractmethod
    def state_model_jacobian(self, u: np.ndarray) -> np.ndarray:
        """Generates the jacobian of the state model function

        :param u: the input command
        :type u: np.ndarray
        :return: the jacobian
        :rtype: np.ndarray
        """
        ...

    @abstractmethod
    def measure_model_jacobian(self) -> np.ndarray:
        """Generates the jacobian of the measurement model function

        :return: the jacobian
        :rtype: np.ndarray
        """
        ...

    def predict(self, u: np.ndarray):
        """Makes a predictive step and saves the new state

        :param u: the input command
        :type u: np.ndarray
        """
        # State prediction
        self.X = self.state_model(u)

        # Covariance prediction
        F = self.state_model_jacobian(u)
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, y: np.ndarray):
        """Makes a corrective step and saves the new state

        :param y: the true measurements from the drone
        :type y: np.ndarray
        """
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
