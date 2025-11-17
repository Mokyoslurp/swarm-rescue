"""A plotter implementation to visualize acquired data"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DataPlotter:
    """Plotter for data gathered with the DataLogger during a simulation"""

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        self.data = data

    def plot_position(
        self,
        estimated_x: Optional[np.ndarray] = None,
        estimated_y: Optional[np.ndarray] = None,
    ):
        """Plots the true, measured and estimated position if provided

        Args:
            estimated_x (Optional[np.ndarray]): the estimated x position
            estimated_y (Optional[np.ndarray]): teh estimated y position
        """

        # Create figure with two subplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("GPS Coordinates Data")

        # Plot X coordinates
        ax[0].plot(self.data["true_x"], label="X coordinates")
        ax[0].plot(self.data["measured_x"], label="Measured X coordinates")
        if estimated_x is not None:
            ax[0].plot(estimated_x, label="Estimated X coordinates")

        ax[0].set_ylabel("X coordinates (m)")
        ax[0].legend()
        ax[0].grid(True)

        # Plot Y coordinates
        ax[1].plot(self.data["true_y"], label="Y coordinates")
        ax[1].plot(self.data["measured_y"], label="Measured Y coordinates")
        if estimated_y is not None:
            ax[1].plot(estimated_y, label="Estimated Y coordinates")

        ax[1].set_xlabel("Index")
        ax[1].set_ylabel("Y coordinates (m)")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def plot_angle(self, estimated_theta: Optional[np.ndarray] = None):
        """Plots the true, measured, and estimated angle if provided

        Args:
            estimated_theta (Optional[np.ndarray], optional): estimated angle data.
                Defaults to None.
        """
        # Conversion to degrees
        true_theta = np.array(self.data["true_theta"]) * (180.0 / np.pi)
        measured_theta = np.array(self.data["measured_theta"]) * (180.0 / np.pi)

        # Create figure with two subplots
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Magnetic Angle Data")

        # Plot X coordinates
        ax.plot(true_theta, label="True angles(°)")
        ax.plot(measured_theta, label="Measured angles(°)")

        if estimated_theta is not None:
            estimated_theta = np.array(estimated_theta) * (180.0 / np.pi)
            ax.plot(estimated_theta, label="Estimated angles(°)")

        ax.set_xlabel("Index")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

    def plot_speed(
        self,
        estimated_vx: Optional[np.ndarray] = None,
        estimated_vy: Optional[np.ndarray] = None,
    ):
        """Plots the true, measured and estimated speed if provided

        Args:
            estimated_vx (Optional[np.ndarray]): the estimated x speed
            estimated_vy (Optional[np.ndarray]): teh estimated y speed
        """

        # Create figure with two subplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("Odometer Speed Data")

        # Plot X coordinates
        ax[0].plot(self.data["true_vx"], label="Vx speed")
        ax[0].plot(self.data["measured_vx"], label="Measured Vx speed")
        if estimated_vx is not None:
            ax[0].plot(estimated_vx, label="Estimated Vx speed")

        ax[0].set_ylabel("Vx speed (m)")
        ax[0].legend()
        ax[0].grid(True)

        # Plot Y speed
        ax[1].plot(self.data["true_vy"], label="Vy speed")
        ax[1].plot(self.data["measured_vy"], label="Measured Vy speed")
        if estimated_vy is not None:
            ax[1].plot(estimated_vy, label="Estimated Vy speed")

        ax[1].set_xlabel("Index")
        ax[1].set_ylabel("Vy speed (m)")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
