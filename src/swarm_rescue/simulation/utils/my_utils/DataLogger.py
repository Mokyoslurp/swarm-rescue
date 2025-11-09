import matplotlib.pyplot as plt


class DataLogger:
    """
    The DataVisualize class provides functions to visualize data about the simulation.
    """

    def __init__(self,
                plot_gps_data: bool = False,
                plot_mag_data: bool = False,
                simulation_score: bool = False,
            ) -> None:
        """
        Initialize DataVisualize.

        Args:
            plot_gps_data (bool): Plot the real, measured, and estimated GPS points.
            plot_mag_data (bool): Plot the real, measured, and estimated compass points.
            simulation_score (bool): Print the final score for the simulation.
            true_gps_points (List[Tuple[float, float]]): List to store true GPS points.
            measured_gps_points (List[Tuple[float, float]]): List to store measured GPS points.
        """

        self.plot_gps_data = plot_gps_data
        self.plot_mag_data = plot_mag_data
        self.simulation_score = simulation_score
        self._drones = []

        # Initialize data storage lists
        self.true_gps_points = []
        self.measured_gps_points = []
        self.true_mag_points = []
        self.measured_mag_points = []
        self.true_velocity_points = []
        self.true_angular_velocity_points = []


    def record_all_data(self, playground_agents) -> None:
        """Record GPS, magnetic, velocity and angular velocity data."""
        for drone in playground_agents:
            self.true_gps_points.append([drone.true_position()[0], drone.true_position()[1]])
            self.measured_gps_points.append([drone.measured_gps_position()[0], drone.measured_gps_position()[1]])
            self.true_mag_points.append(drone.true_angle())
            self.measured_mag_points.append(drone.measured_compass_angle())
            self.true_velocity_points.append(drone.true_velocity())
            self.true_angular_velocity_points.append(drone.true_angular_velocity())

    def plot_gps(self) -> plt.Figure:
        """
        Plot the GPS data (true and measured positions).
        """
        # Unzip the list of (x, y) tuples for true and measured GPS points
        true_x_coords, true_y_coords = zip(*self.true_gps_points)
        true_x_coords = list(true_x_coords)
        true_y_coords = list(true_y_coords)
        measured_x_coords, measured_y_coords = zip(*self.measured_gps_points)
        measured_x_coords = list(measured_x_coords)
        measured_y_coords = list(measured_y_coords)

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("GPS Coordinates Data")

        # Plot X coordinates
        axes[0].plot(true_x_coords, label='X coordinates')
        axes[0].plot(measured_x_coords, label='Measured X coordinates')
        axes[0].set_ylabel('X')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Y coordinates
        axes[1].plot(true_y_coords, label='Y coordinates')
        axes[1].plot(measured_y_coords, label='Measured Y coordinates')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])


    def plot_mag(self):
        """Placeholder for magnetic data visualization."""
        # Create figure with two subplots
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Magnetic Angle Data")

        # Plot X coordinates
        ax.plot(self.true_mag_points, label='True angles')
        ax.plot(self.measured_mag_points, label='Measured angles')
        ax.set_ylabel('X')
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        

    def display_score(self):
        """Placeholder for displaying simulation score."""
        if self.simulation_score:
            print("Displaying simulation score...")


    def display(self):
        """Run the data visualization based on the enabled options."""
        if self.plot_gps_data:
            self.plot_gps()
        if self.plot_mag_data:
            self.plot_mag()
        if self.simulation_score:
            self.display_score()

        plt.show()  # show all figures at once

