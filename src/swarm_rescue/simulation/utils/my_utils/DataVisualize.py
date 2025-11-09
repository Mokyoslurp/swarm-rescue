import matplotlib.pyplot as plt


class DataVisualize:
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

        # Initialize GPS data storage
        self.true_gps_points = []
        self.measured_gps_points = []

    def record_gps(self):
        """Record GPS data."""
        if self.plot_gps_data:
            print("Recording GPS data...")

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
        fig.suptitle("GPS Coordinate Components")

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
        plt.show()


    def plot_mag(self):
        """Placeholder for magnetic data visualization."""
        if self.plot_mag_data:
            print("Plotting magnetic data...")

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

