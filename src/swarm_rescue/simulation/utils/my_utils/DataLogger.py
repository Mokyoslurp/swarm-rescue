import matplotlib.pyplot as plt
import numpy as np

from swarm_rescue.simulation import drone

#from swarm_rescue.solutions.my_solution.estimators.kalman_filter import KalmanFilter


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
            true_mag_points (List[float]): List to store true magnetic angle points.
            measured_mag_points (List[float]): List to store measured magnetic angle points.
            true_velocity_points (List[float]): List to store true velocity points.
            true_angular_velocity_points (List[float]): List to store true angular velocity points.
        """

        self.plot_gps_data = plot_gps_data
        self.plot_mag_data = plot_mag_data
        self.simulation_score = simulation_score
        self._drones = []

        # Initialize data storage lists
        self.true_gps_points = []
        self.measured_gps_points = []
        self.estimated_gps_points = []
        self.true_mag_points = []
        self.measured_mag_points = []
        self.estimated_mag_points = []
        self.true_velocity_points = []
        self.true_angular_velocity_points = []

        #self.kalmanfilter = KalmanFilter(dt=1, m=1.2, k1=2.0, k2=1.5, k3=0.8, alpha=0.1)


    def record_all_data(self, playground_agents, drones_commands) -> None:
        """Record GPS, magnetic, velocity and angular velocity data."""
        for drone in playground_agents:
            # Record GPS data
            self.true_gps_points.append([drone.true_position()[0], drone.true_position()[1]])
            self.measured_gps_points.append([drone.measured_gps_position()[0], drone.measured_gps_position()[1]])

            x = self.measured_gps_points[-1][0]
            y = self.measured_gps_points[-1][1]
            theta = drone.true_angle()
            cmd = drones_commands.get(drone, {})
            u1 = cmd.get("forward", 0.0)
            u2 = cmd.get("lateral", 0.0)
            u3 = cmd.get("rotation", 0.0)
            g  = cmd.get("grasper", 0)
            self.estimated_gps_points.append(kalmanfilter(x, y, theta, u1, u2, u3))

            # Record magnetic data
            self.true_mag_points.append(drone.true_angle())
            self.measured_mag_points.append(drone.measured_compass_angle())

            # Record velocity and angular velocity data
            self.true_velocity_points.append(drone.true_velocity())
            self.true_angular_velocity_points.append(drone.true_angular_velocity())


    def plot_gps(self) -> plt.Figure:
        """
        Plot the GPS data (true, measured, and estimated positions).
        """
        # Unzip the list of (x, y) tuples for true, measured, and estimated GPS points
        true_x_coords, true_y_coords = zip(*self.true_gps_points)
        true_x_coords = list(true_x_coords)
        true_y_coords = list(true_y_coords)

        measured_x_coords, measured_y_coords = zip(*self.measured_gps_points)
        measured_x_coords = list(measured_x_coords)
        measured_y_coords = list(measured_y_coords)

        estimated_x_coords, estimated_y_coords = zip(*self.estimated_gps_points)
        estimated_x_coords = list(estimated_x_coords)
        estimated_y_coords = list(estimated_y_coords)

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("GPS Coordinates Data")

        # Plot X coordinates
        axes[0].plot(true_x_coords, label='X coordinates')
        axes[0].plot(measured_x_coords, label='Measured X coordinates')
        axes[0].plot(estimated_x_coords, label='Estimated X coordinates')
        axes[0].set_ylabel('X coordinates (m)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Y coordinates
        axes[1].plot(true_y_coords, label='Y coordinates')
        axes[1].plot(measured_y_coords, label='Measured Y coordinates')
        axes[1].plot(estimated_y_coords, label='Estimated Y coordinates')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Y coordinates (m)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Calculate errors
        errors = np.sqrt((np.array(true_x_coords) - np.array(measured_x_coords)) ** 2 + (np.array(true_y_coords) - np.array(measured_y_coords)) ** 2)
        errors = errors[~np.isnan(errors)]

        # Calculate error statistics
        avg_error = np.mean(errors)
        max_error = np.max(np.abs(errors))
        std_error = np.std(errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        # Calculate estimation errors
        estimation_errors = np.sqrt((np.array(true_x_coords) - np.array(estimated_x_coords)) ** 2 + (np.array(true_y_coords) - np.array(estimated_y_coords)) ** 2)
        estimation_errors = estimation_errors[~np.isnan(estimation_errors)]

        # Calculate estimation error statistics
        avg_estimation_error = np.mean(estimation_errors)
        max_estimation_error = np.max(np.abs(estimation_errors))
        std_estimation_error = np.std(estimation_errors)
        estimation_rmse = np.sqrt(np.mean(estimation_errors ** 2))

        # Calculate estimation error statistics
        avg_estimation_error = np.mean(estimation_errors)
        max_estimation_error = np.max(np.abs(estimation_errors))
        std_estimation_error = np.std(estimation_errors)
        estimation_rmse = np.sqrt(np.mean(estimation_errors ** 2))

        # Print error statistics
        print("\n")
        print("===================================")
        print("GPS Error Statistics:")
        print(f"  Average error: {avg_error:.3f}m")
        print(f"  Max error: {max_error:.3f}m")
        print(f"  Std deviation: {std_error:.3f}m")
        print(f"  RMSE: {rmse:.3f}m")
        print("===================================")

         # Print estimation error statistics
        print("\n")
        print("===================================")
        print("GPS Estimation Error Statistics:")
        print(f"  Average error: {avg_estimation_error:.3f}m")
        print(f"  Max error: {max_estimation_error:.3f}m")
        print(f"  Std deviation: {std_estimation_error:.3f}m")
        print(f"  RMSE: {estimation_rmse:.3f}m")
        print("===================================")


    def plot_mag(self):
        """Placeholder for magnetic data visualization."""
        # Convert to numpy arrays for easier calculations
        true_mag_points = np.array(self.true_mag_points) * (180.0 / np.pi)  # Convert to degrees
        measured_mag_points = np.array(self.measured_mag_points) * (180.0 / np.pi)  # Convert to degrees

        # Create figure with two subplots
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle("Magnetic Angle Data")

        # Plot X coordinates
        ax.plot(true_mag_points, label='True angles(°)')
        ax.plot(measured_mag_points, label='Measured angles(°)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Angle (°)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Calculate  errors
        errors = true_mag_points - measured_mag_points
        errors = errors[~np.isnan(errors)]

        # Calculate error statistics
        avg_error = np.mean(errors)
        max_error = np.max(np.abs(errors))
        std_error = np.std(errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        # Print error statistics
        print("\n")
        print("===================================")
        print("Magnetic Angle Error Statistics:")
        print(f"  Average error: {avg_error:.3f}°")
        print(f"  Max error: {max_error:.3f}°")
        print(f"  Std deviation: {std_error:.3f}°")
        print(f"  RMSE: {rmse:.3f}°")
        print("===================================")
        

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


























def kalmanfilter(x_meas, y_meas, theta, u1, u2, u3):
    # --- Constants (adjust for your drone) ---
    m = 50.0        # mass
    alpha = 0.05    # drag coefficient
    k1 = 10.0
    k2 = 10.0
    k3 = 10.0
    dt = 1       # sampling time

    print("u1:", u1, " u2:", u2, " u3:", u3)

    # --- Persistent EKF state (could be part of your class) ---
    x_hat = np.zeros((5, 1))     # [x, y, theta, xdot, ydot]
    P = np.eye(5) * 0.1          # initial covariance
    Q = np.diag([0.01, 0.01, 0.001, 0.05, 0.05])  # process noise
    R = np.diag([0.5, 0.5])      # measurement noise (for GPS x,y)
    """
    Extended Kalman Filter update for the drone model.
    Inputs:
      x_meas, y_meas  -> GPS measurements
      theta            -> measured or estimated heading
      u1, u2, u3       -> control inputs
    Returns:
      estimated position (x_est, y_est)
    """

    # === 1. Predict step ===
    x1, x2, x3, x4, x5 = x_hat.flatten()

    # Nonlinear dynamics (f)
    fx = np.array([
        x4,
        x5,
        k3 * u3,
        (1/m) * (k1*u1*np.cos(x3) - k2*u2*np.sin(x3) - alpha*x4),
        (1/m) * (k1*u1*np.sin(x3) + k2*u2*np.cos(x3) - alpha*x5)
    ]).reshape(-1, 1)

    # Predicted state
    x_hat = x_hat + fx * dt

    # Jacobian A(x,u)
    A = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, -(k1*u1*np.sin(x3) + k2*u2*np.cos(x3))/m, -alpha/m, 0],
        [0, 0,  (k1*u1*np.cos(x3) - k2*u2*np.sin(x3))/m, 0, -alpha/m]
    ])

    # Linearized state transition matrix
    F = np.eye(5) + A * dt

    # Covariance prediction
    P = F @ P @ F.T + Q

    # === 2. Update step (measurement z = [x_meas, y_meas]) ===
    z = np.array([[x_meas], [y_meas]])
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]])

    # Innovation
    y_tilde = z - H @ x_hat

    # Kalman gain
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # State update
    x_hat = x_hat + K @ y_tilde

    # Covariance update
    P = (np.eye(5) - K @ H) @ P

    # Return estimated GPS position
    return x_hat[0,0], x_hat[1,0]
