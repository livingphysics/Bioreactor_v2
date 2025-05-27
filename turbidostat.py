import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from config import Config as cfg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for growth rate estimation in a turbidostat.
    
    Based on: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181923
    
    This implements the algorithm from the cited paper, which uses an EKF to 
    estimate the specific growth rate (mu) and biomass concentration.
    """
    
    def __init__(
        self,
        initial_biomass: float,
        initial_growth_rate: float,
        process_noise_biomass: float = 1e-6,
        process_noise_growth_rate: float = 1e-7,
        measurement_noise: float = 0.01,
        dt: float = 1.0
    ):
        """
        Initialize the Extended Kalman Filter
        
        Args:
            initial_biomass: Initial biomass concentration (OD units)
            initial_growth_rate: Initial specific growth rate (h^-1)
            process_noise_biomass: Process noise variance for biomass state
            process_noise_growth_rate: Process noise variance for growth rate state
            measurement_noise: Measurement noise variance
            dt: Time step in seconds
        """
        # State vector [biomass, growth_rate]
        self.x = np.array([initial_biomass, initial_growth_rate])
        
        # State covariance matrix
        self.P = np.array([[0.1, 0], [0, 0.1]])
        
        # Process noise covariance
        self.Q = np.array([
            [process_noise_biomass, 0],
            [0, process_noise_growth_rate]
        ])
        
        # Measurement noise variance
        self.R = measurement_noise
        
        # Time step in hours (converting from seconds to hours for growth rate)
        self.dt = dt / 3600.0
        
        # History for plotting
        self.biomass_history = [initial_biomass]
        self.growth_rate_history = [initial_growth_rate]
        self.time_history = [0]
        self.measurement_history = [initial_biomass]
        self.total_time = 0.0

    def predict(self, flow_rate: float) -> None:
        """
        Predict step of the EKF.
        
        Args:
            flow_rate: The dilution rate (h^-1)
        """
        # Extract current state
        x = self.x[0]  # Biomass
        mu = self.x[1]  # Growth rate
        
        # Predict next state
        # dx/dt = mu*x - D*x (where D is the dilution rate)
        x_next = x + (mu * x - flow_rate * x) * self.dt
        mu_next = mu  # Growth rate changes slowly, predict same value
        
        # Update state
        self.x = np.array([x_next, mu_next])
        
        # Linearized system matrix (Jacobian of state transition function)
        F = np.array([
            [1 + (mu - flow_rate) * self.dt, x * self.dt],
            [0, 1]
        ])
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
        # Update total time
        self.total_time += self.dt * 3600.0  # Convert back to seconds for tracking
        
    def update(self, measurement: float) -> None:
        """
        Update step of the EKF.
        
        Args:
            measurement: Measured optical density (biomass)
        """
        # Measurement matrix (H) - we measure only the biomass
        H = np.array([[1.0, 0.0]])
        
        # Innovation (measurement residual)
        y = measurement - self.x[0]
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # Update state estimate
        self.x = self.x + K * y
        
        # Update covariance
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P
        
        # Update history
        self.biomass_history.append(self.x[0])
        self.growth_rate_history.append(self.x[1])
        self.time_history.append(self.total_time)
        self.measurement_history.append(measurement)
    
    def get_state(self) -> Tuple[float, float]:
        """Return the current state estimate (biomass, growth_rate)."""
        return self.x[0], self.x[1]
    
    def plot_history(self):
        """Plot the history of biomass and growth rate estimates."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Convert time from seconds to hours
        time_hours = np.array(self.time_history) / 3600.0
        
        # Plot biomass
        ax1.plot(time_hours, self.biomass_history, 'b-', label='Estimated Biomass')
        ax1.scatter(time_hours, self.measurement_history, c='r', marker='.', label='Measurements')
        ax1.set_ylabel('Optical Density (OD)')
        ax1.set_title('Estimated Biomass vs Measurements')
        ax1.grid(True)
        ax1.legend()
        
        # Plot growth rate
        ax2.plot(time_hours, self.growth_rate_history, 'g-', label='Estimated Growth Rate')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Growth Rate (h^-1)')
        ax2.set_title('Estimated Growth Rate')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('ekf_estimation.png')
        plt.show()


class Turbidostat:
    """
    Turbidostat controller with Extended Kalman Filter for growth rate estimation.
    Maintains culture optical density at a target setpoint by controlling dilution rate.
    """
    
    def __init__(
        self,
        bioreactor,
        target_od: float,
        pump_name: str,
        initial_flow_rate_ml_s: float = 0.01,
        temp_setpoint: float = 37.0,
        sampling_interval: float = 60.0,  # seconds
        process_noise_biomass: float = 1e-5,
        process_noise_growth_rate: float = 1e-6,
        measurement_noise: float = 0.05,
        control_gain: float = 0.5,  # Controller gain
        flow_rate_max_ml_s: float = 0.1,  # Maximum flow rate in ml/s
        od_sensor_channel: int = 0,  # Default photodiode channel for OD measurement
        dead_zone: float = 0.02,  # Dead zone around target OD
        save_data: bool = True,
        plot_live: bool = False
    ):
        """
        Initialize the turbidostat controller.
        
        Args:
            bioreactor: Bioreactor object
            target_od: Target optical density setpoint
            pump_name: Name of the pump to use (e.g., 'tube_1_in')
            initial_flow_rate_ml_s: Initial flow rate in ml/s
            temp_setpoint: Temperature setpoint in Celsius
            sampling_interval: Time between samples in seconds
            process_noise_biomass: Process noise variance for biomass state
            process_noise_growth_rate: Process noise variance for growth rate state
            measurement_noise: Measurement noise variance
            control_gain: Controller gain for flow rate adjustment
            flow_rate_max_ml_s: Maximum flow rate in ml/s
            od_sensor_channel: Photodiode channel to use for OD measurement
            dead_zone: Dead zone around target OD where no adjustment is made
            save_data: Whether to save data to CSV
            plot_live: Whether to show live plots
        """
        self.bioreactor = bioreactor
        self.target_od = target_od
        self.pump_name = pump_name
        self.flow_rate_ml_s = initial_flow_rate_ml_s
        self.temp_setpoint = temp_setpoint
        self.sampling_interval = sampling_interval
        self.control_gain = control_gain
        self.flow_rate_max_ml_s = flow_rate_max_ml_s
        self.od_sensor_channel = od_sensor_channel
        self.dead_zone = dead_zone
        self.save_data = save_data
        self.plot_live = plot_live
        
        # Calculate initial dilution rate (h^-1) based on flow rate and culture volume
        # Assuming a 10 ml culture volume
        self.culture_volume_ml = 10.0
        self.dilution_rate_h = (self.flow_rate_ml_s * 3600) / self.culture_volume_ml
        
        # Initialize Extended Kalman Filter
        # Get initial OD measurement
        initial_od = self._measure_od()
        initial_growth_rate = 0.2  # Typical starting value for bacterial growth rate (h^-1)
        
        self.ekf = ExtendedKalmanFilter(
            initial_biomass=initial_od,
            initial_growth_rate=initial_growth_rate,
            process_noise_biomass=process_noise_biomass,
            process_noise_growth_rate=process_noise_growth_rate,
            measurement_noise=measurement_noise,
            dt=sampling_interval
        )
        
        # Data logging
        self.time_points = [0]
        self.od_measurements = [initial_od]
        self.flow_rates = [self.flow_rate_ml_s]
        self.estimated_ods = [initial_od]
        self.estimated_growth_rates = [initial_growth_rate]
        self.temperature_readings = [self._measure_temperature()]
        
        # For live plotting
        self.fig = None
        self.animation = None
        if self.plot_live:
            self._setup_live_plotting()
    
    def _measure_od(self) -> float:
        """Measure optical density (OD) using the photodiode sensor."""
        with self.bioreactor.led_context():
            # Get raw photodiode readings
            readings = self.bioreactor.get_photodiodes()
            
            # Use the specified channel for OD measurement
            od_reading = readings[self.od_sensor_channel]
            
            # Apply calibration to convert voltage to OD (this is just an example - adapt to your sensor)
            # Typically, OD = -log10(I/I0) where I is transmitted light and I0 is incident light
            # For a voltage-based sensor, there might be a linear or non-linear relationship
            # This is a placeholder - you should replace with actual calibration
            od = 2.0 * od_reading - 0.1  # Example calibration
            
            return max(0.0, od)  # Ensure non-negative OD
    
    def _measure_temperature(self) -> float:
        """Measure the current temperature."""
        temps = self.bioreactor.get_vial_temp()
        # Return the first temperature reading (assuming it's the relevant one)
        return temps[0] if temps else float('nan')
    
    def _adjust_flow_rate(self, current_od: float, estimated_growth_rate: float) -> None:
        """
        Adjust the flow rate based on current OD and estimated growth rate.
        
        Args:
            current_od: Current measured OD
            estimated_growth_rate: Estimated growth rate (h^-1)
        """
        # Calculate OD error
        error = current_od - self.target_od
        
        # If within dead zone, maintain current flow rate
        if abs(error) <= self.dead_zone:
            return
        
        # Adjust flow rate based on error and estimated growth rate
        if error > 0:  # OD too high, increase flow rate
            # Calculate desired dilution rate to match growth and maintain setpoint
            desired_dilution_rate_h = estimated_growth_rate + self.control_gain * error
            
            # Convert dilution rate to flow rate
            new_flow_rate_ml_s = (desired_dilution_rate_h * self.culture_volume_ml) / 3600
            
            # Apply constraints
            self.flow_rate_ml_s = min(max(0, new_flow_rate_ml_s), self.flow_rate_max_ml_s)
        else:  # OD too low, decrease flow rate
            # Reduce flow rate proportionally to the error
            reduction_factor = 1.0 - self.control_gain * abs(error)
            self.flow_rate_ml_s = max(0, self.flow_rate_ml_s * reduction_factor)
        
        # Update dilution rate based on new flow rate
        self.dilution_rate_h = (self.flow_rate_ml_s * 3600) / self.culture_volume_ml
        
        # Apply the new flow rate to both inlet and outlet pumps
        self.bioreactor.balanced_flow(self.pump_name, self.flow_rate_ml_s)
    
    def _setup_live_plotting(self):
        """Set up live plotting of data."""
        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Initialize empty plots
        self.od_line, = ax1.plot([], [], 'b-', label='OD Measurements')
        self.od_est_line, = ax1.plot([], [], 'r--', label='Estimated OD')
        ax1.axhline(y=self.target_od, color='g', linestyle='-', label='Target OD')
        ax1.set_ylabel('Optical Density')
        ax1.set_title('Turbidostat Control')
        ax1.grid(True)
        ax1.legend()
        
        self.growth_line, = ax2.plot([], [], 'g-', label='Estimated Growth Rate')
        ax2.set_ylabel('Growth Rate (h^-1)')
        ax2.grid(True)
        ax2.legend()
        
        self.flow_line, = ax3.plot([], [], 'm-', label='Flow Rate')
        ax3.set_xlabel('Time (h)')
        ax3.set_ylabel('Flow Rate (ml/s)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        
        # Define update function for animation
        def update_plot(frame):
            # Convert time to hours
            time_h = [t / 3600 for t in self.time_points]
            
            # Update OD plot
            self.od_line.set_data(time_h, self.od_measurements)
            self.od_est_line.set_data(time_h, self.estimated_ods)
            ax1.relim()
            ax1.autoscale_view()
            
            # Update growth rate plot
            self.growth_line.set_data(time_h, self.estimated_growth_rates)
            ax2.relim()
            ax2.autoscale_view()
            
            # Update flow rate plot
            self.flow_line.set_data(time_h, self.flow_rates)
            ax3.relim()
            ax3.autoscale_view()
            
            return self.od_line, self.od_est_line, self.growth_line, self.flow_line
        
        # Create animation
        self.animation = FuncAnimation(self.fig, update_plot, interval=1000, blit=True)
        plt.show(block=False)
    
    def _save_data_to_csv(self):
        """Save collected data to a CSV file."""
        import csv
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"turbidostat_data_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'Time (s)', 'OD Measured', 'OD Estimated',
                'Growth Rate (h^-1)', 'Flow Rate (ml/s)',
                'Temperature (°C)', 'Dilution Rate (h^-1)'
            ])
            
            # Write data
            for i in range(len(self.time_points)):
                writer.writerow([
                    self.time_points[i],
                    self.od_measurements[i],
                    self.estimated_ods[i],
                    self.estimated_growth_rates[i],
                    self.flow_rates[i],
                    self.temperature_readings[i],
                    (self.flow_rates[i] * 3600) / self.culture_volume_ml
                ])
        
        logging.info(f"Data saved to {filename}")
    
    def run(self, duration_seconds: float = None):
        """
        Run the turbidostat control loop.
        
        Args:
            duration_seconds: Duration to run in seconds, None for indefinite
        """
        start_time = time.time()
        last_sample_time = start_time
        
        try:
            # Maintain temperature setpoint
            logging.info(f"Setting temperature to {self.temp_setpoint}°C")
            
            # Initial balanced flow setting
            logging.info(f"Initial flow rate: {self.flow_rate_ml_s} ml/s")
            self.bioreactor.balanced_flow(self.pump_name, self.flow_rate_ml_s)
            
            logging.info("Starting turbidostat control loop")
            logging.info(f"Target OD: {self.target_od}")
            
            # Control loop
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if runtime is up
                if duration_seconds is not None and elapsed >= duration_seconds:
                    logging.info(f"Run duration of {duration_seconds} seconds reached")
                    break
                
                # Check if it's time for a measurement
                if current_time - last_sample_time >= self.sampling_interval:
                    # Measure optical density
                    od = self._measure_od()
                    temp = self._measure_temperature()
                    
                    # Run EKF prediction step
                    self.ekf.predict(self.dilution_rate_h)
                    
                    # Run EKF update step with measurement
                    self.ekf.update(od)
                    
                    # Get estimated state
                    est_od, est_growth_rate = self.ekf.get_state()
                    
                    # Adjust flow rate based on measurement and estimates
                    self._adjust_flow_rate(od, est_growth_rate)
                    
                    # Update temperature control
                    self.bioreactor.pid_temp_controller(self.temp_setpoint, temp)
                    
                    # Store data
                    self.time_points.append(elapsed)
                    self.od_measurements.append(od)
                    self.estimated_ods.append(est_od)
                    self.estimated_growth_rates.append(est_growth_rate)
                    self.flow_rates.append(self.flow_rate_ml_s)
                    self.temperature_readings.append(temp)
                    
                    # Log data
                    logging.info(f"Time: {elapsed:.1f}s, OD: {od:.3f}, Est. OD: {est_od:.3f}, "
                                f"Est. Growth: {est_growth_rate:.4f} h^-1, Flow: {self.flow_rate_ml_s:.4f} ml/s")
                    
                    last_sample_time = current_time
                
                # Sleep to reduce CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logging.info("Turbidostat control stopped by user")
        
        except Exception as e:
            logging.error(f"Error in turbidostat control: {e}")
            raise
        
        finally:
            # Stop all pumps
            self.bioreactor.balanced_flow(self.pump_name, 0)
            
            # Save data if requested
            if self.save_data:
                self._save_data_to_csv()
            
            # Generate final plots
            self.ekf.plot_history()
            
            logging.info("Turbidostat control completed")


if __name__ == "__main__":
    from bioreactor import Bioreactor
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the bioreactor
    try:
        bioreactor = Bioreactor()
        
        # Run turbidostat with EKF
        turbidostat = Turbidostat(
            bioreactor=bioreactor,
            target_od=0.6,            # Target optical density
            pump_name='tube_1_in',    # Pump to use
            initial_flow_rate_ml_s=0.01,
            temp_setpoint=37.0,
            sampling_interval=60.0,   # Sample every 60 seconds
            save_data=True,
            plot_live=True
        )
        
        # Run for 8 hours
        turbidostat.run(duration_seconds=8 * 3600)
        
    except Exception as e:
        logging.error(f"Error: {e}")
    
    finally:
        if 'bioreactor' in locals():
            bioreactor.finish()
