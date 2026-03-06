"""
Baseline Controllers for Tennessee Eastman Process

This module implements various baseline control strategies for comparison
with the multi-agent reinforcement learning approach. Includes PID cascades,
NMPC, and schedule-then-control approaches.

Key Features:
- Well-tuned PID cascade controllers
- Nonlinear Model Predictive Control (NMPC)
- Schedule-then-control sequential approach
- Performance monitoring and logging
- Configurable controller parameters

Author: Implementation for Multi-Agent Digital Twin Research
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PIDParams:
    """PID controller parameters"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    setpoint: float  # Target setpoint
    output_min: float = 0.0  # Minimum output
    output_max: float = 100.0  # Maximum output
    integral_min: float = -100.0  # Anti-windup limits
    integral_max: float = 100.0


@dataclass
class ControllerMetrics:
    """Metrics for controller performance monitoring"""
    tracking_errors: List[float] = None
    control_efforts: List[float] = None
    setpoint_changes: int = 0
    constraint_violations: int = 0
    computation_times: List[float] = None
    
    def __post_init__(self):
        if self.tracking_errors is None:
            self.tracking_errors = []
        if self.control_efforts is None:
            self.control_efforts = []
        if self.computation_times is None:
            self.computation_times = []


class BaseController(ABC):
    """Abstract base class for all controllers"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = ControllerMetrics()
        self.last_time = None
        self.initialized = False
    
    @abstractmethod
    def compute_action(self, 
                      measurements: np.ndarray, 
                      setpoints: np.ndarray, 
                      dt: float) -> np.ndarray:
        """Compute control action given measurements and setpoints"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller internal state"""
        pass
    
    def get_metrics(self) -> ControllerMetrics:
        """Get controller performance metrics"""
        return self.metrics
    
    def log_performance(self, error: float, effort: float, comp_time: float):
        """Log performance metrics"""
        self.metrics.tracking_errors.append(error)
        self.metrics.control_efforts.append(effort)
        self.metrics.computation_times.append(comp_time)


class PIDController(BaseController):
    """Single-loop PID controller with anti-windup"""
    
    def __init__(self, name: str, params: PIDParams):
        super().__init__(name)
        self.params = params
        
        # Internal state
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        
    def compute_action(self, 
                      measurements: np.ndarray, 
                      setpoints: np.ndarray, 
                      dt: float) -> np.ndarray:
        """Compute PID control action"""
        start_time = time.time()
        
        # Extract relevant measurement (assuming single-loop)
        measurement = measurements[0] if len(measurements) > 0 else 0.0
        setpoint = setpoints[0] if len(setpoints) > 0 else self.params.setpoint
        
        # Calculate error
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.params.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.params.integral_min, self.params.integral_max)
        i_term = self.params.ki * self.integral
        
        # Derivative term
        if self.last_measurement is not None:
            derivative = (measurement - self.last_measurement) / dt
            d_term = -self.params.kd * derivative  # Derivative on measurement
        else:
            d_term = 0.0
        
        # Compute output
        output = p_term + i_term + d_term
        output = np.clip(output, self.params.output_min, self.params.output_max)
        
        # Update state
        self.last_error = error
        self.last_measurement = measurement
        
        # Log performance
        comp_time = time.time() - start_time
        self.log_performance(abs(error), abs(output), comp_time)
        
        return np.array([output])
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        self.metrics = ControllerMetrics()


class CascadePIDController(BaseController):
    """Cascade PID controller for multi-loop control"""
    
    def __init__(self, name: str, pid_configs: List[Tuple[int, int, PIDParams]]):
        """
        Initialize cascade PID controller
        
        Args:
            name: Controller name
            pid_configs: List of (measurement_idx, output_idx, PIDParams) tuples
        """
        super().__init__(name)
        self.pid_configs = pid_configs
        self.pid_controllers = []
        
        # Create individual PID controllers
        for i, (meas_idx, out_idx, params) in enumerate(pid_configs):
            pid = PIDController(f"{name}_loop_{i}", params)
            self.pid_controllers.append((meas_idx, out_idx, pid))
    
    def compute_action(self, 
                      measurements: np.ndarray, 
                      setpoints: np.ndarray, 
                      dt: float) -> np.ndarray:
        """Compute cascade PID control actions"""
        start_time = time.time()
        
        # Initialize output array
        outputs = np.zeros(12)  # 12 manipulated variables for TEP
        
        # Compute each PID loop
        total_error = 0.0
        total_effort = 0.0
        
        for meas_idx, out_idx, pid in self.pid_controllers:
            # Extract measurement and setpoint
            measurement = measurements[meas_idx] if meas_idx < len(measurements) else 0.0
            setpoint = setpoints[meas_idx] if meas_idx < len(setpoints) else pid.params.setpoint
            
            # Compute PID action
            action = pid.compute_action(np.array([measurement]), np.array([setpoint]), dt)
            outputs[out_idx] = action[0]
            
            # Accumulate metrics
            total_error += abs(setpoint - measurement)
            total_effort += abs(action[0])
        
        # Log performance
        comp_time = time.time() - start_time
        self.log_performance(total_error, total_effort, comp_time)
        
        return outputs
    
    def reset(self):
        """Reset all PID controllers"""
        for _, _, pid in self.pid_controllers:
            pid.reset()
        self.metrics = ControllerMetrics()


class NMPCController(BaseController):
    """Nonlinear Model Predictive Controller"""
    
    def __init__(self, 
                 name: str,
                 prediction_horizon: int = 20,
                 control_horizon: int = 5,
                 dt: float = 6.0):
        super().__init__(name)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt
        
        # Model parameters (simplified TEP model)
        self.model_params = self._initialize_model_params()
        
        # Optimization weights
        self.Q = np.eye(4) * 10.0  # State weights (key variables)
        self.R = np.eye(12) * 0.1  # Control weights
        self.S = np.eye(4) * 100.0  # Terminal weights
        
        # Constraints
        self.u_min = np.zeros(12)
        self.u_max = np.ones(12) * 100.0
        self.du_max = np.ones(12) * 5.0  # Rate limits
        
        # State
        self.last_u = np.ones(12) * 50.0  # Initial control
        
    def _initialize_model_params(self) -> Dict:
        """Initialize simplified model parameters"""
        return {
            'reactor_time_constant': 600.0,  # 10 minutes
            'separator_time_constant': 300.0,  # 5 minutes
            'stripper_time_constant': 180.0,  # 3 minutes
            'composition_time_constant': 900.0,  # 15 minutes
            'cooling_gain': -0.5,  # Cooling effect on temperature
            'valve_gain': 0.8,  # Valve effect on levels
        }
    
    def _predict_state(self, 
                      current_state: np.ndarray, 
                      control_sequence: np.ndarray) -> np.ndarray:
        """Predict future states using simplified model"""
        
        # Extract key state variables
        reactor_temp = current_state[8]
        separator_level = current_state[11]
        stripper_level = current_state[14]
        composition = np.mean(current_state[22:28])
        
        state = np.array([reactor_temp, separator_level, stripper_level, composition])
        
        # Predict over horizon
        predicted_states = np.zeros((self.prediction_horizon + 1, 4))
        predicted_states[0] = state
        
        for k in range(self.prediction_horizon):
            u = control_sequence[min(k, self.control_horizon - 1)]
            
            # Simplified dynamics
            reactor_cooling = u[10]  # Reactor cooling valve
            separator_valve = u[6]   # Separator liquid valve
            stripper_valve = u[7]    # Stripper liquid valve
            
            # Temperature dynamics
            dtemp_dt = (120.0 - state[0]) / self.model_params['reactor_time_constant'] + \
                      self.model_params['cooling_gain'] * (reactor_cooling - 50.0)
            
            # Level dynamics
            dlevel_sep_dt = (50.0 - state[1]) / self.model_params['separator_time_constant'] + \
                           self.model_params['valve_gain'] * (50.0 - separator_valve)
            
            dlevel_str_dt = (50.0 - state[2]) / self.model_params['stripper_time_constant'] + \
                           self.model_params['valve_gain'] * (50.0 - stripper_valve)
            
            # Composition dynamics (simplified)
            dcomp_dt = (30.0 - state[3]) / self.model_params['composition_time_constant']
            
            # Update state
            state = state + np.array([dtemp_dt, dlevel_sep_dt, dlevel_str_dt, dcomp_dt]) * self.dt
            predicted_states[k + 1] = state
        
        return predicted_states
    
    def _objective_function(self, 
                           control_sequence: np.ndarray, 
                           current_state: np.ndarray, 
                           setpoints: np.ndarray) -> float:
        """NMPC objective function"""
        
        # Reshape control sequence
        u_seq = control_sequence.reshape(self.control_horizon, 12)
        
        # Predict states
        predicted_states = self._predict_state(current_state, u_seq)
        
        # Extract setpoints for key variables
        target_temp = setpoints[8] if len(setpoints) > 8 else 120.0
        target_sep_level = setpoints[11] if len(setpoints) > 11 else 50.0
        target_str_level = setpoints[14] if len(setpoints) > 14 else 50.0
        target_comp = np.mean(setpoints[22:28]) if len(setpoints) > 27 else 30.0
        
        targets = np.array([target_temp, target_sep_level, target_str_level, target_comp])
        
        # Calculate cost
        cost = 0.0
        
        # State tracking cost
        for k in range(1, self.prediction_horizon + 1):
            error = predicted_states[k] - targets
            cost += error.T @ self.Q @ error
        
        # Terminal cost
        terminal_error = predicted_states[-1] - targets
        cost += terminal_error.T @ self.S @ terminal_error
        
        # Control effort cost
        for k in range(self.control_horizon):
            cost += u_seq[k].T @ self.R @ u_seq[k]
        
        # Rate penalty
        for k in range(self.control_horizon):
            if k == 0:
                du = u_seq[k] - self.last_u
            else:
                du = u_seq[k] - u_seq[k-1]
            cost += 0.1 * np.sum(du**2)
        
        return cost
    
    def compute_action(self, 
                      measurements: np.ndarray, 
                      setpoints: np.ndarray, 
                      dt: float) -> np.ndarray:
        """Compute NMPC control action"""
        start_time = time.time()
        
        # Initial guess for control sequence
        u0 = np.tile(self.last_u, self.control_horizon)
        
        # Bounds for optimization
        bounds = []
        for k in range(self.control_horizon):
            for j in range(12):
                bounds.append((self.u_min[j], self.u_max[j]))
        
        # Constraints for rate limits
        constraints = []
        for k in range(self.control_horizon):
            for j in range(12):
                if k == 0:
                    # Rate limit from last control
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda u, k=k, j=j: self.du_max[j] - abs(u[k*12 + j] - self.last_u[j])
                    })
                else:
                    # Rate limit between consecutive controls
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda u, k=k, j=j: self.du_max[j] - abs(u[k*12 + j] - u[(k-1)*12 + j])
                    })
        
        try:
            # Solve optimization problem
            result = minimize(
                fun=lambda u: self._objective_function(u, measurements, setpoints),
                x0=u0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if result.success:
                # Extract first control action
                optimal_u = result.x[:12]
                self.last_u = optimal_u.copy()
            else:
                logger.warning(f"NMPC optimization failed: {result.message}")
                optimal_u = self.last_u.copy()
                
        except Exception as e:
            logger.error(f"NMPC computation error: {e}")
            optimal_u = self.last_u.copy()
        
        # Log performance
        comp_time = time.time() - start_time
        tracking_error = np.sum(np.abs(measurements[:4] - setpoints[:4])) if len(setpoints) >= 4 else 0.0
        control_effort = np.sum(np.abs(optimal_u))
        self.log_performance(tracking_error, control_effort, comp_time)
        
        return optimal_u
    
    def reset(self):
        """Reset NMPC controller"""
        self.last_u = np.ones(12) * 50.0
        self.metrics = ControllerMetrics()


class ScheduleThenControlController(BaseController):
    """Schedule-then-control sequential approach"""
    
    def __init__(self, name: str):
        super().__init__(name)
        
        # Create PID controller for tracking scheduled setpoints
        pid_configs = [
            (8, 10, PIDParams(kp=2.0, ki=0.1, kd=0.5, setpoint=120.0)),  # Reactor temp -> cooling
            (11, 6, PIDParams(kp=1.5, ki=0.05, kd=0.2, setpoint=50.0)),  # Sep level -> valve
            (14, 7, PIDParams(kp=1.2, ki=0.08, kd=0.3, setpoint=50.0)),  # Str level -> valve
        ]
        
        self.pid_controller = CascadePIDController(f"{name}_pid", pid_configs)
        
        # Scheduling parameters
        self.schedule = []
        self.current_schedule_idx = 0
        self.last_schedule_time = 0.0
        
        # Initialize default schedule
        self._initialize_default_schedule()
    
    def _initialize_default_schedule(self):
        """Initialize default production schedule"""
        self.schedule = [
            {'time': 0, 'mode': 1, 'setpoints': {'temp': 120.0, 'sep_level': 50.0, 'str_level': 50.0}},
            {'time': 1200, 'mode': 2, 'setpoints': {'temp': 125.0, 'sep_level': 45.0, 'str_level': 55.0}},
            {'time': 2400, 'mode': 3, 'setpoints': {'temp': 115.0, 'sep_level': 55.0, 'str_level': 45.0}},
            {'time': 3600, 'mode': 1, 'setpoints': {'temp': 120.0, 'sep_level': 50.0, 'str_level': 50.0}},
        ]
    
    def _get_current_setpoints(self, current_time: float) -> Dict:
        """Get current setpoints based on schedule"""
        
        # Find current schedule entry
        current_entry = self.schedule[0]
        for entry in self.schedule:
            if current_time >= entry['time']:
                current_entry = entry
            else:
                break
        
        return current_entry['setpoints']
    
    def compute_action(self, 
                      measurements: np.ndarray, 
                      setpoints: np.ndarray, 
                      dt: float) -> np.ndarray:
        """Compute schedule-then-control action"""
        start_time = time.time()
        
        # Update time
        current_time = self.last_schedule_time + dt
        self.last_schedule_time = current_time
        
        # Get scheduled setpoints
        scheduled_setpoints = self._get_current_setpoints(current_time)
        
        # Convert to setpoint array
        sp_array = np.zeros(len(measurements))
        sp_array[8] = scheduled_setpoints['temp']
        sp_array[11] = scheduled_setpoints['sep_level']
        sp_array[14] = scheduled_setpoints['str_level']
        
        # Use PID controller to track scheduled setpoints
        action = self.pid_controller.compute_action(measurements, sp_array, dt)
        
        # Log performance
        comp_time = time.time() - start_time
        tracking_error = np.sum(np.abs(measurements[[8,11,14]] - sp_array[[8,11,14]]))
        control_effort = np.sum(np.abs(action))
        self.log_performance(tracking_error, control_effort, comp_time)
        
        return action
    
    def reset(self):
        """Reset schedule-then-control controller"""
        self.pid_controller.reset()
        self.current_schedule_idx = 0
        self.last_schedule_time = 0.0
        self.metrics = ControllerMetrics()


def create_tep_pid_controller() -> CascadePIDController:
    """Create well-tuned PID cascade controller for TEP"""
    
    pid_configs = [
        # (measurement_idx, output_idx, PIDParams)
        (8, 10, PIDParams(kp=2.0, ki=0.1, kd=0.5, setpoint=120.0, output_min=0.0, output_max=100.0)),  # Reactor temp
        (11, 6, PIDParams(kp=1.5, ki=0.05, kd=0.2, setpoint=50.0, output_min=0.0, output_max=100.0)),  # Separator level
        (14, 7, PIDParams(kp=1.2, ki=0.08, kd=0.3, setpoint=50.0, output_min=0.0, output_max=100.0)),  # Stripper level
        (12, 6, PIDParams(kp=0.8, ki=0.02, kd=0.1, setpoint=2633.7, output_min=0.0, output_max=100.0)),  # Separator pressure
    ]
    
    return CascadePIDController("TEP_PID_Cascade", pid_configs)


def create_tep_nmpc_controller() -> NMPCController:
    """Create NMPC controller for TEP"""
    return NMPCController(
        name="TEP_NMPC",
        prediction_horizon=20,
        control_horizon=5,
        dt=6.0
    )


def create_schedule_then_control() -> ScheduleThenControlController:
    """Create schedule-then-control controller for TEP"""
    return ScheduleThenControlController("TEP_Schedule_Then_Control")


# Test the baseline controllers
if __name__ == "__main__":
    # Test PID controller
    print("Testing PID Controller...")
    pid_controller = create_tep_pid_controller()
    
    # Simulate some measurements
    measurements = np.random.randn(41) * 5 + np.array([
        0.25052, 4.0975, 9.3477, 22.949, 18.776, 50.338, 2705.0, 75.0,
        120.4, 0.33712, 80.109, 50.0, 2633.7, 25.16, 50.0, 3102.2,
        22.949, 65.731, 230.31, 341.43, 94.599, 77.297, 32.188, 13.823,
        24.644, 18.776, 8.4036, 1.5699, 26.902, 4.5301, 7.2996, 51.595,
        11.859, 1.7677, 0.31827, 0.01787, 53.722, 43.827, 1.8968, 0.42073, 0.024297
    ])
    
    setpoints = measurements.copy()
    setpoints[8] = 125.0  # Change reactor temperature setpoint
    
    action = pid_controller.compute_action(measurements, setpoints, 6.0)
    print(f"PID action: {action}")
    
    # Test NMPC controller
    print("\nTesting NMPC Controller...")
    nmpc_controller = create_tep_nmpc_controller()
    
    action = nmpc_controller.compute_action(measurements, setpoints, 6.0)
    print(f"NMPC action: {action}")
    
    # Test Schedule-then-Control
    print("\nTesting Schedule-then-Control...")
    stc_controller = create_schedule_then_control()
    
    action = stc_controller.compute_action(measurements, setpoints, 6.0)
    print(f"Schedule-then-Control action: {action}")
    
    print("\nBaseline controllers test completed successfully!")

