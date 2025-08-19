"""
Tennessee Eastman Process Environment

This module implements a comprehensive simulation environment for the Tennessee
Eastman Process (TEP), a widely-used benchmark for process control research.

Key Features:
- Multi-time-scale operation (6s control, 5min scheduling)
- Five evaluation scenarios (S1-S5) with varying complexity
- Realistic process dynamics and constraints
- Economic cost calculation and performance metrics
- Support for both single-agent and multi-agent control

The TEP consists of five major unit operations:
1. Reactor - where reactions A+C→D, A+C+D→E, A+E→F, 3D→G occur
2. Product condenser - separates reactor output
3. Vapor-liquid separator - separates condensed and vapor phases
4. Product stripper - removes remaining lights
5. Compressor - recycles unreacted materials

Author: Implementation for Multi-Agent Digital Twin Research
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Enumeration of TEP evaluation scenarios"""
    S1_BASIC_CHANGES = "S1_basic_changes"
    S2_FEED_DRIFT = "S2_feed_drift"
    S3_COOLING_DISTURBANCE = "S3_cooling_disturbance"
    S4_SENSOR_BIAS = "S4_sensor_bias"
    S5_RANDOM_FAULTS = "S5_random_faults"


class ProductionMode(Enum):
    """Production grade modes"""
    GRADE_A = "grade_a"
    GRADE_B = "grade_b"
    GRADE_C = "grade_c"


@dataclass
class TEPConfig:
    """Configuration parameters for TEP environment"""
    # Time parameters
    dt: float = 6.0  # Control time step (seconds)
    scheduling_dt: float = 300.0  # Scheduling time step (5 minutes)
    max_episode_steps: int = 4800  # 8 hours at 6s steps
    
    # Economic parameters
    product_value: float = 1000.0  # $/kmol product
    raw_material_cost: float = 500.0  # $/kmol raw material
    energy_cost: float = 0.1  # $/kJ energy
    off_spec_penalty: float = 2000.0  # $/hour off-spec
    
    # Process constraints
    max_reactor_temp: float = 175.0  # °C
    max_separator_pressure: float = 3000.0  # kPa
    max_stripper_level: float = 95.0  # %
    min_stripper_level: float = 5.0  # %
    
    # Noise parameters
    measurement_noise_std: float = 0.01
    process_noise_std: float = 0.005
    
    # Safety parameters
    enable_safety_constraints: bool = True
    constraint_violation_penalty: float = 10000.0


class TEPEnvironment(gym.Env):
    """
    Tennessee Eastman Process Environment
    
    This environment simulates the Tennessee Eastman Process with realistic
    dynamics, disturbances, and economic objectives.
    """
    
    def __init__(self, 
                 scenario: ScenarioType = ScenarioType.S1_BASIC_CHANGES,
                 config: Optional[TEPConfig] = None):
        super().__init__()
        
        self.config = config or TEPConfig()
        self.scenario = scenario
        
        # Initialize process state
        self._init_process_variables()
        
        # Define observation and action spaces
        self._init_spaces()
        
        # Initialize scenario-specific parameters
        self._init_scenario()
        
        # Logging and metrics
        self.episode_data = []
        self.current_step = 0
        self.episode_count = 0
        
        logger.info(f"TEP Environment initialized with scenario {scenario}")
        
        # Initialize scenario configuration
        self._init_scenario()
        
        # Initialize process variables
        self._init_process_variables()
    
    def _init_process_variables(self):
        """Initialize process variables"""
        # Process measurements (41 variables)
        # Based on standard TEP variable definitions
        self.measurement_names = [
            'A_feed_flow', 'D_feed_flow', 'E_feed_flow', 'A_C_feed_flow',
            'Recycle_flow', 'Reactor_feed_rate', 'Reactor_pressure', 'Reactor_level',
            'Reactor_temperature', 'Purge_rate', 'Product_sep_temp', 'Product_sep_level',
            'Product_sep_pressure', 'Product_sep_underflow', 'Stripper_level', 'Stripper_pressure',
            'Stripper_underflow', 'Stripper_temperature', 'Stripper_steam_flow', 'Compressor_work',
            'Reactor_cooling_temp', 'Separator_cooling_temp', 'A_composition', 'B_composition',
            'C_composition', 'D_composition', 'E_composition', 'F_composition', 'A_purge_comp',
            'B_purge_comp', 'C_purge_comp', 'D_purge_comp', 'E_purge_comp', 'F_purge_comp',
            'G_purge_comp', 'H_purge_comp', 'D_product_comp', 'E_product_comp', 'F_product_comp',
            'G_product_comp', 'H_product_comp'
        ]
        
        # Manipulated variables (12 variables)
        self.manipulated_names = [
            'D_feed_flow_valve', 'E_feed_flow_valve', 'A_feed_flow_valve', 'A_C_feed_flow_valve',
            'Compressor_recycle_valve', 'Purge_valve', 'Separator_pot_liquid_valve',
            'Stripper_liquid_product_valve', 'Stripper_steam_valve', 'Reactor_cooling_valve',
            'Condenser_cooling_valve', 'Agitator_speed'
        ]
        
        # Initialize state variables
        self._init_process_state()
        
        # Initialize economic tracking
        self.total_economic_cost = 0.0
        self.total_off_spec_time = 0.0
        self.constraint_violations = 0
        
        # Production tracking
        self.current_production_mode = ProductionMode.GRADE_A
        self.production_targets = {
            ProductionMode.GRADE_A: {'D_product': 0.9, 'E_product': 0.1},
            ProductionMode.GRADE_B: {'D_product': 0.7, 'E_product': 0.3},
            ProductionMode.GRADE_C: {'D_product': 0.5, 'E_product': 0.5}
        }
    
    def _init_process_state(self):
        """Initialize process state variables to steady-state values"""
        # Steady-state values for TEP (from literature)
        self.measurements = np.array([
            0.25052, 4.0975, 9.3477, 22.949, 18.776, 50.338, 2705.0, 75.0,
            120.4, 0.33712, 80.109, 50.0, 2633.7, 25.16, 50.0, 3102.2,
            22.949, 65.731, 230.31, 341.43, 94.599, 77.297, 32.188, 13.823,
            24.644, 18.776, 8.4036, 1.5699, 26.902, 4.5301, 7.2996, 51.595,
            11.859, 1.7677, 0.31827, 0.01787, 53.722, 43.827, 1.8968, 0.42073, 0.024297
        ])
        
        # Manipulated variables (steady-state setpoints)
        self.manipulated_vars = np.array([
            63.053, 53.980, 24.644, 61.302, 22.210, 40.064, 38.100,
            46.534, 47.446, 41.106, 18.114, 50.0
        ])
        
        # Internal state variables for dynamics
        self.reactor_holdup = 1000.0  # kmol
        self.separator_holdup = 500.0  # kmol
        self.stripper_holdup = 200.0  # kmol
        
        # Disturbance variables
        self.disturbances = np.zeros(20)  # 20 possible disturbances
        
        # Time tracking
        self.process_time = 0.0
        
    def _init_spaces(self):
        """Initialize observation and action spaces"""
        # Observation space: measurements + some internal states
        obs_dim = len(self.measurement_names) + 10  # Extra for internal states
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: continuous adjustments to manipulated variables
        # Actions are residual adjustments to baseline control
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(self.manipulated_names),), dtype=np.float32
        )
        
        # Scheduler action space (discrete)
        self.scheduler_action_space = spaces.MultiDiscrete([3, 3, 3, 3])  # mode, quality, throughput, horizon
        
    def _init_scenario(self):
        """Initialize scenario-specific parameters"""
        self.scenario_config = {
            ScenarioType.S1_BASIC_CHANGES: {
                'grade_changes': [(1200, ProductionMode.GRADE_B), (2400, ProductionMode.GRADE_C), (3600, ProductionMode.GRADE_A)],
                'disturbances': [],
                'faults': []
            },
            ScenarioType.S2_FEED_DRIFT: {
                'grade_changes': [(1000, ProductionMode.GRADE_B), (2000, ProductionMode.GRADE_C), (3000, ProductionMode.GRADE_A), (4000, ProductionMode.GRADE_B)],
                'disturbances': [('feed_composition_drift', 0, 4800)],
                'faults': []
            },
            ScenarioType.S3_COOLING_DISTURBANCE: {
                'grade_changes': [(1200, ProductionMode.GRADE_B), (2400, ProductionMode.GRADE_C), (3600, ProductionMode.GRADE_A)],
                'disturbances': [('reactor_cooling_loss', 1500, 2000)],
                'faults': []
            },
            ScenarioType.S4_SENSOR_BIAS: {
                'grade_changes': [(1200, ProductionMode.GRADE_B), (2400, ProductionMode.GRADE_C)],
                'disturbances': [('composition_sensor_bias', 800, 4800)],
                'faults': []
            },
            ScenarioType.S5_RANDOM_FAULTS: {
                'grade_changes': [(1000, ProductionMode.GRADE_B), (2500, ProductionMode.GRADE_C), (4000, ProductionMode.GRADE_A)],
                'disturbances': [],
                'faults': [('random_fault_1', 1800, 2200), ('random_fault_2', 3200, 3800)]
            }
        }
        
        # Add string mapping for convenience
        scenario_mapping = {
            'S1': ScenarioType.S1_BASIC_CHANGES,
            'S2': ScenarioType.S2_FEED_DRIFT,
            'S3': ScenarioType.S3_COOLING_DISTURBANCE,
            'S4': ScenarioType.S4_SENSOR_BIAS,
            'S5': ScenarioType.S5_RANDOM_FAULTS
        }
        
        # Convert string scenario to enum if needed
        if isinstance(self.scenario, str):
            self.scenario = scenario_mapping.get(self.scenario, ScenarioType.S1_BASIC_CHANGES)
        
        self.current_scenario_config = self.scenario_config[self.scenario]
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset process state
        self._init_process_state()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_data = []
        self.total_off_spec_time = 0.0
        
        # Initialize random variations if specified
        if options and options.get('randomize', False):
            self._apply_initial_randomization()
        
        # Log episode start
        self.episode_count += 1
        logger.info(f"Environment reset for episode {self.episode_count}")
        
        # Return initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment"""
        
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action to manipulated variables (residual control)
        self.manipulated_vars += action * 0.1  # Scale factor for stability
        self.manipulated_vars = np.clip(self.manipulated_vars, 0.0, 100.0)
        
        # Update process dynamics
        self._update_process_dynamics()
        
        # Apply scenario-specific disturbances
        self._apply_scenario_disturbances()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_episode_steps
        
        # Update step counter
        self.current_step += 1
        self.process_time += self.config.dt
        
        # Store episode data
        self.episode_data.append({
            'step': self.current_step,
            'time': self.process_time,
            'measurements': self.measurements.copy(),
            'manipulated_vars': self.manipulated_vars.copy(),
            'reward': reward,
            'economic_cost': self._calculate_economic_cost()
        })
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _update_process_dynamics(self):
        """Update process dynamics based on current state and actions"""
        
        # Simplified TEP dynamics (linearized around operating point)
        # In practice, this would use detailed process models
        
        dt = self.config.dt
        
        # Reactor dynamics
        reactor_temp_change = (
            0.1 * (self.manipulated_vars[9] - 50.0) +  # Cooling effect
            0.05 * (self.manipulated_vars[0] - 50.0) +  # Feed effect
            np.random.normal(0, self.config.process_noise_std)
        )
        self.measurements[8] += reactor_temp_change * dt / 60.0  # Convert to per-second
        
        # Separator dynamics
        separator_level_change = (
            0.2 * (self.manipulated_vars[6] - 50.0) +  # Liquid valve effect
            0.1 * (self.measurements[5] - 50.0) +  # Feed rate effect
            np.random.normal(0, self.config.process_noise_std)
        )
        self.measurements[11] += separator_level_change * dt / 60.0
        
        # Stripper dynamics
        stripper_level_change = (
            0.3 * (self.manipulated_vars[7] - 50.0) +  # Product valve effect
            0.15 * (self.manipulated_vars[8] - 50.0) +  # Steam effect
            np.random.normal(0, self.config.process_noise_std)
        )
        self.measurements[14] += stripper_level_change * dt / 60.0
        
        # Composition dynamics (simplified)
        for i in range(22, 28):  # Product compositions
            comp_change = np.random.normal(0, self.config.process_noise_std * 0.1)
            self.measurements[i] += comp_change * dt / 60.0
            self.measurements[i] = np.clip(self.measurements[i], 0.0, 100.0)
        
        # Apply measurement noise
        noise = np.random.normal(0, self.config.measurement_noise_std, len(self.measurements))
        self.measurements += noise
        
        # Ensure physical constraints
        self.measurements[8] = np.clip(self.measurements[8], 100.0, 200.0)  # Reactor temp
        self.measurements[11] = np.clip(self.measurements[11], 0.0, 100.0)  # Separator level
        self.measurements[14] = np.clip(self.measurements[14], 0.0, 100.0)  # Stripper level
        self.measurements[12] = np.clip(self.measurements[12], 2000.0, 4000.0)  # Separator pressure
    
    def _apply_scenario_disturbances(self):
        """Apply scenario-specific disturbances and faults"""
        
        current_time_minutes = self.process_time / 60.0
        
        # Apply disturbances based on scenario
        for disturbance_name, start_time, end_time in self.current_scenario_config['disturbances']:
            if start_time <= current_time_minutes <= end_time:
                self._apply_disturbance(disturbance_name)
        
        # Apply faults based on scenario
        for fault_name, start_time, end_time in self.current_scenario_config['faults']:
            if start_time <= current_time_minutes <= end_time:
                self._apply_fault(fault_name)
        
        # Apply grade changes
        for change_time, new_mode in self.current_scenario_config['grade_changes']:
            if abs(current_time_minutes - change_time) < 1.0:  # Within 1 minute
                self.current_production_mode = new_mode
                logger.info(f"Production mode changed to {new_mode.value} at time {current_time_minutes:.1f} min")
    
    def _apply_disturbance(self, disturbance_name: str):
        """Apply specific disturbance to the process"""
        
        if disturbance_name == 'feed_composition_drift':
            # Gradual drift in feed composition
            drift_rate = 0.001  # %/min
            self.measurements[22] += drift_rate * self.config.dt / 60.0
            
        elif disturbance_name == 'reactor_cooling_loss':
            # Temporary loss of cooling effectiveness
            self.measurements[8] += 2.0 * self.config.dt / 60.0  # Temperature rise
            
        elif disturbance_name == 'composition_sensor_bias':
            # Sensor bias in composition measurement
            self.measurements[25] += 1.0  # Constant bias
    
    def _apply_fault(self, fault_name: str):
        """Apply specific fault to the process"""
        
        if fault_name == 'random_fault_1':
            # Random valve sticking
            stuck_valve = np.random.randint(0, len(self.manipulated_vars))
            self.manipulated_vars[stuck_valve] = 50.0  # Stuck at 50%
            
        elif fault_name == 'random_fault_2':
            # Random sensor failure
            failed_sensor = np.random.randint(0, len(self.measurements))
            self.measurements[failed_sensor] = np.nan  # Sensor failure
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on economic performance and constraints"""
        
        # Economic cost component
        economic_cost = self._calculate_economic_cost()
        
        # Constraint violation penalty
        constraint_penalty = self._calculate_constraint_penalty()
        
        # Off-specification penalty
        off_spec_penalty = self._calculate_off_spec_penalty()
        
        # Total reward (negative cost)
        reward = -(economic_cost + constraint_penalty + off_spec_penalty)
        
        # Update totals
        self.total_economic_cost += economic_cost
        if off_spec_penalty > 0:
            self.total_off_spec_time += self.config.dt / 3600.0  # Convert to hours
        
        return reward
    
    def _calculate_economic_cost(self) -> float:
        """Calculate economic cost for current time step"""
        
        # Production rate (simplified)
        production_rate = self.measurements[13]  # Product separator underflow
        
        # Raw material consumption
        raw_material_rate = sum(self.measurements[0:4])  # Feed flows
        
        # Energy consumption
        energy_rate = (
            self.measurements[19] +  # Compressor work
            self.measurements[18] +  # Steam flow
            abs(self.measurements[20] - 80.0)  # Cooling deviation
        )
        
        # Calculate costs (per time step)
        production_value = production_rate * self.config.product_value * self.config.dt / 3600.0
        raw_material_cost = raw_material_rate * self.config.raw_material_cost * self.config.dt / 3600.0
        energy_cost = energy_rate * self.config.energy_cost * self.config.dt / 3600.0
        
        # Net cost (negative profit)
        net_cost = raw_material_cost + energy_cost - production_value
        
        return net_cost
    
    def _calculate_constraint_penalty(self) -> float:
        """Calculate penalty for constraint violations"""
        
        penalty = 0.0
        
        if self.config.enable_safety_constraints:
            # Reactor temperature constraint
            if self.measurements[8] > self.config.max_reactor_temp:
                penalty += self.config.constraint_violation_penalty
                self.constraint_violations += 1
            
            # Separator pressure constraint
            if self.measurements[12] > self.config.max_separator_pressure:
                penalty += self.config.constraint_violation_penalty
                self.constraint_violations += 1
            
            # Stripper level constraints
            if (self.measurements[14] > self.config.max_stripper_level or 
                self.measurements[14] < self.config.min_stripper_level):
                penalty += self.config.constraint_violation_penalty
                self.constraint_violations += 1
        
        return penalty
    
    def _calculate_off_spec_penalty(self) -> float:
        """Calculate penalty for off-specification production"""
        
        # Get current production targets
        targets = self.production_targets[self.current_production_mode]
        
        # Calculate composition deviations
        d_product_actual = self.measurements[36]  # D product composition
        e_product_actual = self.measurements[37]  # E product composition
        
        d_target = targets['D_product'] * 100.0
        e_target = targets['E_product'] * 100.0
        
        # Calculate off-spec penalty
        d_deviation = abs(d_product_actual - d_target)
        e_deviation = abs(e_product_actual - e_target)
        
        # Penalty if deviation exceeds tolerance
        tolerance = 5.0  # 5% tolerance
        penalty = 0.0
        
        if d_deviation > tolerance or e_deviation > tolerance:
            penalty = self.config.off_spec_penalty * self.config.dt / 3600.0
        
        return penalty
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to safety violations"""
        
        # Terminate if critical safety limits are exceeded
        if self.measurements[8] > 200.0:  # Critical reactor temperature
            logger.warning("Episode terminated: Critical reactor temperature exceeded")
            return True
        
        if self.measurements[12] > 4000.0:  # Critical separator pressure
            logger.warning("Episode terminated: Critical separator pressure exceeded")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        
        # Combine measurements with additional state information
        additional_state = np.array([
            self.process_time / 3600.0,  # Time in hours
            self.current_production_mode.value == ProductionMode.GRADE_A.value,
            self.current_production_mode.value == ProductionMode.GRADE_B.value,
            self.current_production_mode.value == ProductionMode.GRADE_C.value,
            self.total_economic_cost / 10000.0,  # Normalized economic cost
            self.total_off_spec_time,
            self.constraint_violations,
            np.mean(self.manipulated_vars),  # Average manipulated variable
            np.std(self.manipulated_vars),   # Manipulated variable variance
            len(self.episode_data)  # Episode progress
        ])
        
        obs = np.concatenate([self.measurements, additional_state])
        
        # Handle NaN values (sensor failures)
        obs = np.nan_to_num(obs, nan=0.0)
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        
        return {
            'process_time': self.process_time,
            'production_mode': self.current_production_mode.value,
            'economic_cost': self.total_economic_cost,
            'off_spec_time': self.total_off_spec_time,
            'constraint_violations': self.constraint_violations,
            'measurements': self.measurements.copy(),
            'manipulated_vars': self.manipulated_vars.copy(),
            'scenario': self.scenario.value
        }
    
    def _apply_initial_randomization(self):
        """Apply initial randomization to process state"""
        
        # Add random variations to initial conditions
        measurement_variation = np.random.normal(0, 0.05, len(self.measurements))
        self.measurements += measurement_variation * self.measurements
        
        manipulated_variation = np.random.normal(0, 0.02, len(self.manipulated_vars))
        self.manipulated_vars += manipulated_variation * self.manipulated_vars
        
        # Ensure constraints are still satisfied
        self.measurements = np.clip(self.measurements, 0.0, 1000.0)
        self.manipulated_vars = np.clip(self.manipulated_vars, 0.0, 100.0)
    
    def get_episode_data(self) -> List[Dict]:
        """Get complete episode data for analysis"""
        return self.episode_data.copy()
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for current episode"""
        
        if not self.episode_data:
            return {}
        
        rewards = [d['reward'] for d in self.episode_data]
        economic_costs = [d['economic_cost'] for d in self.episode_data]
        
        return {
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'total_economic_cost': sum(economic_costs),
            'average_economic_cost': np.mean(economic_costs),
            'total_off_spec_time': self.total_off_spec_time,
            'constraint_violations': self.constraint_violations,
            'episode_length': len(self.episode_data),
            'final_production_mode': self.current_production_mode.value
        }


# Test the environment
if __name__ == "__main__":
    # Create environment
    env = TEPEnvironment(scenario=ScenarioType.S1_BASIC_CHANGES)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            break
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("TEP Environment test completed successfully!")

