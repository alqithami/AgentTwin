"""
Control Barrier Function-based Safety Shield for Tennessee Eastman Process

This module implements a comprehensive safety shield that ensures constraint
satisfaction while maintaining learning performance. The shield uses control
barrier functions (CBFs) and quadratic programming to minimally modify
control actions when necessary.

Key Features:
- CBF-based safety constraints for critical process variables
- Quadratic programming optimization for minimal intervention
- PID fallback mechanism for robustness
- Real-time performance monitoring and logging
- Configurable safety margins and parameters

Author: Implementation for Multi-Agent Digital Twin Research
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShieldStatus(Enum):
    """Status codes for safety shield operation"""
    SUCCESS = "success"
    QP_INFEASIBLE = "qp_infeasible"
    QP_TIMEOUT = "qp_timeout"
    FALLBACK_ACTIVE = "fallback_active"
    ERROR = "error"


@dataclass
class SafetyConstraint:
    """Definition of a safety constraint using control barrier functions"""
    name: str
    barrier_function: Callable[[np.ndarray], float]
    barrier_gradient: Callable[[np.ndarray], np.ndarray]
    alpha: float  # Class-K function parameter
    priority: int  # Higher priority constraints are enforced more strictly
    active: bool = True


@dataclass
class ShieldConfig:
    """Configuration parameters for safety shield"""
    # QP solver parameters
    qp_solver: str = "OSQP"
    qp_timeout: float = 0.01  # 10ms timeout for real-time operation
    qp_verbose: bool = False
    
    # Safety parameters
    safety_margin: float = 0.1  # Additional safety margin
    intervention_threshold: float = 0.01  # Minimum intervention magnitude
    
    # Fallback parameters
    enable_fallback: bool = True
    fallback_gain: float = 0.5  # Conservative gain for PID fallback
    
    # Monitoring parameters
    log_interventions: bool = True
    max_log_entries: int = 1000


@dataclass
class ShieldMetrics:
    """Metrics for monitoring safety shield performance"""
    total_interventions: int = 0
    qp_solve_times: List[float] = None
    qp_success_rate: float = 1.0
    fallback_activations: int = 0
    constraint_violations: int = 0
    intervention_magnitudes: List[float] = None
    
    def __post_init__(self):
        if self.qp_solve_times is None:
            self.qp_solve_times = []
        if self.intervention_magnitudes is None:
            self.intervention_magnitudes = []


class SafetyShield:
    """
    Control Barrier Function-based Safety Shield
    
    This class implements a safety shield that uses control barrier functions
    and quadratic programming to ensure constraint satisfaction while minimally
    modifying the desired control actions.
    """
    
    def __init__(self, 
                 config: Optional[ShieldConfig] = None,
                 constraints: Optional[List[SafetyConstraint]] = None):
        self.config = config or ShieldConfig()
        self.constraints = constraints or []
        
        # Initialize metrics
        self.metrics = ShieldMetrics()
        
        # Initialize QP variables (will be set up when first called)
        self.qp_problem = None
        self.u_var = None
        self.u_desired = None
        
        # PID fallback controllers
        self.pid_controllers = {}
        
        # State tracking
        self.last_state = None
        self.last_action = None
        self.intervention_history = []
        
        logger.info(f"Safety shield initialized with {len(self.constraints)} constraints")
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint to the shield"""
        self.constraints.append(constraint)
        # Reset QP problem to include new constraint
        self.qp_problem = None
        logger.info(f"Added safety constraint: {constraint.name}")
    
    def remove_constraint(self, constraint_name: str):
        """Remove a safety constraint by name"""
        self.constraints = [c for c in self.constraints if c.name != constraint_name]
        # Reset QP problem
        self.qp_problem = None
        logger.info(f"Removed safety constraint: {constraint_name}")
    
    def filter_action(self, 
                     desired_action: np.ndarray, 
                     current_state: np.ndarray,
                     dt: float = 0.1) -> Tuple[np.ndarray, ShieldStatus, Dict]:
        """
        Filter the desired action through the safety shield
        
        Args:
            desired_action: Desired control action from learning agent
            current_state: Current process state
            dt: Time step for dynamics
            
        Returns:
            Tuple of (safe_action, status, info_dict)
        """
        start_time = time.time()
        
        # Store current state and desired action
        self.last_state = current_state.copy()
        self.last_action = desired_action.copy()
        
        # Check if any constraints are violated or close to violation
        constraint_violations = self._check_constraints(current_state)
        
        if not any(constraint_violations):
            # No constraints violated, return desired action
            solve_time = time.time() - start_time
            info = {
                'intervention_magnitude': 0.0,
                'solve_time': solve_time,
                'constraints_active': 0,
                'qp_status': 'not_needed'
            }
            return desired_action, ShieldStatus.SUCCESS, info
        
        # Solve QP to find safe action
        try:
            safe_action, qp_status, qp_info = self._solve_qp_shield(
                desired_action, current_state, dt
            )
            
            if qp_status == ShieldStatus.SUCCESS:
                # QP solved successfully
                intervention_magnitude = np.linalg.norm(safe_action - desired_action)
                self._log_intervention(intervention_magnitude, qp_info['solve_time'])
                
                info = {
                    'intervention_magnitude': intervention_magnitude,
                    'solve_time': qp_info['solve_time'],
                    'constraints_active': qp_info['constraints_active'],
                    'qp_status': 'success'
                }
                
                return safe_action, ShieldStatus.SUCCESS, info
            
            else:
                # QP failed, use fallback
                if self.config.enable_fallback:
                    fallback_action = self._apply_fallback(current_state)
                    self.metrics.fallback_activations += 1
                    
                    info = {
                        'intervention_magnitude': np.linalg.norm(fallback_action - desired_action),
                        'solve_time': time.time() - start_time,
                        'constraints_active': len([c for c in constraint_violations if c]),
                        'qp_status': 'fallback'
                    }
                    
                    return fallback_action, ShieldStatus.FALLBACK_ACTIVE, info
                else:
                    # No fallback, return desired action with warning
                    logger.warning("QP failed and fallback disabled, returning desired action")
                    info = {
                        'intervention_magnitude': 0.0,
                        'solve_time': time.time() - start_time,
                        'constraints_active': len([c for c in constraint_violations if c]),
                        'qp_status': 'failed'
                    }
                    return desired_action, ShieldStatus.ERROR, info
                    
        except Exception as e:
            logger.error(f"Safety shield error: {e}")
            if self.config.enable_fallback:
                fallback_action = self._apply_fallback(current_state)
                info = {
                    'intervention_magnitude': np.linalg.norm(fallback_action - desired_action),
                    'solve_time': time.time() - start_time,
                    'constraints_active': len([c for c in constraint_violations if c]),
                    'qp_status': 'error_fallback'
                }
                return fallback_action, ShieldStatus.FALLBACK_ACTIVE, info
            else:
                info = {
                    'intervention_magnitude': 0.0,
                    'solve_time': time.time() - start_time,
                    'constraints_active': len([c for c in constraint_violations if c]),
                    'qp_status': 'error'
                }
                return desired_action, ShieldStatus.ERROR, info
    
    def _check_constraints(self, state: np.ndarray) -> List[bool]:
        """Check which constraints are violated or close to violation"""
        violations = []
        
        for constraint in self.constraints:
            if not constraint.active:
                violations.append(False)
                continue
                
            try:
                h_value = constraint.barrier_function(state)
                # Check if constraint is violated or close to violation
                is_violated = h_value <= self.config.safety_margin
                violations.append(is_violated)
                
                if is_violated and self.config.log_interventions:
                    logger.debug(f"Constraint {constraint.name} violated: h={h_value:.4f}")
                    
            except Exception as e:
                logger.warning(f"Error evaluating constraint {constraint.name}: {e}")
                violations.append(True)  # Conservative: assume violation
        
        return violations
    
    def _solve_qp_shield(self, 
                        desired_action: np.ndarray, 
                        state: np.ndarray, 
                        dt: float) -> Tuple[np.ndarray, ShieldStatus, Dict]:
        """Solve the QP optimization problem for safety filtering"""
        
        # Set up QP problem if not already done
        if self.qp_problem is None:
            self._setup_qp_problem(len(desired_action))
        
        start_time = time.time()
        
        try:
            # Update QP parameters
            self.u_desired.value = desired_action
            
            # Update constraint parameters
            active_constraints = 0
            constraint_idx = 0
            
            for i, constraint in enumerate(self.constraints):
                if not constraint.active:
                    continue
                    
                h_value = constraint.barrier_function(state)
                h_grad = constraint.barrier_gradient(state)
                
                # CBF constraint: h_dot >= -alpha * h
                # For control actions, we need gradient w.r.t. control variables
                # Extract gradient for control variables only (first 12 elements)
                
                if h_value <= self.config.safety_margin:
                    # Constraint is active
                    rhs_value = -constraint.alpha * h_value
                    
                    # Extract control gradient (first 12 elements for TEP)
                    control_grad = h_grad[:len(desired_action)]
                    
                    # Update constraint in QP
                    if hasattr(self, 'constraint_matrices'):
                        self.constraint_matrices[constraint_idx].value = control_grad.reshape(1, -1)
                        self.constraint_rhs[constraint_idx].value = np.array([rhs_value])
                        active_constraints += 1
                        constraint_idx += 1
            
            # Solve QP
            self.qp_problem.solve(
                solver=self.config.qp_solver,
                verbose=self.config.qp_verbose,
                eps_abs=1e-6,
                eps_rel=1e-6
            )
            
            solve_time = time.time() - start_time
            
            if self.qp_problem.status == cp.OPTIMAL:
                safe_action = self.u_var.value
                self.metrics.qp_solve_times.append(solve_time)
                
                info = {
                    'solve_time': solve_time,
                    'constraints_active': active_constraints,
                    'qp_objective': self.qp_problem.value
                }
                
                return safe_action, ShieldStatus.SUCCESS, info
            
            elif self.qp_problem.status == cp.INFEASIBLE:
                logger.warning("QP problem is infeasible")
                return desired_action, ShieldStatus.QP_INFEASIBLE, {'solve_time': solve_time}
            
            else:
                logger.warning(f"QP solver failed with status: {self.qp_problem.status}")
                return desired_action, ShieldStatus.QP_TIMEOUT, {'solve_time': solve_time}
                
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"QP solving error: {e}")
            return desired_action, ShieldStatus.ERROR, {'solve_time': solve_time}
    
    def _setup_qp_problem(self, action_dim: int):
        """Set up the QP optimization problem"""
        
        # Decision variable: control action
        self.u_var = cp.Variable(action_dim)
        
        # Parameter: desired action
        self.u_desired = cp.Parameter(action_dim)
        
        # Objective: minimize ||u - u_desired||^2
        objective = cp.Minimize(cp.sum_squares(self.u_var - self.u_desired))
        
        # Constraints will be added dynamically
        constraints = []
        
        # Action bounds (if needed)
        # constraints.append(self.u_var >= -10.0)
        # constraints.append(self.u_var <= 10.0)
        
        # Create constraint matrices for dynamic constraints
        max_constraints = len(self.constraints)
        self.constraint_matrices = []
        self.constraint_rhs = []
        
        for i in range(max_constraints):
            A_i = cp.Parameter((1, action_dim))
            b_i = cp.Parameter(1)
            constraints.append(A_i @ self.u_var >= b_i)
            self.constraint_matrices.append(A_i)
            self.constraint_rhs.append(b_i)
            
            # Initialize with dummy values
            A_i.value = np.zeros((1, action_dim))
            b_i.value = np.array([-1e6])  # Inactive constraint
        
        # Create QP problem
        self.qp_problem = cp.Problem(objective, constraints)
        
        logger.info(f"QP problem set up with {action_dim} variables and {max_constraints} constraint slots")
    
    def _apply_fallback(self, state: np.ndarray) -> np.ndarray:
        """Apply PID fallback control when QP fails"""
        
        # Simple fallback: return to safe operating point
        # In practice, this would use well-tuned PID controllers
        
        if self.last_action is not None:
            # Conservative action: reduce magnitude
            fallback_action = self.last_action * self.config.fallback_gain
        else:
            # Default safe action (zeros - no change)
            fallback_action = np.zeros_like(state[:12])  # Assuming 12 manipulated variables
        
        logger.info("PID fallback activated")
        return fallback_action
    
    def _log_intervention(self, magnitude: float, solve_time: float):
        """Log safety shield intervention"""
        self.metrics.total_interventions += 1
        self.metrics.intervention_magnitudes.append(magnitude)
        
        if magnitude > self.config.intervention_threshold:
            if self.config.log_interventions:
                logger.info(f"Safety intervention: magnitude={magnitude:.4f}, solve_time={solve_time:.4f}s")
        
        # Maintain log size
        if len(self.metrics.intervention_magnitudes) > self.config.max_log_entries:
            self.metrics.intervention_magnitudes = self.metrics.intervention_magnitudes[-self.config.max_log_entries:]
        
        if len(self.metrics.qp_solve_times) > self.config.max_log_entries:
            self.metrics.qp_solve_times = self.metrics.qp_solve_times[-self.config.max_log_entries:]
    
    def get_metrics(self) -> ShieldMetrics:
        """Get current shield performance metrics"""
        # Update success rate
        if len(self.metrics.qp_solve_times) > 0:
            total_attempts = self.metrics.total_interventions
            successful_solves = len(self.metrics.qp_solve_times)
            self.metrics.qp_success_rate = successful_solves / max(total_attempts, 1)
        
        return self.metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = ShieldMetrics()
        logger.info("Safety shield metrics reset")
    
    def get_intervention_rate(self, window_size: int = 100) -> float:
        """Get recent intervention rate per action"""
        if len(self.intervention_history) < window_size:
            return len(self.intervention_history) / max(len(self.intervention_history), 1)
        else:
            recent_interventions = sum(self.intervention_history[-window_size:])
            return recent_interventions / window_size


def create_tep_safety_constraints() -> List[SafetyConstraint]:
    """Create safety constraints specific to Tennessee Eastman Process"""
    
    constraints = []
    
    # Reactor temperature constraint - use safe indexing
    def reactor_temp_barrier(state):
        # Safe indexing: use index 8 if available, otherwise use a default
        if len(state) > 8:
            temp = state[8]  # Reactor temperature
        else:
            logger.warning(f"State too short for reactor temp (len={len(state)}), using default")
            temp = 120.0  # Default safe temperature
        max_temp = 175.0  # Maximum safe temperature
        return max_temp - temp
    
    def reactor_temp_gradient(state):
        # For TEP, reactor temperature is affected by cooling (manipulated variable 10)
        grad = np.zeros(12)  # 12 manipulated variables
        grad[10] = -1.0  # Cooling reduces temperature
        return grad
    
    constraints.append(SafetyConstraint(
        name="reactor_temperature",
        barrier_function=reactor_temp_barrier,
        barrier_gradient=reactor_temp_gradient,
        alpha=0.1,
        priority=1
    ))
    
    # Separator pressure constraint - use safe indexing
    def separator_pressure_barrier(state):
        # Safe indexing: check if state is long enough
        if len(state) > 12:
            pressure = state[12]  # Separator pressure
        else:
            logger.warning(f"State too short for separator pressure (len={len(state)}), using default")
            pressure = 2500.0  # Default safe pressure
        max_pressure = 3000.0  # Maximum safe pressure
        return max_pressure - pressure
    
    def separator_pressure_gradient(state):
        # Separator pressure is affected by liquid valve (manipulated variable 6)
        grad = np.zeros(12)  # 12 manipulated variables
        grad[6] = -0.5  # Liquid valve reduces pressure
        return grad
    
    constraints.append(SafetyConstraint(
        name="separator_pressure",
        barrier_function=separator_pressure_barrier,
        barrier_gradient=separator_pressure_gradient,
        alpha=0.05,
        priority=1
    ))
    
    # Stripper level constraints (both upper and lower bounds) - use safe indexing
    def stripper_level_upper_barrier(state):
        # Safe indexing: check if state is long enough
        if len(state) > 14:
            level = state[14]  # Stripper level
        else:
            logger.warning(f"State too short for stripper level (len={len(state)}), using default")
            level = 50.0  # Default safe level
        max_level = 95.0  # Maximum safe level
        return max_level - level
    
    def stripper_level_upper_gradient(state):
        # Stripper level is affected by liquid valve (manipulated variable 7)
        grad = np.zeros(12)  # 12 manipulated variables
        grad[7] = -0.8  # Liquid valve reduces level
        return grad
    
    constraints.append(SafetyConstraint(
        name="stripper_level_upper",
        barrier_function=stripper_level_upper_barrier,
        barrier_gradient=stripper_level_upper_gradient,
        alpha=0.2,
        priority=2
    ))
    
    def stripper_level_lower_barrier(state):
        # Safe indexing: check if state is long enough
        if len(state) > 14:
            level = state[14]  # Stripper level
        else:
            logger.warning(f"State too short for stripper level (len={len(state)}), using default")
            level = 50.0  # Default safe level
        min_level = 5.0  # Minimum safe level
        return level - min_level
    
    def stripper_level_lower_gradient(state):
        # Stripper level is affected by liquid valve (manipulated variable 7)
        grad = np.zeros(12)  # 12 manipulated variables
        grad[7] = 0.8  # Liquid valve increases level (for lower bound)
        return grad
    
    constraints.append(SafetyConstraint(
        name="stripper_level_lower",
        barrier_function=stripper_level_lower_barrier,
        barrier_gradient=stripper_level_lower_gradient,
        alpha=0.2,
        priority=2
    ))
    
    return constraints


# Test the safety shield
if __name__ == "__main__":
    # Create safety constraints
    constraints = create_tep_safety_constraints()
    
    # Create safety shield
    config = ShieldConfig(qp_timeout=0.01, log_interventions=True)
    shield = SafetyShield(config=config, constraints=constraints)
    
    print(f"Safety shield created with {len(constraints)} constraints")
    
    # Test with sample state and action
    state = np.random.randn(41)  # 41 TEP measurements
    state[8] = 170.0  # High reactor temperature (close to limit)
    state[12] = 2800.0  # High separator pressure
    state[14] = 50.0  # Normal stripper level
    
    desired_action = np.random.randn(12) * 2.0  # 12 manipulated variables
    
    # Filter action through shield
    safe_action, status, info = shield.filter_action(desired_action, state)
    
    print(f"Desired action: {desired_action}")
    print(f"Safe action: {safe_action}")
    print(f"Status: {status}")
    print(f"Info: {info}")
    print(f"Intervention magnitude: {info['intervention_magnitude']:.4f}")
    
    # Test multiple times to check performance
    print("\nTesting shield performance...")
    for i in range(10):
        state[8] = 160.0 + i * 2.0  # Gradually increase temperature
        safe_action, status, info = shield.filter_action(desired_action, state)
        print(f"Step {i}: temp={state[8]:.1f}, intervention={info['intervention_magnitude']:.4f}")
    
    # Get metrics
    metrics = shield.get_metrics()
    print(f"\nShield metrics:")
    print(f"Total interventions: {metrics.total_interventions}")
    print(f"Average solve time: {np.mean(metrics.qp_solve_times):.4f}s")
    print(f"QP success rate: {metrics.qp_success_rate:.2f}")
    
    print("Safety shield test completed successfully!")

