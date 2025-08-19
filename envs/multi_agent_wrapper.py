"""
Multi-Agent Wrapper for Tennessee Eastman Process Environment

This module provides a wrapper that enables multi-agent operation
with separate scheduler and controller agents operating on different
time scales.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .tep_env import TEPEnvironment, TEPConfig, ProductionMode, ScenarioType


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_id: str
    agent_type: str  # 'scheduler' or 'controller'
    control_variables: List[int]  # Indices of controlled variables
    observation_variables: List[int]  # Indices of observed variables
    time_scale: str  # 'fast' or 'slow'


class MultiAgentTEPWrapper:
    """
    Multi-agent wrapper for Tennessee Eastman Process
    
    This wrapper enables multi-agent operation with:
    - One scheduler agent (slow time scale)
    - Multiple controller agents (fast time scale)
    - Coordination through intent vectors and readiness signals
    """
    
    def __init__(self, 
                 base_env: TEPEnvironment,
                 agent_configs: List[AgentConfig]):
        self.base_env = base_env
        self.agent_configs = {config.agent_id: config for config in agent_configs}
        
        # Time scale management
        self.fast_dt = base_env.config.dt  # Use dt instead of fast_dt
        self.slow_dt = base_env.config.scheduling_dt  # Use scheduling_dt instead of slow_dt
        self.steps_per_slow_action = int(self.slow_dt / self.fast_dt)  # 50 steps
        
        # Agent management
        self.scheduler_agents = [config for config in agent_configs if config.agent_type == 'scheduler']
        self.controller_agents = [config for config in agent_configs if config.agent_type == 'controller']
        
        # Coordination mechanisms
        self.intent_vector = np.zeros(4)  # [mode, quality, throughput, horizon]
        self.readiness_signals = {agent.agent_id: 0.5 for agent in self.controller_agents}
        
        # Step tracking
        self.fast_step_count = 0
        self.slow_step_count = 0
        
        # Initialize observation and action spaces
        self._init_agent_spaces()
        
        # Episode tracking
        self.episode_rewards = {agent.agent_id: 0.0 for agent in agent_configs}
        self.coordination_history = []
    
    def _init_agent_spaces(self):
        """Initialize observation and action spaces for each agent"""
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for agent_id, config in self.agent_configs.items():
            if config.agent_type == 'scheduler':
                # Scheduler: compact observation + readiness signals
                obs_dim = 15 + len(self.controller_agents)  # Compact state + readiness
                self.observation_spaces[agent_id] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                # Scheduler: discrete actions for mode, quality, throughput, horizon
                self.action_spaces[agent_id] = spaces.MultiDiscrete([3, 3, 3, 3])
                
            else:  # controller
                # Controller: subset of measurements + intent vector
                obs_dim = len(config.observation_variables) + 4  # Observations + intent
                self.observation_spaces[agent_id] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                # Controller: continuous actions for controlled variables
                action_dim = len(config.control_variables)
                self.action_spaces[agent_id] = spaces.Box(
                    low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32
                )
    
    def get_agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        return list(self.agent_configs.keys())
    
    def get_observation_space(self, agent_id: str) -> spaces.Space:
        """Get observation space for specific agent"""
        return self.observation_spaces[agent_id]
    
    def get_action_space(self, agent_id: str) -> spaces.Space:
        """Get action space for specific agent"""
        return self.action_spaces[agent_id]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset environment and return initial observations for all agents"""
        
        # Reset base environment
        base_obs, base_info = self.base_env.reset(seed=seed, options=options)
        
        # Reset coordination mechanisms
        self.intent_vector = np.array([1.0, 1.0, 1.0, 600.0])  # Default intent
        self.readiness_signals = {agent.agent_id: 0.5 for agent in self.controller_agents}
        
        # Reset step counters
        self.fast_step_count = 0
        self.slow_step_count = 0
        
        # Reset episode tracking
        self.episode_rewards = {agent.agent_id: 0.0 for agent in self.agent_configs.values()}
        self.coordination_history = []
        
        # Generate initial observations for all agents
        observations = self._get_agent_observations()
        
        # Generate info for all agents
        infos = {agent_id: base_info.copy() for agent_id in self.agent_configs.keys()}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """Execute actions for all agents"""
        
        # Determine which agents should act this step
        acting_agents = self._get_acting_agents()
        
        # Apply scheduler actions (if any)
        for agent_id in acting_agents:
            config = self.agent_configs[agent_id]
            if config.agent_type == 'scheduler' and agent_id in actions:
                self._apply_scheduler_action(agent_id, actions[agent_id])
        
        # Combine controller actions
        combined_action = self._combine_controller_actions(actions, acting_agents)
        
        # Execute action in base environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(combined_action)
        
        # Update readiness signals based on controller performance
        self._update_readiness_signals(actions, acting_agents, base_reward)
        
        # Generate observations for all agents
        observations = self._get_agent_observations()
        
        # Calculate individual rewards
        rewards = self._calculate_agent_rewards(base_reward, actions, acting_agents)
        
        # Determine termination for each agent
        terminated = {agent_id: base_terminated for agent_id in self.agent_configs.keys()}
        truncated = {agent_id: base_truncated for agent_id in self.agent_configs.keys()}
        
        # Generate info for all agents
        infos = self._generate_agent_infos(base_info, actions, acting_agents)
        
        # Update step counters
        self.fast_step_count += 1
        if self.fast_step_count % self.steps_per_slow_action == 0:
            self.slow_step_count += 1
        
        # Store coordination data
        self._store_coordination_data(actions, rewards)
        
        return observations, rewards, terminated, truncated, infos
    
    def _get_acting_agents(self) -> List[str]:
        """Determine which agents should act this step"""
        acting_agents = []
        
        # Controller agents act every fast step
        for agent in self.controller_agents:
            acting_agents.append(agent.agent_id)
        
        # Scheduler agents act every slow step
        if self.fast_step_count % self.steps_per_slow_action == 0:
            for agent in self.scheduler_agents:
                acting_agents.append(agent.agent_id)
        
        return acting_agents
    
    def _apply_scheduler_action(self, agent_id: str, action: Optional[np.ndarray]):
        """Apply scheduler action to update intent vector"""
        if action is None:
            return
        
        mode_idx, quality_idx, throughput_idx, horizon_idx = action
        
        # Update intent vector
        self.intent_vector[0] = float(mode_idx + 1)  # Production mode (1-3)
        self.intent_vector[1] = 0.9 + 0.1 * quality_idx  # Quality target (0.9-1.1)
        self.intent_vector[2] = 0.98 + 0.02 * throughput_idx  # Throughput target (0.98-1.02)
        self.intent_vector[3] = 300.0 + 300.0 * horizon_idx  # Horizon (300-900s)
        
        # Apply mode change to base environment if needed
        new_mode_value = int(self.intent_vector[0])
        if new_mode_value == 1:
            new_mode = ProductionMode.GRADE_A
        elif new_mode_value == 2:
            new_mode = ProductionMode.GRADE_B
        else:
            new_mode = ProductionMode.GRADE_C
            
        if new_mode != self.base_env.current_production_mode:
            self.base_env.current_production_mode = new_mode
    
    def _combine_controller_actions(self, actions: Dict[str, np.ndarray], acting_agents: List[str]) -> np.ndarray:
        """Combine controller actions into single action for base environment"""
        # Initialize with zeros (no change)
        combined_action = np.zeros(len(self.base_env.manipulated_names))
        
        # Apply each controller's action to its controlled variables
        for agent_id in acting_agents:
            config = self.agent_configs[agent_id]
            if config.agent_type == 'controller' and agent_id in actions:
                action = actions[agent_id]
                for i, var_idx in enumerate(config.control_variables):
                    if var_idx < len(combined_action):
                        combined_action[var_idx] = action[i] if i < len(action) else 0.0
        
        return combined_action
    
    def _get_agent_observations(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents"""
        observations = {}
        base_obs = self.base_env._get_observation()
        
        for agent_id, config in self.agent_configs.items():
            if config.agent_type == 'scheduler':
                # Scheduler observation: compact state + readiness signals
                scheduler_obs = self._get_scheduler_observation(base_obs)
                readiness_values = [self.readiness_signals[agent.agent_id] for agent in self.controller_agents]
                obs = np.concatenate([scheduler_obs, readiness_values])
                
            else:  # controller
                # Controller observation: subset of measurements + intent vector
                controller_obs = base_obs[config.observation_variables]
                obs = np.concatenate([controller_obs, self.intent_vector])
            
            observations[agent_id] = obs.astype(np.float32)
        
        return observations
    
    def _get_scheduler_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Generate compact observation for scheduler agent"""
        # Select key process variables for scheduling decisions
        key_indices = [
            8,   # Reactor temperature
            11,  # Separator level
            14,  # Stripper level
            12,  # Separator pressure
            36,  # D product composition
            37,  # E product composition
            13,  # Product flow rate
            5,   # Reactor feed rate
            19,  # Compressor work
            20,  # Reactor cooling temp
        ]
        
        scheduler_obs = base_obs[key_indices]
        
        # Add economic and coordination information
        economic_info = np.array([
            self.base_env.total_economic_cost / 10000.0,  # Normalized cost
            self.base_env.total_off_spec_time,
            self.base_env.constraint_violations,
            self.base_env.process_time / 3600.0,  # Time in hours
            float(self.base_env.current_production_mode.value == ProductionMode.GRADE_A.value)
        ])
        
        return np.concatenate([scheduler_obs, economic_info])
    
    def _update_readiness_signals(self, actions: Dict[str, np.ndarray], acting_agents: List[str], base_reward: float):
        """Update readiness signals based on controller performance"""
        
        for agent_id in acting_agents:
            config = self.agent_configs[agent_id]
            if config.agent_type == 'controller':
                # Simple readiness update based on reward and action magnitude
                if agent_id in actions:
                    action_magnitude = np.linalg.norm(actions[agent_id])
                    
                    # Higher readiness for good performance and moderate actions
                    readiness_change = 0.01 * (base_reward / 1000.0 - action_magnitude / 10.0)
                    self.readiness_signals[agent_id] += readiness_change
                    self.readiness_signals[agent_id] = np.clip(self.readiness_signals[agent_id], 0.0, 1.0)
    
    def _calculate_agent_rewards(self, base_reward: float, actions: Dict[str, np.ndarray], acting_agents: List[str]) -> Dict[str, float]:
        """Calculate individual rewards for each agent"""
        rewards = {}
        
        # Base reward shared among all agents
        shared_reward = base_reward * 0.7
        
        for agent_id, config in self.agent_configs.items():
            agent_reward = shared_reward / len(self.agent_configs)
            
            if config.agent_type == 'scheduler':
                # Scheduler gets bonus for good coordination
                avg_readiness = np.mean(list(self.readiness_signals.values()))
                coordination_bonus = 100.0 * (avg_readiness - 0.5)
                agent_reward += coordination_bonus
                
            else:  # controller
                # Controller gets bonus for readiness and penalty for large actions
                readiness_bonus = 50.0 * (self.readiness_signals[agent_id] - 0.5)
                
                action_penalty = 0.0
                if agent_id in actions and agent_id in acting_agents:
                    action_magnitude = np.linalg.norm(actions[agent_id])
                    action_penalty = -10.0 * action_magnitude
                
                agent_reward += readiness_bonus + action_penalty
            
            rewards[agent_id] = agent_reward
            self.episode_rewards[agent_id] += agent_reward
        
        return rewards
    
    def _generate_agent_infos(self, base_info: Dict, actions: Dict[str, np.ndarray], acting_agents: List[str]) -> Dict[str, Dict]:
        """Generate info dictionaries for each agent"""
        infos = {}
        
        for agent_id, config in self.agent_configs.items():
            info = base_info.copy()
            info.update({
                'agent_id': agent_id,
                'agent_type': config.agent_type,
                'acting_this_step': agent_id in acting_agents,
                'episode_reward': self.episode_rewards[agent_id],
                'intent_vector': self.intent_vector.copy(),
                'readiness_signal': self.readiness_signals.get(agent_id, 0.5),
                'fast_step': self.fast_step_count,
                'slow_step': self.slow_step_count
            })
            
            if agent_id in actions:
                info['action'] = actions[agent_id].copy()
            
            # Ensure info is always a proper dictionary
            if not isinstance(info, dict):
                info = {
                    'agent_id': agent_id,
                    'agent_type': config.agent_type,
                    'acting_this_step': agent_id in acting_agents
                }
            
            infos[agent_id] = info
        
        return infos
    
    def _store_coordination_data(self, actions: Dict[str, np.ndarray], rewards: Dict[str, float]):
        """Store coordination data for analysis"""
        
        coordination_data = {
            'step': self.fast_step_count,
            'intent_vector': self.intent_vector.copy(),
            'readiness_signals': self.readiness_signals.copy(),
            'avg_readiness': np.mean(list(self.readiness_signals.values())),
            'total_reward': sum(rewards.values()),
            'action_variance': 0.0
        }
        
        # Calculate action variance across controllers
        controller_actions = []
        for agent_id, action in actions.items():
            config = self.agent_configs[agent_id]
            if config.agent_type == 'controller':
                controller_actions.extend(action.tolist())
        
        if controller_actions:
            coordination_data['action_variance'] = np.var(controller_actions)
        
        self.coordination_history.append(coordination_data)
    
    def get_coordination_metrics(self) -> Dict:
        """Get coordination effectiveness metrics"""
        if not self.coordination_history:
            return {}
        
        readiness_values = [d['avg_readiness'] for d in self.coordination_history]
        action_variances = [d['action_variance'] for d in self.coordination_history]
        total_rewards = [d['total_reward'] for d in self.coordination_history]
        
        return {
            'avg_readiness': np.mean(readiness_values),
            'readiness_std': np.std(readiness_values),
            'avg_action_variance': np.mean(action_variances),
            'action_variance_std': np.std(action_variances),
            'avg_total_reward': np.mean(total_rewards),
            'total_reward_std': np.std(total_rewards),
            'coordination_steps': len(self.coordination_history)
        }


def create_default_agent_configs() -> List[AgentConfig]:
    """Create default agent configurations for TEP"""
    
    configs = [
        # Scheduler agent
        AgentConfig(
            agent_id="scheduler",
            agent_type="scheduler",
            control_variables=[],  # No direct control
            observation_variables=list(range(15)),  # Compact observation
            time_scale="slow"
        ),
        
        # Reactor controller
        AgentConfig(
            agent_id="reactor_controller",
            agent_type="controller",
            control_variables=[9, 11],  # Reactor cooling, agitator speed
            observation_variables=[6, 7, 8, 20, 21],  # Reactor-related measurements
            time_scale="fast"
        ),
        
        # Separator controller
        AgentConfig(
            agent_id="separator_controller",
            agent_type="controller",
            control_variables=[6, 10],  # Separator liquid valve, condenser cooling
            observation_variables=[10, 11, 12, 13],  # Separator-related measurements
            time_scale="fast"
        ),
        
        # Stripper controller
        AgentConfig(
            agent_id="stripper_controller",
            agent_type="controller",
            control_variables=[7, 8],  # Stripper product valve, steam valve
            observation_variables=[14, 15, 16, 17, 18],  # Stripper-related measurements
            time_scale="fast"
        ),
        
        # Feed controller
        AgentConfig(
            agent_id="feed_controller",
            agent_type="controller",
            control_variables=[0, 1, 2, 3, 4, 5],  # Feed valves and purge
            observation_variables=[0, 1, 2, 3, 4, 5, 9],  # Feed-related measurements
            time_scale="fast"
        )
    ]
    
    return configs


def create_default_multi_agent_env(scenario: str = "S1") -> MultiAgentTEPWrapper:
    """Create default multi-agent TEP environment"""
    
    # Create base environment
    base_env = TEPEnvironment(scenario=scenario)
    
    # Create agent configurations
    agent_configs = create_default_agent_configs()
    
    # Create multi-agent wrapper
    ma_env = MultiAgentTEPWrapper(base_env, agent_configs)
    
    return ma_env


# Test the multi-agent wrapper
if __name__ == "__main__":
    # Create multi-agent environment
    env = create_default_multi_agent_env()
    
    print(f"Agent IDs: {env.get_agent_ids()}")
    
    for agent_id in env.get_agent_ids():
        obs_space = env.get_observation_space(agent_id)
        action_space = env.get_action_space(agent_id)
        print(f"{agent_id}: obs_space={obs_space.shape}, action_space={action_space}")
    
    # Test reset
    obs_dict, info_dict = env.reset()
    print(f"Initial observations: {[obs.shape for obs in obs_dict.values()]}")
    
    # Test a few steps
    for step in range(3):
        actions = {}
        for agent_id in env.get_agent_ids():
            actions[agent_id] = env.get_action_space(agent_id).sample()
        
        obs_dict, reward_dict, done_dict, truncated_dict, info_dict = env.step(actions)
        
        print(f"Step {step}: rewards={[f'{r:.3f}' for r in reward_dict.values()]}")
    
    # Get coordination metrics
    coord_metrics = env.get_coordination_metrics()
    print(f"Coordination metrics: {coord_metrics}")
    
    print("Multi-agent wrapper test completed successfully!")

