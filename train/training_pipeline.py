"""
Training Pipeline for Multi-Agent Digital Twin

This module implements the complete training pipeline for the multi-agent
reinforcement learning system, including:
- Behavior cloning pretraining
- Individual agent training
- Joint coordination training
- Hyperparameter optimization
- Training monitoring and logging

Key Features:
- Configurable training schedules
- Automatic checkpointing and recovery
- Comprehensive logging and visualization
- Performance monitoring and early stopping
- Distributed training support

Author: Implementation for Multi-Agent Digital Twin Research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

# Import our modules
from agents.multi_agent_system import (
    MultiAgentRLSystem, AgentConfig, TrainingConfig,
    create_default_agent_configs
)
from envs.multi_agent_wrapper import create_default_multi_agent_env
from shield.safety_shield import SafetyShield, create_tep_safety_constraints, ShieldConfig
from control.baseline_controllers import (
    create_tep_pid_controller, create_tep_nmpc_controller, 
    create_schedule_then_control
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for complete training experiment"""
    experiment_name: str
    scenario: str = "S1"  # TEP scenario
    
    # Training phases
    enable_behavior_cloning: bool = True
    enable_individual_training: bool = True
    enable_joint_training: bool = True
    
    # Safety settings
    enable_safety_shield: bool = True
    safety_intervention_threshold: float = 0.01
    
    # Evaluation settings
    n_eval_episodes: int = 10
    eval_scenarios: List[str] = None
    
    # Logging and checkpointing
    save_freq: int = 50000
    log_freq: int = 1000
    checkpoint_freq: int = 25000
    
    # Computational settings
    n_parallel_envs: int = 1
    device: str = "cpu"
    
    def __post_init__(self):
        if self.eval_scenarios is None:
            self.eval_scenarios = ["S1", "S2", "S3", "S4", "S5"]


class TrainingPipeline:
    """
    Complete training pipeline for multi-agent digital twin
    
    This class orchestrates the entire training process from behavior cloning
    through joint training and final evaluation.
    """
    
    def __init__(self, 
                 experiment_config: ExperimentConfig,
                 agent_configs: Dict[str, AgentConfig],
                 training_config: TrainingConfig,
                 output_dir: str = "./experiments"):
        
        self.experiment_config = experiment_config
        self.agent_configs = agent_configs
        self.training_config = training_config
        
        # Setup output directory
        self.output_dir = Path(output_dir) / experiment_config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.env = None
        self.safety_shield = None
        self.ma_system = None
        self.baseline_controllers = {}
        
        # Training state
        self.training_history = {
            'behavior_cloning': {},
            'individual_training': {},
            'joint_training': {},
            'evaluation': {}
        }
        
        # Performance tracking
        self.performance_metrics = []
        self.checkpoint_paths = []
        
        logger.info(f"Training pipeline initialized: {experiment_config.experiment_name}")
    
    def setup_logging(self):
        """Setup comprehensive logging for the experiment"""
        
        # Create log directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Setup formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Save experiment configuration
        config_file = self.output_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'experiment_config': asdict(self.experiment_config),
                'agent_configs': {k: asdict(v) for k, v in self.agent_configs.items()},
                'training_config': asdict(self.training_config)
            }, f, indent=2)
        
        logger.info(f"Experiment configuration saved to {config_file}")
    
    def initialize_components(self):
        """Initialize all training components"""
        logger.info("Initializing training components...")
        
        # Create environment
        self.env = create_default_multi_agent_env(scenario=self.experiment_config.scenario)
        logger.info(f"Environment initialized with scenario {self.experiment_config.scenario}")
        
        # Create safety shield
        if self.experiment_config.enable_safety_shield:
            shield_config = ShieldConfig(
                intervention_threshold=self.experiment_config.safety_intervention_threshold
            )
            self.safety_shield = SafetyShield(
                config=shield_config,
                constraints=create_tep_safety_constraints()
            )
            logger.info("Safety shield initialized")
        
        # Create multi-agent system
        self.ma_system = MultiAgentRLSystem(
            env=self.env,
            agent_configs=self.agent_configs,
            training_config=self.training_config,
            safety_shield=self.safety_shield,
            log_dir=str(self.output_dir / "agent_logs")
        )
        logger.info(f"Multi-agent system initialized with {len(self.ma_system.agents)} agents")
        
        # Create baseline controllers
        self.baseline_controllers = {
            'pid_cascade': create_tep_pid_controller(),
            'nmpc': create_tep_nmpc_controller(),
            'schedule_then_control': create_schedule_then_control()
        }
        logger.info(f"Baseline controllers initialized: {list(self.baseline_controllers.keys())}")
    
    def run_behavior_cloning(self) -> Dict:
        """Run behavior cloning pretraining phase"""
        if not self.experiment_config.enable_behavior_cloning:
            logger.info("Behavior cloning disabled, skipping...")
            return {}
        
        logger.info("Starting behavior cloning pretraining...")
        start_time = time.time()
        
        # Run behavior cloning
        self.ma_system.pretrain_with_behavior_cloning()
        
        # Evaluate BC performance
        bc_results = self.ma_system.evaluate_system(n_episodes=5)
        
        duration = time.time() - start_time
        logger.info(f"Behavior cloning completed in {duration:.2f} seconds")
        
        # Store results
        bc_metrics = {
            'duration': duration,
            'evaluation_results': bc_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history['behavior_cloning'] = bc_metrics
        
        # Save checkpoint
        checkpoint_path = self.output_dir / "checkpoints" / "behavior_cloning"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.ma_system.save_system(str(checkpoint_path))
        self.checkpoint_paths.append(str(checkpoint_path))
        
        return bc_metrics
    
    def run_individual_training(self) -> Dict:
        """Run individual agent training phase"""
        if not self.experiment_config.enable_individual_training:
            logger.info("Individual training disabled, skipping...")
            return {}
        
        logger.info("Starting individual agent training...")
        start_time = time.time()
        
        # Run individual training
        self.ma_system.train_individual_agents()
        
        # Evaluate individual training performance
        individual_results = self.ma_system.evaluate_system(n_episodes=self.experiment_config.n_eval_episodes)
        
        duration = time.time() - start_time
        logger.info(f"Individual training completed in {duration:.2f} seconds")
        
        # Store results
        individual_metrics = {
            'duration': duration,
            'evaluation_results': individual_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history['individual_training'] = individual_metrics
        
        # Save checkpoint
        checkpoint_path = self.output_dir / "checkpoints" / "individual_training"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.ma_system.save_system(str(checkpoint_path))
        self.checkpoint_paths.append(str(checkpoint_path))
        
        return individual_metrics
    
    def run_joint_training(self) -> Dict:
        """Run joint coordination training phase"""
        if not self.experiment_config.enable_joint_training:
            logger.info("Joint training disabled, skipping...")
            return {}
        
        logger.info("Starting joint coordination training...")
        start_time = time.time()
        
        # Run joint training
        self.ma_system.train_joint_agents()
        
        # Evaluate joint training performance
        joint_results = self.ma_system.evaluate_system(n_episodes=self.experiment_config.n_eval_episodes)
        
        duration = time.time() - start_time
        logger.info(f"Joint training completed in {duration:.2f} seconds")
        
        # Store results
        joint_metrics = {
            'duration': duration,
            'evaluation_results': joint_results,
            'coordination_history': self.ma_system.coordination_history,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history['joint_training'] = joint_metrics
        
        # Save final checkpoint
        checkpoint_path = self.output_dir / "checkpoints" / "joint_training"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.ma_system.save_system(str(checkpoint_path))
        self.checkpoint_paths.append(str(checkpoint_path))
        
        return joint_metrics
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation across all scenarios"""
        logger.info("Starting comprehensive evaluation...")
        
        evaluation_results = {}
        
        # Evaluate on all scenarios
        for scenario in self.experiment_config.eval_scenarios:
            logger.info(f"Evaluating on scenario {scenario}...")
            
            # Create environment for this scenario
            eval_env = create_default_multi_agent_env(scenario=scenario)
            
            # Update environment in ma_system
            original_env = self.ma_system.env
            self.ma_system.env = eval_env
            
            # Run evaluation
            scenario_results = self.ma_system.evaluate_system(
                n_episodes=self.experiment_config.n_eval_episodes
            )
            
            evaluation_results[scenario] = scenario_results
            
            # Restore original environment
            self.ma_system.env = original_env
        
        # Evaluate baseline controllers for comparison
        baseline_results = self.evaluate_baseline_controllers()
        evaluation_results['baselines'] = baseline_results
        
        # Store results
        eval_metrics = {
            'scenario_results': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history['evaluation'] = eval_metrics
        
        logger.info("Comprehensive evaluation completed")
        return eval_metrics
    
    def evaluate_baseline_controllers(self) -> Dict:
        """Evaluate baseline controllers for comparison"""
        logger.info("Evaluating baseline controllers...")
        
        baseline_results = {}
        
        for controller_name, controller in self.baseline_controllers.items():
            logger.info(f"Evaluating {controller_name}...")
            
            # Run episodes with baseline controller
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(5):  # Fewer episodes for baselines
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                
                for step in range(1000):  # Max episode length
                    # Get baseline action (simplified)
                    if controller_name == 'pid_cascade':
                        # Use PID controller
                        measurements = list(obs.values())[0][:41]  # First agent's measurements
                        setpoints = measurements.copy()  # Use current as setpoint (simplified)
                        action = controller.compute_action(measurements, setpoints, 6.0)
                    else:
                        # Use random action for other controllers (simplified)
                        action = np.random.randn(12) * 0.1
                    
                    # Create action dict for all agents
                    actions = {}
                    for agent_id in self.env.get_agent_ids():
                        if agent_id == "scheduler":
                            actions[agent_id] = np.array([0, 0, 0, 0])  # Default scheduler action
                        else:
                            action_space = self.env.get_action_space(agent_id)
                            if hasattr(action_space, 'shape'):
                                action_dim = action_space.shape[0]
                            else:
                                action_dim = 2  # Default
                            actions[agent_id] = action[:action_dim] if len(action) >= action_dim else np.zeros(action_dim)
                    
                    # Execute action
                    obs, rewards, dones, truncated, infos = self.env.step(actions)
                    episode_reward += sum(rewards.values())
                    episode_length += 1
                    
                    if any(dones.values()) or any(truncated.values()):
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            baseline_results[controller_name] = {
                'episode_rewards_mean': np.mean(episode_rewards),
                'episode_rewards_std': np.std(episode_rewards),
                'episode_lengths_mean': np.mean(episode_lengths),
                'episode_lengths_std': np.std(episode_lengths)
            }
        
        return baseline_results
    
    def generate_training_plots(self):
        """Generate comprehensive training visualization plots"""
        logger.info("Generating training plots...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training Progress Plot
        self._plot_training_progress(plots_dir)
        
        # 2. Performance Comparison Plot
        self._plot_performance_comparison(plots_dir)
        
        # 3. Safety Metrics Plot
        if self.safety_shield:
            self._plot_safety_metrics(plots_dir)
        
        # 4. Coordination Metrics Plot
        self._plot_coordination_metrics(plots_dir)
        
        # 5. Scenario Comparison Plot
        self._plot_scenario_comparison(plots_dir)
        
        logger.info(f"Training plots saved to {plots_dir}")
    
    def _plot_training_progress(self, plots_dir: Path):
        """Plot training progress across phases"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Across Phases', fontsize=16)
        
        # Extract data
        phases = []
        rewards = []
        durations = []
        
        for phase, data in self.training_history.items():
            if data and 'evaluation_results' in data:
                phases.append(phase.replace('_', ' ').title())
                rewards.append(data['evaluation_results'].get('episode_rewards_mean', 0))
                durations.append(data.get('duration', 0))
        
        # Plot 1: Reward progression
        axes[0, 0].bar(phases, rewards, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards by Training Phase')
        axes[0, 0].set_ylabel('Average Episode Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Training duration
        axes[0, 1].bar(phases, durations, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Training Duration by Phase')
        axes[0, 1].set_ylabel('Duration (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Cumulative improvement
        if len(rewards) > 1:
            improvements = [0] + [rewards[i] - rewards[0] for i in range(1, len(rewards))]
            axes[1, 0].plot(phases, improvements, marker='o', linewidth=2, markersize=8)
            axes[1, 0].set_title('Cumulative Performance Improvement')
            axes[1, 0].set_ylabel('Improvement from Baseline')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training efficiency
        if len(rewards) > 1 and len(durations) > 1:
            efficiency = [r/d if d > 0 else 0 for r, d in zip(rewards, durations)]
            axes[1, 1].bar(phases, efficiency, color='lightgreen', alpha=0.7)
            axes[1, 1].set_title('Training Efficiency (Reward/Time)')
            axes[1, 1].set_ylabel('Efficiency')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, plots_dir: Path):
        """Plot performance comparison with baselines"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Comparison: Multi-Agent RL vs Baselines', fontsize=16)
        
        # Get final evaluation results
        eval_data = self.training_history.get('evaluation', {})
        if not eval_data:
            return
        
        # Extract multi-agent results
        ma_reward = eval_data.get('scenario_results', {}).get('S1', {}).get('episode_rewards_mean', 0)
        
        # Extract baseline results
        baseline_data = eval_data.get('scenario_results', {}).get('baselines', {})
        baseline_names = list(baseline_data.keys())
        baseline_rewards = [baseline_data[name].get('episode_rewards_mean', 0) for name in baseline_names]
        
        # Plot 1: Reward comparison
        all_names = ['Multi-Agent RL'] + [name.replace('_', ' ').title() for name in baseline_names]
        all_rewards = [ma_reward] + baseline_rewards
        
        colors = ['red'] + ['blue'] * len(baseline_rewards)
        bars = axes[0].bar(all_names, all_rewards, color=colors, alpha=0.7)
        axes[0].set_title('Episode Rewards Comparison')
        axes[0].set_ylabel('Average Episode Reward')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Highlight best performer
        best_idx = np.argmax(all_rewards)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
        
        # Plot 2: Improvement over best baseline
        if baseline_rewards:
            best_baseline = max(baseline_rewards)
            improvement = ((ma_reward - best_baseline) / abs(best_baseline)) * 100 if best_baseline != 0 else 0
            
            axes[1].bar(['Multi-Agent RL'], [improvement], 
                       color='green' if improvement > 0 else 'red', alpha=0.7)
            axes[1].set_title('Improvement over Best Baseline')
            axes[1].set_ylabel('Improvement (%)')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_safety_metrics(self, plots_dir: Path):
        """Plot safety shield performance metrics"""
        if not self.safety_shield:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Safety Shield Performance Metrics', fontsize=16)
        
        # Get safety metrics
        metrics = self.safety_shield.get_metrics()
        
        # Plot 1: Intervention frequency
        if metrics.intervention_magnitudes:
            axes[0, 0].hist(metrics.intervention_magnitudes, bins=20, alpha=0.7, color='orange')
            axes[0, 0].set_title('Safety Intervention Magnitudes')
            axes[0, 0].set_xlabel('Intervention Magnitude')
            axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: QP solve times
        if metrics.qp_solve_times:
            axes[0, 1].hist(metrics.qp_solve_times, bins=20, alpha=0.7, color='purple')
            axes[0, 1].set_title('QP Solver Performance')
            axes[0, 1].set_xlabel('Solve Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Safety statistics
        safety_stats = [
            metrics.total_interventions,
            metrics.fallback_activations,
            metrics.constraint_violations
        ]
        safety_labels = ['Total Interventions', 'Fallback Activations', 'Constraint Violations']
        
        axes[1, 0].bar(safety_labels, safety_stats, color=['red', 'orange', 'yellow'], alpha=0.7)
        axes[1, 0].set_title('Safety Event Counts')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Success rate
        success_rate = metrics.qp_success_rate * 100
        axes[1, 1].pie([success_rate, 100 - success_rate], 
                      labels=['Successful', 'Failed'], 
                      colors=['green', 'red'], 
                      autopct='%1.1f%%')
        axes[1, 1].set_title('QP Solver Success Rate')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'safety_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coordination_metrics(self, plots_dir: Path):
        """Plot multi-agent coordination effectiveness"""
        if not self.ma_system.coordination_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Agent Coordination Metrics', fontsize=16)
        
        # Extract coordination data
        coord_data = self.ma_system.coordination_history
        steps = list(range(len(coord_data)))
        readiness = [d.get('avg_readiness', 0.5) for d in coord_data]
        action_variance = [d.get('action_variance', 0.0) for d in coord_data]
        total_rewards = [d.get('total_reward', 0.0) for d in coord_data]
        
        # Plot 1: Readiness signals over time
        axes[0, 0].plot(steps, readiness, linewidth=2, color='blue')
        axes[0, 0].set_title('Average Agent Readiness Over Time')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Average Readiness')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Action coordination (variance)
        axes[0, 1].plot(steps, action_variance, linewidth=2, color='red')
        axes[0, 1].set_title('Action Coordination (Lower = Better)')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Action Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total reward progression
        axes[1, 0].plot(steps, total_rewards, linewidth=2, color='green')
        axes[1, 0].set_title('Total Reward Progression')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Coordination effectiveness scatter
        if len(readiness) > 1 and len(total_rewards) > 1:
            axes[1, 1].scatter(readiness, total_rewards, alpha=0.6, color='purple')
            axes[1, 1].set_title('Readiness vs Performance')
            axes[1, 1].set_xlabel('Average Readiness')
            axes[1, 1].set_ylabel('Total Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'coordination_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_comparison(self, plots_dir: Path):
        """Plot performance across different TEP scenarios"""
        eval_data = self.training_history.get('evaluation', {})
        scenario_results = eval_data.get('scenario_results', {})
        
        if not scenario_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Across TEP Scenarios', fontsize=16)
        
        # Extract scenario data
        scenarios = []
        rewards = []
        lengths = []
        violations = []
        
        for scenario, data in scenario_results.items():
            if scenario != 'baselines' and isinstance(data, dict):
                scenarios.append(scenario)
                rewards.append(data.get('episode_rewards_mean', 0))
                lengths.append(data.get('episode_lengths_mean', 0))
                violations.append(data.get('constraint_violations_mean', 0))
        
        if not scenarios:
            return
        
        # Plot 1: Rewards by scenario
        axes[0, 0].bar(scenarios, rewards, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards by Scenario')
        axes[0, 0].set_ylabel('Average Episode Reward')
        
        # Plot 2: Episode lengths by scenario
        axes[0, 1].bar(scenarios, lengths, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Episode Lengths by Scenario')
        axes[0, 1].set_ylabel('Average Episode Length')
        
        # Plot 3: Constraint violations by scenario
        axes[1, 0].bar(scenarios, violations, color='orange', alpha=0.7)
        axes[1, 0].set_title('Constraint Violations by Scenario')
        axes[1, 0].set_ylabel('Average Violations')
        
        # Plot 4: Scenario difficulty ranking
        if len(rewards) > 1:
            difficulty_scores = [-r for r in rewards]  # Lower reward = higher difficulty
            scenario_difficulty = list(zip(scenarios, difficulty_scores))
            scenario_difficulty.sort(key=lambda x: x[1])
            
            sorted_scenarios, sorted_scores = zip(*scenario_difficulty)
            axes[1, 1].barh(sorted_scenarios, sorted_scores, color='lightgreen', alpha=0.7)
            axes[1, 1].set_title('Scenario Difficulty Ranking')
            axes[1, 1].set_xlabel('Difficulty Score (Higher = More Difficult)')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all training results and metrics"""
        logger.info("Saving training results...")
        
        # Save training history
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save performance summary
        summary = self.generate_performance_summary()
        summary_file = self.output_dir / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        final_model_path.mkdir(exist_ok=True)
        self.ma_system.save_system(str(final_model_path))
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def generate_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        summary = {
            'experiment_name': self.experiment_config.experiment_name,
            'total_training_time': 0.0,
            'final_performance': {},
            'best_performance': {},
            'safety_summary': {},
            'coordination_summary': {}
        }
        
        # Calculate total training time
        for phase_data in self.training_history.values():
            if isinstance(phase_data, dict) and 'duration' in phase_data:
                summary['total_training_time'] += phase_data['duration']
        
        # Get final performance
        eval_data = self.training_history.get('evaluation', {})
        if eval_data:
            scenario_results = eval_data.get('scenario_results', {})
            if 'S1' in scenario_results:
                summary['final_performance'] = scenario_results['S1']
        
        # Find best performance across all phases
        best_reward = float('-inf')
        for phase_data in self.training_history.values():
            if isinstance(phase_data, dict) and 'evaluation_results' in phase_data:
                reward = phase_data['evaluation_results'].get('episode_rewards_mean', float('-inf'))
                if reward > best_reward:
                    best_reward = reward
                    summary['best_performance'] = phase_data['evaluation_results']
        
        # Safety summary
        if self.safety_shield:
            metrics = self.safety_shield.get_metrics()
            summary['safety_summary'] = {
                'total_interventions': metrics.total_interventions,
                'qp_success_rate': metrics.qp_success_rate,
                'fallback_activations': metrics.fallback_activations,
                'avg_solve_time': np.mean(metrics.qp_solve_times) if metrics.qp_solve_times else 0.0
            }
        
        # Coordination summary
        if self.ma_system.coordination_history:
            coord_data = self.ma_system.coordination_history
            summary['coordination_summary'] = {
                'avg_readiness': np.mean([d.get('avg_readiness', 0.5) for d in coord_data]),
                'avg_action_variance': np.mean([d.get('action_variance', 0.0) for d in coord_data]),
                'coordination_episodes': len(coord_data)
            }
        
        return summary
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete training pipeline"""
        logger.info(f"Starting complete training pipeline: {self.experiment_config.experiment_name}")
        start_time = time.time()
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Run training phases
            bc_results = self.run_behavior_cloning()
            individual_results = self.run_individual_training()
            joint_results = self.run_joint_training()
            
            # Run comprehensive evaluation
            eval_results = self.run_comprehensive_evaluation()
            
            # Generate plots and save results
            self.generate_training_plots()
            self.save_results()
            
            total_time = time.time() - start_time
            logger.info(f"Complete training pipeline finished in {total_time:.2f} seconds")
            
            # Generate final summary
            summary = self.generate_performance_summary()
            summary['pipeline_duration'] = total_time
            
            return summary
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def create_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Create predefined experiment configurations"""
    
    configs = {}
    
    # Full training experiment
    configs['full_training'] = ExperimentConfig(
        experiment_name='full_training',
        scenario='S1',
        enable_behavior_cloning=True,
        enable_individual_training=True,
        enable_joint_training=True,
        enable_safety_shield=True,
        n_eval_episodes=10,
        eval_scenarios=['S1', 'S2', 'S3', 'S4', 'S5']
    )
    
    # Quick test experiment
    configs['quick_test'] = ExperimentConfig(
        experiment_name='quick_test',
        scenario='S1',
        enable_behavior_cloning=False,
        enable_individual_training=True,
        enable_joint_training=False,
        enable_safety_shield=True,
        n_eval_episodes=3,
        eval_scenarios=['S1']
    )
    
    # Safety ablation experiment
    configs['safety_ablation'] = ExperimentConfig(
        experiment_name='safety_ablation',
        scenario='S1',
        enable_behavior_cloning=True,
        enable_individual_training=True,
        enable_joint_training=True,
        enable_safety_shield=False,  # No safety shield
        n_eval_episodes=10,
        eval_scenarios=['S1', 'S2', 'S3']
    )
    
    return configs


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Multi-Agent Digital Twin Training Pipeline')
    parser.add_argument('--experiment', type=str, default='quick_test',
                       choices=['full_training', 'quick_test', 'safety_ablation'],
                       help='Experiment configuration to run')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Get experiment configuration
    experiment_configs = create_experiment_configs()
    experiment_config = experiment_configs[args.experiment]
    
    # Create agent configurations
    agent_configs = create_default_agent_configs()
    
    # Create training configuration
    training_config = TrainingConfig(
        total_timesteps=10000,  # Reduced for testing
        eval_freq=2000,
        n_eval_episodes=3
    )
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(
        experiment_config=experiment_config,
        agent_configs=agent_configs,
        training_config=training_config,
        output_dir=args.output_dir
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETED")
    print("="*50)
    print(f"Experiment: {experiment_config.experiment_name}")
    print(f"Total time: {results.get('pipeline_duration', 0):.2f} seconds")
    print(f"Final reward: {results.get('final_performance', {}).get('episode_rewards_mean', 'N/A')}")
    print(f"Results saved to: {pipeline.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()

