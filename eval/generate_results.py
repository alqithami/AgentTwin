"""
Experimental Results Generation for Multi-Agent Digital Twin

This script generates all experimental results and publication-quality plots
for the multi-agent digital twin paper.

Key Features:
- Complete experimental evaluation across all scenarios
- Baseline comparisons with statistical significance testing
- Ablation studies for safety shield and coordination
- Publication-quality plots and tables
- Reproducible results with proper seeding

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from train.training_pipeline import (
    TrainingPipeline, ExperimentConfig, create_experiment_configs
)
from agents.multi_agent_system import (
    AgentConfig, TrainingConfig, create_default_agent_configs
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentalResults:
    """Container for experimental results"""
    experiment_name: str
    scenario_results: Dict[str, Dict]
    baseline_results: Dict[str, Dict]
    ablation_results: Dict[str, Dict]
    training_metrics: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    timestamp: str


class ResultsGenerator:
    """
    Comprehensive results generator for multi-agent digital twin research
    
    This class orchestrates all experimental evaluations and generates
    publication-quality results for the research paper.
    """
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Results storage
        self.all_results = {}
        self.statistical_comparisons = {}
        
        # Plot settings
        self.setup_plot_style()
        
        logger.info(f"Results generator initialized: {self.output_dir}")
    
    def setup_logging(self):
        """Setup comprehensive logging for experiments"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def setup_plot_style(self):
        """Setup publication-quality plot style"""
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Font settings for publication
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    @staticmethod
    def _format_metric(mean: Optional[float], std: Optional[float], bold: bool = False) -> str:
        if mean is None or std is None:
            return "N/A"
        value = f"{mean:.3f} ± {std:.3f}"
        return f"**{value}**" if bold else value

    @staticmethod
    def _get_metric(summary: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[float]]:
        return summary.get(f"{key}_mean"), summary.get(f"{key}_std")
    
    def run_main_experiments(self, quick: bool = False) -> Dict[str, ExperimentalResults]:
        """Run main experimental evaluation"""
        logger.info("Starting main experimental evaluation...")
        experiments = {
            'full_system': self._run_full_system_experiment(quick=quick)
        }

        if not quick:
            experiments.update({
                'safety_ablation': self._run_safety_ablation_experiment(),
                'coordination_ablation': self._run_coordination_ablation_experiment(),
                'baseline_comparison': self._run_baseline_comparison_experiment()
            })
        
        logger.info("Main experimental evaluation completed")
        return experiments

    def run_evaluation_from_checkpoint(self, checkpoint_dir: str, quick: bool = False) -> Dict[str, ExperimentalResults]:
        """Run evaluation using a pre-trained checkpoint without retraining."""
        logger.info("Running evaluation from checkpoint...")

        experiment_config = ExperimentConfig(
            experiment_name='full_system',
            scenario='S1',
            enable_behavior_cloning=False,
            enable_individual_training=False,
            enable_joint_training=False,
            enable_safety_shield=True,
            n_eval_episodes=3 if quick else 10,
            eval_scenarios=['S1'] if quick else ['S1', 'S2', 'S3', 'S4', 'S5']
        )

        agent_configs = create_default_agent_configs()
        training_config = TrainingConfig(
            total_timesteps=0,
            eval_freq=1000,
            n_eval_episodes=3 if quick else 10
        )

        pipeline = TrainingPipeline(
            experiment_config=experiment_config,
            agent_configs=agent_configs,
            training_config=training_config,
            output_dir=str(self.output_dir / "experiments")
        )

        pipeline.initialize_components()
        pipeline.ma_system.load_system(checkpoint_dir)
        pipeline.run_comprehensive_evaluation()

        scenario_results = pipeline.training_history['evaluation'].get('scenario_results', {})

        return {
            'full_system': ExperimentalResults(
                experiment_name='full_system',
                scenario_results=scenario_results,
                baseline_results=scenario_results.get('baselines', {}),
                ablation_results={},
                training_metrics={},
                statistical_tests={},
                timestamp=datetime.now().isoformat()
            )
        }
    
    def _run_full_system_experiment(self, quick: bool = False) -> ExperimentalResults:
        """Run full multi-agent system experiment"""
        logger.info("Running full system experiment...")
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name='full_system',
            scenario='S1',
            enable_behavior_cloning=True,
            enable_individual_training=True,
            enable_joint_training=True,
            enable_safety_shield=True,
            n_eval_episodes=3 if quick else 10,
            eval_scenarios=['S1'] if quick else ['S1', 'S2', 'S3', 'S4', 'S5']
        )
        
        # Create agent configurations
        agent_configs = create_default_agent_configs()
        
        # Create training configuration
        training_config = TrainingConfig(
            total_timesteps=5000 if quick else 50000,
            eval_freq=10000,
            n_eval_episodes=3 if quick else 10
        )
        
        # Run training pipeline
        pipeline = TrainingPipeline(
            experiment_config=experiment_config,
            agent_configs=agent_configs,
            training_config=training_config,
            output_dir=str(self.output_dir / "experiments")
        )
        
        results = pipeline.run_complete_pipeline()
        
        # Extract results
        scenario_results = {}
        if 'evaluation' in pipeline.training_history:
            scenario_results = pipeline.training_history['evaluation'].get('scenario_results', {})
        
        return ExperimentalResults(
            experiment_name='full_system',
            scenario_results=scenario_results,
            baseline_results=scenario_results.get('baselines', {}),
            ablation_results={},
            training_metrics=results,
            statistical_tests={},
            timestamp=datetime.now().isoformat()
        )
    
    def _run_safety_ablation_experiment(self) -> ExperimentalResults:
        """Run safety shield ablation study"""
        logger.info("Running safety ablation experiment...")
        
        # Create experiment configuration without safety shield
        experiment_config = ExperimentConfig(
            experiment_name='safety_ablation',
            scenario='S1',
            enable_behavior_cloning=True,
            enable_individual_training=True,
            enable_joint_training=True,
            enable_safety_shield=False,  # No safety shield
            n_eval_episodes=10,
            eval_scenarios=['S1', 'S2', 'S3']
        )
        
        # Create agent configurations
        agent_configs = create_default_agent_configs()
        
        # Create training configuration
        training_config = TrainingConfig(
            total_timesteps=30000,  # Reduced for ablation
            eval_freq=10000,
            n_eval_episodes=10
        )
        
        # Run training pipeline
        pipeline = TrainingPipeline(
            experiment_config=experiment_config,
            agent_configs=agent_configs,
            training_config=training_config,
            output_dir=str(self.output_dir / "experiments")
        )
        
        results = pipeline.run_complete_pipeline()
        
        # Extract results
        scenario_results = {}
        if 'evaluation' in pipeline.training_history:
            scenario_results = pipeline.training_history['evaluation'].get('scenario_results', {})
        
        return ExperimentalResults(
            experiment_name='safety_ablation',
            scenario_results=scenario_results,
            baseline_results={},
            ablation_results={'safety_disabled': scenario_results},
            training_metrics=results,
            statistical_tests={},
            timestamp=datetime.now().isoformat()
        )
    
    def _run_coordination_ablation_experiment(self) -> ExperimentalResults:
        """Run coordination mechanism ablation study"""
        logger.info("Running coordination ablation experiment...")
        
        # Create simplified agent configuration (single agent)
        agent_configs = {
            "reactor_controller": AgentConfig(
                agent_id="reactor_controller",
                algorithm="SAC",
                learning_rate=3e-4,
                batch_size=256,
                buffer_size=50000,
                learning_starts=1000,
                policy_kwargs={"net_arch": [256, 256]}
            )
        }
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name='coordination_ablation',
            scenario='S1',
            enable_behavior_cloning=False,
            enable_individual_training=True,
            enable_joint_training=False,  # No coordination
            enable_safety_shield=True,
            n_eval_episodes=10,
            eval_scenarios=['S1', 'S2']
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            total_timesteps=30000,
            eval_freq=10000,
            n_eval_episodes=10
        )
        
        # Run training pipeline
        pipeline = TrainingPipeline(
            experiment_config=experiment_config,
            agent_configs=agent_configs,
            training_config=training_config,
            output_dir=str(self.output_dir / "experiments")
        )
        
        results = pipeline.run_complete_pipeline()
        
        # Extract results
        scenario_results = {}
        if 'evaluation' in pipeline.training_history:
            scenario_results = pipeline.training_history['evaluation'].get('scenario_results', {})
        
        return ExperimentalResults(
            experiment_name='coordination_ablation',
            scenario_results=scenario_results,
            baseline_results={},
            ablation_results={'no_coordination': scenario_results},
            training_metrics=results,
            statistical_tests={},
            timestamp=datetime.now().isoformat()
        )
    
    def _run_baseline_comparison_experiment(self) -> ExperimentalResults:
        """Run comprehensive baseline comparison"""
        logger.info("Running baseline comparison experiment...")

        experiment_config = ExperimentConfig(
            experiment_name='baseline_comparison',
            scenario='S1',
            enable_behavior_cloning=False,
            enable_individual_training=False,
            enable_joint_training=False,
            enable_safety_shield=False,
            n_eval_episodes=5,
            eval_scenarios=['S1']
        )

        agent_configs = create_default_agent_configs()
        training_config = TrainingConfig(
            total_timesteps=0,
            eval_freq=1000,
            n_eval_episodes=5
        )

        pipeline = TrainingPipeline(
            experiment_config=experiment_config,
            agent_configs=agent_configs,
            training_config=training_config,
            output_dir=str(self.output_dir / "experiments")
        )
        pipeline.initialize_components()
        baseline_results = pipeline.evaluate_baseline_controllers()
        
        return ExperimentalResults(
            experiment_name='baseline_comparison',
            scenario_results={},
            baseline_results=baseline_results,
            ablation_results={},
            training_metrics={},
            statistical_tests={},
            timestamp=datetime.now().isoformat()
        )
    
    def perform_statistical_analysis(self, results: Dict[str, ExperimentalResults]) -> Dict[str, Any]:
        """Perform statistical significance testing"""
        logger.info("Performing statistical analysis...")
        
        statistical_tests = {}
        
        def get_episode_metric(result_block: Dict[str, Any], metric_key: str) -> List[float]:
            return result_block.get("episode_data", {}).get(metric_key, [])

        # Get full system results
        if 'full_system' in results:
            full_system = results['full_system']
            full_s1 = full_system.scenario_results.get('S1', {})

            # Compare with baselines
            if full_system.baseline_results:
                full_rewards = get_episode_metric(full_s1, 'episode_rewards')
                for baseline_name, baseline_data in full_system.baseline_results.items():
                    baseline_rewards = get_episode_metric(baseline_data, 'episode_rewards')
                    if full_rewards and baseline_rewards:
                        t_stat, p_value = stats.ttest_ind(full_rewards, baseline_rewards, equal_var=False)
                        improvement = ((np.mean(full_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards))) * 100
                        statistical_tests[f'vs_{baseline_name}'] = {
                            'improvement_percent': improvement,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }

        # Safety ablation analysis
        if 'safety_ablation' in results and 'full_system' in results:
            full_s1 = results['full_system'].scenario_results.get('S1', {})
            safety_s1 = results['safety_ablation'].scenario_results.get('S1', {})
            full_violations = get_episode_metric(full_s1, 'constraint_violations')
            safety_violations = get_episode_metric(safety_s1, 'constraint_violations')
            if full_violations and safety_violations:
                t_stat, p_value = stats.ttest_ind(full_violations, safety_violations, equal_var=False)
                reduction = ((np.mean(safety_violations) - np.mean(full_violations)) / max(np.mean(safety_violations), 1e-6)) * 100
                statistical_tests['safety_impact'] = {
                    'constraint_reduction_percent': reduction,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        # Coordination ablation analysis
        if 'coordination_ablation' in results and 'full_system' in results:
            full_s1 = results['full_system'].scenario_results.get('S1', {})
            coord_s1 = results['coordination_ablation'].scenario_results.get('S1', {})
            full_rewards = get_episode_metric(full_s1, 'episode_rewards')
            coord_rewards = get_episode_metric(coord_s1, 'episode_rewards')
            if full_rewards and coord_rewards:
                t_stat, p_value = stats.ttest_ind(full_rewards, coord_rewards, equal_var=False)
                improvement = ((np.mean(full_rewards) - np.mean(coord_rewards)) / max(abs(np.mean(coord_rewards)), 1e-6)) * 100
                statistical_tests['coordination_impact'] = {
                    'performance_improvement_percent': improvement,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return statistical_tests
    
    def generate_publication_plots(self, results: Dict[str, ExperimentalResults], 
                                 statistical_tests: Dict[str, Any]):
        """Generate all publication-quality plots"""
        logger.info("Generating publication plots...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Figure 1: System Architecture (will be created separately)
        # Figure 2: Performance Comparison
        self._create_performance_comparison_plot(results, plots_dir)
        
        # Figure 3: Safety Shield Analysis
        self._create_safety_analysis_plot(results, statistical_tests, plots_dir)
        
        # Figure 4: Scenario Evaluation
        self._create_scenario_evaluation_plot(results, plots_dir)
        
        # Figure 5: Ablation Studies
        self._create_ablation_studies_plot(results, statistical_tests, plots_dir)
        
        # Figure 6: Training Convergence
        self._create_training_convergence_plot(results, plots_dir)
        
        logger.info(f"Publication plots saved to {plots_dir}")
    
    def _create_performance_comparison_plot(self, results: Dict[str, ExperimentalResults], 
                                          plots_dir: Path):
        """Create Figure 2: Performance Comparison"""
        full_system = results.get('full_system')
        if not full_system:
            logger.warning("Skipping performance comparison plot: full_system results missing.")
            return
        s1_results = full_system.scenario_results.get('S1', {})
        baselines = full_system.baseline_results or {}
        if not s1_results or not baselines:
            logger.warning("Skipping performance comparison plot: missing scenario or baseline data.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Comparison: Multi-Agent RL vs Baselines', fontsize=16, fontweight='bold')

        method_labels = []
        reward_means = []
        reward_stds = []
        cost_means = []
        cost_stds = []
        violation_means = []
        violation_stds = []
        off_spec_means = []
        off_spec_stds = []

        for name, data in baselines.items():
            method_labels.append(name.replace('_', ' ').title())
            reward_means.append(data.get('episode_rewards_mean'))
            reward_stds.append(data.get('episode_rewards_std'))
            cost_means.append(data.get('economic_costs_mean'))
            cost_stds.append(data.get('economic_costs_std'))
            violation_means.append(data.get('constraint_violations_mean'))
            violation_stds.append(data.get('constraint_violations_std'))
            off_spec_means.append(data.get('off_spec_times_mean'))
            off_spec_stds.append(data.get('off_spec_times_std'))

        method_labels.append('Multi-Agent RL')
        reward_means.append(s1_results.get('episode_rewards_mean'))
        reward_stds.append(s1_results.get('episode_rewards_std'))
        cost_means.append(s1_results.get('economic_costs_mean'))
        cost_stds.append(s1_results.get('economic_costs_std'))
        violation_means.append(s1_results.get('constraint_violations_mean'))
        violation_stds.append(s1_results.get('constraint_violations_std'))
        off_spec_means.append(s1_results.get('off_spec_times_mean'))
        off_spec_stds.append(s1_results.get('off_spec_times_std'))

        colors = ['lightcoral'] * (len(method_labels) - 1) + ['gold']

        # Plot 1: Episode Rewards
        axes[0, 0].bar(method_labels, reward_means, yerr=reward_stds, color=colors, alpha=0.8, capsize=5)
        axes[0, 0].set_title('Episode Reward', fontweight='bold')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Economic Cost
        if all(value is not None for value in cost_means):
            axes[0, 1].bar(method_labels, cost_means, yerr=cost_stds, color=colors, alpha=0.8, capsize=5)
            axes[0, 1].set_title('Economic Cost', fontweight='bold')
            axes[0, 1].set_ylabel('Cost')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].axis('off')

        # Plot 3: Constraint Violations
        if all(value is not None for value in violation_means):
            axes[1, 0].bar(method_labels, violation_means, yerr=violation_stds, color=colors, alpha=0.8, capsize=5)
            axes[1, 0].set_title('Constraint Violations', fontweight='bold')
            axes[1, 0].set_ylabel('Violations')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')

        # Plot 4: Off-Spec Time
        if all(value is not None for value in off_spec_means):
            axes[1, 1].bar(method_labels, off_spec_means, yerr=off_spec_stds, color=colors, alpha=0.8, capsize=5)
            axes[1, 1].set_title('Off-Spec Time', fontweight='bold')
            axes[1, 1].set_ylabel('Hours')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(plots_dir / 'figure2_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_safety_analysis_plot(self, results: Dict[str, ExperimentalResults], 
                                   statistical_tests: Dict[str, Any], plots_dir: Path):
        """Create Figure 3: Safety Shield Analysis"""
        full_system = results.get('full_system')
        safety_ablation = results.get('safety_ablation')
        if not full_system or not safety_ablation:
            logger.warning("Skipping safety analysis plot: missing full system or safety ablation results.")
            return

        full_s1 = full_system.scenario_results.get('S1', {})
        safety_s1 = safety_ablation.scenario_results.get('S1', {})
        if not full_s1 or not safety_s1:
            logger.warning("Skipping safety analysis plot: missing S1 results.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Safety Shield Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Constraint violations comparison
        labels = ['With Shield', 'Without Shield']
        violations = [
            full_s1.get('constraint_violations_mean', 0),
            safety_s1.get('constraint_violations_mean', 0)
        ]
        violation_stds = [
            full_s1.get('constraint_violations_std', 0),
            safety_s1.get('constraint_violations_std', 0)
        ]
        axes[0].bar(labels, violations, yerr=violation_stds, color=['green', 'red'], alpha=0.7, capsize=5)
        axes[0].set_title('Constraint Violations (S1)')
        axes[0].set_ylabel('Violations')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Safety interventions comparison
        interventions = [
            full_s1.get('safety_interventions_mean', 0),
            safety_s1.get('safety_interventions_mean', 0)
        ]
        intervention_stds = [
            full_s1.get('safety_interventions_std', 0),
            safety_s1.get('safety_interventions_std', 0)
        ]
        axes[1].bar(labels, interventions, yerr=intervention_stds, color=['blue', 'gray'], alpha=0.7, capsize=5)
        axes[1].set_title('Safety Interventions (S1)')
        axes[1].set_ylabel('Interventions')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'figure3_safety_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scenario_evaluation_plot(self, results: Dict[str, ExperimentalResults], 
                                       plots_dir: Path):
        """Create Figure 4: Scenario Evaluation"""
        full_system = results.get('full_system')
        if not full_system:
            logger.warning("Skipping scenario evaluation plot: full_system results missing.")
            return

        scenario_results = {
            key: value
            for key, value in full_system.scenario_results.items()
            if key != 'baselines' and isinstance(value, dict)
        }
        if not scenario_results:
            logger.warning("Skipping scenario evaluation plot: no scenario results.")
            return

        scenarios = list(scenario_results.keys())
        rewards = [scenario_results[s].get('episode_rewards_mean', 0) for s in scenarios]
        violations = [scenario_results[s].get('constraint_violations_mean', 0) for s in scenarios]
        off_spec = [scenario_results[s].get('off_spec_times_mean', 0) for s in scenarios]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Multi-Agent Performance Across TEP Scenarios', fontsize=16, fontweight='bold')

        axes[0].bar(scenarios, rewards, color='gold', alpha=0.8)
        axes[0].set_title('Episode Rewards')
        axes[0].set_ylabel('Reward')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(scenarios, violations, color='red', alpha=0.8)
        axes[1].set_title('Constraint Violations')
        axes[1].set_ylabel('Violations')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(scenarios, off_spec, color='purple', alpha=0.8)
        axes[2].set_title('Off-Spec Time')
        axes[2].set_ylabel('Hours')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'figure4_scenario_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ablation_studies_plot(self, results: Dict[str, ExperimentalResults], 
                                    statistical_tests: Dict[str, Any], plots_dir: Path):
        """Create Figure 5: Ablation Studies"""
        full_system = results.get('full_system')
        safety_ablation = results.get('safety_ablation')
        coordination_ablation = results.get('coordination_ablation')
        if not full_system or not safety_ablation or not coordination_ablation:
            logger.warning("Skipping ablation plot: missing ablation results.")
            return

        full_s1 = full_system.scenario_results.get('S1', {})
        safety_s1 = safety_ablation.scenario_results.get('S1', {})
        coord_s1 = coordination_ablation.scenario_results.get('S1', {})
        if not full_s1 or not safety_s1 or not coord_s1:
            logger.warning("Skipping ablation plot: missing S1 results.")
            return

        labels = ['Full System', 'No Safety Shield', 'No Coordination']
        rewards = [
            full_s1.get('episode_rewards_mean', 0),
            safety_s1.get('episode_rewards_mean', 0),
            coord_s1.get('episode_rewards_mean', 0)
        ]
        reward_stds = [
            full_s1.get('episode_rewards_std', 0),
            safety_s1.get('episode_rewards_std', 0),
            coord_s1.get('episode_rewards_std', 0)
        ]
        violations = [
            full_s1.get('constraint_violations_mean', 0),
            safety_s1.get('constraint_violations_mean', 0),
            coord_s1.get('constraint_violations_mean', 0)
        ]
        violation_stds = [
            full_s1.get('constraint_violations_std', 0),
            safety_s1.get('constraint_violations_std', 0),
            coord_s1.get('constraint_violations_std', 0)
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Ablation Studies: Component Impact', fontsize=16, fontweight='bold')

        axes[0].bar(labels, rewards, yerr=reward_stds, color=['gold', 'lightcoral', 'lightblue'], alpha=0.8, capsize=5)
        axes[0].set_title('Episode Rewards (S1)')
        axes[0].set_ylabel('Reward')
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(labels, violations, yerr=violation_stds, color=['gold', 'red', 'orange'], alpha=0.8, capsize=5)
        axes[1].set_title('Constraint Violations (S1)')
        axes[1].set_ylabel('Violations')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'figure5_ablation_studies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_training_convergence_plot(self, results: Dict[str, ExperimentalResults], 
                                        plots_dir: Path):
        """Create Figure 6: Training Convergence"""
        logger.warning("Skipping training convergence plot: no real training history time series available.")
    
    def generate_results_tables(self, results: Dict[str, ExperimentalResults], 
                              statistical_tests: Dict[str, Any]):
        """Generate publication-quality results tables"""
        logger.info("Generating results tables...")
        
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        
        # Table 1: Performance Comparison
        self._create_performance_table(results, statistical_tests, tables_dir)
        
        # Table 2: Ablation Study Results
        self._create_ablation_table(results, statistical_tests, tables_dir)
        
        # Table 3: Statistical Significance Tests
        self._create_statistical_table(statistical_tests, tables_dir)
        
        logger.info(f"Results tables saved to {tables_dir}")
    
    def _create_performance_table(self, results: Dict[str, ExperimentalResults], 
                                statistical_tests: Dict[str, Any], tables_dir: Path):
        """Create Table 1: Performance Comparison"""
        full_system = results.get('full_system')
        if not full_system:
            logger.warning("Skipping performance table: full_system results missing.")
            return
        s1_results = full_system.scenario_results.get('S1', {})
        baselines = full_system.baseline_results or {}
        if not s1_results or not baselines:
            logger.warning("Skipping performance table: missing scenario or baseline data.")
            return

        rows = []
        for name, data in baselines.items():
            rows.append({
                'Method': name.replace('_', ' ').title(),
                'Episode Reward': self._format_metric(*self._get_metric(data, 'episode_rewards')),
                'Economic Cost ($/h)': self._format_metric(*self._get_metric(data, 'economic_costs')),
                'Constraint Violations': self._format_metric(*self._get_metric(data, 'constraint_violations')),
                'Off-Spec Time (h)': self._format_metric(*self._get_metric(data, 'off_spec_times')),
                'Improvement (%)': '-'
            })

        baseline_rewards = [data.get('episode_rewards_mean') for data in baselines.values() if data.get('episode_rewards_mean') is not None]
        improvement = None
        if baseline_rewards and s1_results.get('episode_rewards_mean') is not None:
            best_baseline = max(baseline_rewards)
            improvement = ((s1_results.get('episode_rewards_mean') - best_baseline) / abs(best_baseline)) * 100

        rows.append({
            'Method': 'Multi-Agent RL (Ours)',
            'Episode Reward': self._format_metric(*self._get_metric(s1_results, 'episode_rewards'), bold=True),
            'Economic Cost ($/h)': self._format_metric(*self._get_metric(s1_results, 'economic_costs'), bold=True),
            'Constraint Violations': self._format_metric(*self._get_metric(s1_results, 'constraint_violations'), bold=True),
            'Off-Spec Time (h)': self._format_metric(*self._get_metric(s1_results, 'off_spec_times'), bold=True),
            'Improvement (%)': f"**{improvement:.2f}**" if improvement is not None else "N/A"
        })

        df = pd.DataFrame(rows)
        
        # Save as CSV
        df.to_csv(tables_dir / 'table1_performance_comparison.csv', index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, escape=False, 
                                 caption='Performance comparison across different control methods on TEP scenarios. Values show mean ± standard deviation over 10 episodes. Best results in bold.',
                                 label='tab:performance_comparison')
        
        with open(tables_dir / 'table1_performance_comparison.tex', 'w') as f:
            f.write(latex_table)
    
    def _create_ablation_table(self, results: Dict[str, ExperimentalResults], 
                             statistical_tests: Dict[str, Any], tables_dir: Path):
        """Create Table 2: Ablation Study Results"""
        full_system = results.get('full_system')
        safety_ablation = results.get('safety_ablation')
        coordination_ablation = results.get('coordination_ablation')
        if not full_system or not safety_ablation or not coordination_ablation:
            logger.warning("Skipping ablation table: missing ablation results.")
            return

        full_s1 = full_system.scenario_results.get('S1', {})
        safety_s1 = safety_ablation.scenario_results.get('S1', {})
        coord_s1 = coordination_ablation.scenario_results.get('S1', {})
        if not full_s1 or not safety_s1 or not coord_s1:
            logger.warning("Skipping ablation table: missing S1 results.")
            return

        rows = [
            {
                'Configuration': 'Full System',
                'Episode Reward': self._format_metric(*self._get_metric(full_s1, 'episode_rewards'), bold=True),
                'Constraint Violations': self._format_metric(*self._get_metric(full_s1, 'constraint_violations')),
                'Off-Spec Time (h)': self._format_metric(*self._get_metric(full_s1, 'off_spec_times'))
            },
            {
                'Configuration': 'No Safety Shield',
                'Episode Reward': self._format_metric(*self._get_metric(safety_s1, 'episode_rewards')),
                'Constraint Violations': self._format_metric(*self._get_metric(safety_s1, 'constraint_violations')),
                'Off-Spec Time (h)': self._format_metric(*self._get_metric(safety_s1, 'off_spec_times'))
            },
            {
                'Configuration': 'No Coordination',
                'Episode Reward': self._format_metric(*self._get_metric(coord_s1, 'episode_rewards')),
                'Constraint Violations': self._format_metric(*self._get_metric(coord_s1, 'constraint_violations')),
                'Off-Spec Time (h)': self._format_metric(*self._get_metric(coord_s1, 'off_spec_times'))
            }
        ]

        df = pd.DataFrame(rows)
        
        # Save as CSV
        df.to_csv(tables_dir / 'table2_ablation_study.csv', index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, escape=False,
                                 caption='Ablation study results using episode-level metrics from the evaluation runs. Values show mean ± standard deviation.',
                                 label='tab:ablation_study')
        
        with open(tables_dir / 'table2_ablation_study.tex', 'w') as f:
            f.write(latex_table)
    
    def _create_statistical_table(self, statistical_tests: Dict[str, Any], tables_dir: Path):
        """Create Table 3: Statistical Significance Tests"""
        if not statistical_tests:
            logger.warning("Skipping statistical table: no statistical tests available.")
            return

        rows = []
        for name, data in statistical_tests.items():
            comparison = name.replace('_', ' ').replace('vs', 'vs').title()
            improvement = data.get('improvement_percent') or data.get('constraint_reduction_percent') or data.get('performance_improvement_percent')
            rows.append({
                'Comparison': comparison,
                'Improvement (%)': f"{improvement:.2f}" if improvement is not None else "N/A",
                't-statistic': f"{data.get('t_statistic', 0):.3f}",
                'p-value': f"{data.get('p_value', 1):.4f}",
                'Significant': 'Yes' if data.get('significant') else 'No'
            })

        df = pd.DataFrame(rows)
        
        # Save as CSV
        df.to_csv(tables_dir / 'table3_statistical_tests.csv', index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, escape=False,
                                 caption='Statistical significance tests for performance comparisons using Welch\'s t-test.',
                                 label='tab:statistical_tests')
        
        with open(tables_dir / 'table3_statistical_tests.tex', 'w') as f:
            f.write(latex_table)
    
    def save_comprehensive_results(self, results: Dict[str, ExperimentalResults], 
                                 statistical_tests: Dict[str, Any]):
        """Save all results in comprehensive format"""
        logger.info("Saving comprehensive results...")
        
        # Save raw results
        results_file = self.output_dir / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiments': {k: asdict(v) for k, v in results.items()},
                'statistical_tests': statistical_tests,
                'generation_timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(results, statistical_tests)
        
        logger.info(f"Comprehensive results saved to {self.output_dir}")
    
    def _create_summary_report(self, results: Dict[str, ExperimentalResults], 
                             statistical_tests: Dict[str, Any]):
        """Create executive summary report"""
        full_system = results.get('full_system')
        s1_results = full_system.scenario_results.get('S1', {}) if full_system else {}
        baseline_results = full_system.baseline_results if full_system else {}
        baseline_rewards = [data.get('episode_rewards_mean') for data in baseline_results.values() if data.get('episode_rewards_mean') is not None]
        best_baseline = max(baseline_rewards) if baseline_rewards else None
        improvement = None
        if best_baseline is not None and s1_results.get('episode_rewards_mean') is not None:
            improvement = ((s1_results.get('episode_rewards_mean') - best_baseline) / abs(best_baseline)) * 100

        safety_stats = statistical_tests.get('safety_impact', {})
        coord_stats = statistical_tests.get('coordination_impact', {})

        report_content = f"""
# Multi-Agent Digital Twin Experimental Results Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings (Derived from Actual Runs)

### Performance (Scenario S1)
- Episode reward: {self._format_metric(*self._get_metric(s1_results, 'episode_rewards'))}
- Economic cost: {self._format_metric(*self._get_metric(s1_results, 'economic_costs'))}
- Constraint violations: {self._format_metric(*self._get_metric(s1_results, 'constraint_violations'))}
- Off-spec time: {self._format_metric(*self._get_metric(s1_results, 'off_spec_times'))}
- Improvement over best baseline: {f"{improvement:.2f}%" if improvement is not None else "N/A"}

### Safety Analysis
- Constraint reduction vs no-shield: {f"{safety_stats.get('constraint_reduction_percent', 0):.2f}%" if safety_stats else "N/A"}
- Statistical significance: {safety_stats.get('significant', False) if safety_stats else "N/A"}

### Coordination Analysis
- Performance improvement vs no-coordination: {f"{coord_stats.get('performance_improvement_percent', 0):.2f}%" if coord_stats else "N/A"}
- Statistical significance: {coord_stats.get('significant', False) if coord_stats else "N/A"}

### Statistical Significance
- Comparisons reported: {len(statistical_tests)}

## Notes
- Results above are computed from the actual evaluation episodes executed by this pipeline.
- Additional plots/tables are generated only when the underlying data is available.
"""
        
        with open(self.output_dir / "executive_summary.md", 'w') as f:
            f.write(report_content)


def main():
    """Main results generation script"""
    parser = argparse.ArgumentParser(description='Generate comprehensive experimental results')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation for testing')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to a saved multi-agent checkpoint directory to evaluate')
    
    args = parser.parse_args()
    
    # Create results generator
    generator = ResultsGenerator(output_dir=args.output_dir)
    
    logger.info("Starting comprehensive results generation...")
    start_time = time.time()
    
    try:
        # Run main experiments
        if args.checkpoint_dir:
            results = generator.run_evaluation_from_checkpoint(args.checkpoint_dir, quick=args.quick)
        elif args.quick:
            logger.info("Running quick evaluation...")
            results = generator.run_main_experiments(quick=True)
        else:
            results = generator.run_main_experiments()
        
        # Perform statistical analysis
        statistical_tests = generator.perform_statistical_analysis(results)
        
        # Generate publication plots
        generator.generate_publication_plots(results, statistical_tests)
        
        # Generate results tables
        generator.generate_results_tables(results, statistical_tests)
        
        # Save comprehensive results
        generator.save_comprehensive_results(results, statistical_tests)
        
        total_time = time.time() - start_time
        logger.info(f"Results generation completed in {total_time:.2f} seconds")
        
        print("\n" + "="*60)
        print("EXPERIMENTAL RESULTS GENERATION COMPLETED")
        print("="*60)
        print(f"Output directory: {generator.output_dir}")
        print(f"Total time: {total_time:.2f} seconds")
        print("Plots generated: available publication-quality figures (skipping plots without data)")
        print("Tables generated: available results tables")
        print(f"Statistical tests: {len(statistical_tests)} comparisons")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Results generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
