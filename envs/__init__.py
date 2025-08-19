"""
Environment package for Multi-Agent Digital Twin

This package contains the Tennessee Eastman Process environment
and related utilities for multi-agent reinforcement learning.
"""

from .tep_env import TEPEnvironment, TEPConfig, ProductionMode, ScenarioType

__all__ = ['TEPEnvironment', 'TEPConfig', 'ProductionMode', 'ScenarioType']

