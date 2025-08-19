#!/usr/bin/env python3
"""
Quick training configuration for fast testing and validation
"""

class QuickTrainingConfig:
    """Quick configuration for rapid testing - completes in 5-10 minutes"""
    
    def __init__(self):
        # Drastically reduced training parameters for speed
        self.total_timesteps = 5000           # Very small for quick test
        self.eval_freq = 1000                 # Evaluate often
        self.save_freq = 2000                 # Save often
        
        # Behavior cloning parameters (quick)
        self.bc_learning_rate = 1e-3
        self.bc_epochs = 5                    # Very few epochs
        self.bc_batch_size = 16               # Small batch
        
        # Agent parameters (quick)
        self.learning_rate = 3e-4
        self.batch_size = 32                  # Small batch
        self.buffer_size = 5000               # Small buffer
        self.learning_starts = 10             # Start learning quickly
        
        # Safety parameters
        self.safety_weight = 0.1
        self.barrier_margin = 0.1
        
        # Logging
        self.log_dir = "quick_logs"
        self.tensorboard_log = "quick_tb_logs"
        
        # Device
        self.device = "cpu"

# For immediate use in scripts
def create_quick_config():
    """Create and return quick training config"""
    return QuickTrainingConfig()
