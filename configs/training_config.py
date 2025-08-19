"""
Basic training configuration for the multi-agent system
"""

class TrainingConfig:
    """Configuration for training multi-agent system"""
    
    def __init__(self):
        # Training parameters
        self.total_timesteps = 10000
        self.eval_freq = 1000
        self.save_freq = 2000
        
        # Behavior cloning parameters
        self.bc_learning_rate = 1e-3
        self.bc_epochs = 50
        self.bc_batch_size = 32
        
        # Agent parameters
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.buffer_size = 100000
        
        # Safety parameters
        self.safety_weight = 0.1
        self.barrier_margin = 0.1
        
        # Logging
        self.log_dir = "logs"
        self.tensorboard_log = "tb_logs"
        
        # Device
        self.device = "cpu"  # Force CPU to avoid MPS issues
