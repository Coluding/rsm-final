"""
Unified metric logging interface for tensorboard and wandb.
"""
import logging
from typing import Optional, Dict, Any, Union
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricLogger:
    """
    Unified logging interface that can log to tensorboard and/or wandb.
    """
    
    def __init__(
        self,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        tensorboard_comment: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the metric logger.
        
        Args:
            use_tensorboard: Whether to enable tensorboard logging
            use_wandb: Whether to enable wandb logging
            tensorboard_comment: Comment for tensorboard run
            wandb_project: Wandb project name
            wandb_entity: Wandb entity/team name
            wandb_run_name: Wandb run name
            wandb_config: Configuration dict for wandb
        """
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.tensorboard_writer = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize tensorboard
        if self.use_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                self.logger.warning("tensorboard is not available. Install it with 'pip install tensorboard'")
                self.use_tensorboard = False
            else:
                self.tensorboard_writer = SummaryWriter(comment=tensorboard_comment)
        
        # Initialize wandb
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                self.logger.warning("wandb is not available. Install it with 'pip install wandb'")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project or "rl-training",
                    entity=wandb_entity,
                    name=wandb_run_name,
                    config=wandb_config,
                    reinit=True
                )
    
    def add_scalar(self, tag: str, scalar_value: Union[float, int], global_step: Optional[int] = None):
        """Log a scalar value."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, scalar_value, global_step)
        
        if self.use_wandb:
            log_dict = {tag: scalar_value}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Union[float, int]], global_step: Optional[int] = None):
        """Log multiple scalar values."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        
        if self.use_wandb:
            log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
    
    def add_figure(self, tag: str, figure: plt.Figure, global_step: Optional[int] = None, close: bool = True):
        """Log a matplotlib figure."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_figure(tag, figure, global_step, close=False)
        
        if self.use_wandb:
            log_dict = {tag: wandb.Image(figure)}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
        
        if close:
            plt.close(figure)
    
    def add_histogram(self, tag: str, values, global_step: Optional[int] = None, bins: str = 'tensorflow'):
        """Log a histogram."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, global_step, bins)
        
        if self.use_wandb:
            log_dict = {tag: wandb.Histogram(values)}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
    
    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None):
        """Log text."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_text(tag, text_string, global_step)
        
        if self.use_wandb:
            log_dict = {tag: text_string}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
    
    def add_image(self, tag: str, img_tensor, global_step: Optional[int] = None, dataformats: str = 'CHW'):
        """Log an image tensor."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_image(tag, img_tensor, global_step, dataformats)
        
        if self.use_wandb:
            # Convert tensor to wandb image format
            if hasattr(img_tensor, 'cpu'):
                img_tensor = img_tensor.cpu()
            log_dict = {tag: wandb.Image(img_tensor)}
            if global_step is not None:
                log_dict["step"] = global_step
            wandb.log(log_dict)
    
    def flush(self):
        """Flush any pending logs."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.flush()
    
    def close(self):
        """Close the loggers."""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
