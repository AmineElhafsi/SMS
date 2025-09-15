import numpy as np

def get_exponential_lr_scheduler(
    lr_init: float, 
    lr_final: float, 
    lr_delay_steps: int = 0, 
    lr_delay_mult: float = 1.0, 
    max_steps: int = int(1e6),
):
    """
    Adapted from https://github.com/VladimirYugay/Gaussian-SLAM/blob/main/src/utils/gaussian_model_utils.py
    and Plenoxels.

    Returns exponential decay learning rate scheduler. Initial lr_delay_steps are scaled by a delay_rate
    which is on a reverse cosine decay schedule.
    
    Args:
        lr_init (float): Initial learning rate.
        lr_final (float): Final learning rate.
        lr_delay_steps (int): Number of steps to delay learning rate decay.
        lr_delay_mult (float): Multiplier for delayed learning rate decay.
        max_steps (int): Maximum number of steps.
        
    Returns:
        scheduler (function): Learning rate scheduler.
    """

    def scheduler(step):
        # set learning rate to 0 if step is negative or if initial and final learning rates are 0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        
        # compute learning rate delay multiplier
        if lr_delay_steps > 0:
            p = np.clip(step / lr_delay_steps, 0, 1)
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * p)
        else:
            delay_rate = 1
        
        # compute learning rate
        p = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp((1 - p) * np.log(lr_init) + p * np.log(lr_final))
        return log_lerp * delay_rate
    
    return scheduler