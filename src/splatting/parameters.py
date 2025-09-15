class OptimizationParams(object):
    def __init__(self):
        self.iterations = 30_000
        self.position_lr_init = 0.0001
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.feature_extra_lr = 0.01
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.005 # 0.001
        self.polarity_lr = 0.01
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002