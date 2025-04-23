from lightning import Trainer, LightningModule
import torch
import matplotlib.pyplot as plt
from aim import Distribution
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CometLogger



class GradientNormLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        # Log gradient norms for each parameter
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                pl_module.log(f'grad_norm_{name}', grad_norm)