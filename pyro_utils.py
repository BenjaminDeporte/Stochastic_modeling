# Some utilities for KL annealing

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList

import pyro.poutine as poutine

from pyro.infer.enum import get_importance_trace
# from pyro.infer.elbo import get_approx_kl_loss



class AnnealingTrace_ELBO(pyro.infer.Trace_ELBO):
    """
    A Trace_ELBO variant that implements a dynamic KL annealing schedule.
    """
    def __init__(self, kl_annealing_steps, final_kl_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_annealing_steps = kl_annealing_steps
        self.step = 0 # Initialize step counter
        self.final_kl_weight = final_kl_weight
        
    def _linear_kl_annealing(self, step, total_annealing_steps):
        """
        Linearly ramps the KL weight from 0.0 to 1.0 over total_annealing_steps.
        """
        # Ensure the weight doesn't exceed 1.0
        return min(1.0, step / total_annealing_steps)

    def _get_kl_weight(self, *args, **kwargs):
        """
        Overrides the standard method to return the scheduled weight.
        """
        # Call the scheduler function
        weight = self._linear_kl_annealing(self.step, self.kl_annealing_steps) * self.final_kl_weight
        return weight

    def loss(self, *args, **kwargs):
        # 💡 Increment the step counter before computing the loss
        self.step += 1
        return super().loss(*args, **kwargs)