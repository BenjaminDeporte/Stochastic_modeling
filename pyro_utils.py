# Some utilities for KL annealing

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList

import pyro.poutine as poutine

from pyro.infer.enum import get_importance_trace
# from pyro.infer.elbo import get_approx_kl_loss

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



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
    
    
    
def plot_sde_samples(ts, ys, title=None):
    """
    Utility functions to plot the sampled SDE solutions
    """
    fig, ax = plt.subplots(figsize=(10,6))
    n_points = ts.size()[0]
    n_paths = ys.size()[1]
    
    for i, y in enumerate(ys.permute(1,0,2)):  # iterate over paths
        ax.plot(ts.detach().cpu().numpy(), y.detach().cpu().numpy(), lw=1, alpha=1.0, label=f'Path {i+1}' if i<10 else None)  # plot each path
    
    if title is None:
        title = f"DATA (SDE): \n{n_paths} Sampled path(s) of the OSDE ({n_points:.0f} points)"
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.legend()
    ax.grid()
        
    return fig, ax






def display_diag_MCMC_Numpyro(diags):

    # Extract all 'n_eff' tensors for example : number of effective samples from the Markov chains
    n_eff_list = []
    for rv_name, rv_diagnostics in diags.items():
        
        # We only care about n_eff for the parameters we are sampling (weights, biases, sigma)
        # Exclude auxiliary or internal diagnostics if they exist.
        if 'n_eff' in rv_diagnostics and rv_name not in  ['_last_state', 'potential_energy']:
            
            # n_eff might be a tensor if the site is multidimensional (like a weight matrix)
            n_eff_tensor = rv_diagnostics['n_eff']
            
            # Flatten the tensor and add all its elements to our list
            n_eff_list.append(n_eff_tensor)

    # Concatenate all n_eff tensors into a single 1D tensor
    all_n_eff = np.array(n_eff_list)

    # Compute the average and minimum ESS
    average_n_eff = all_n_eff.mean().item()
    min_n_eff = all_n_eff.min().item()

    print(f"Total number of stochastic parameters analyzed: {len(all_n_eff)}")
    print(f"Overall Average Effective Sample Size (ESS): {average_n_eff:.2f}")
    print(f"Minimum Effective Sample Size (ESS): {min_n_eff:.2f}")
    
    
    
    
def display_posterior_samples(numpyro_posterior_samples, lambda_gt, sigma_gt):
    
    N_BINS=100

    lambda_samples = np.array(numpyro_posterior_samples.get("log_lambda"))
    sigma_samples = np.array(numpyro_posterior_samples.get("log_sigma"))

    lambda_samples = np.exp(lambda_samples)
    hist_counts, bin_edges = np.histogram(lambda_samples, bins=N_BINS)
    lambda_map = bin_edges[hist_counts.argmax()]  # approximate MAP

    sigma_samples = np.exp(sigma_samples)
    hist_counts, bin_edges = np.histogram(sigma_samples, bins=N_BINS)
    sigma_map = bin_edges[hist_counts.argmax()]  # approximate MAP

    print(f'lambda :')
    print(f'\tground truth : \t{lambda_gt:.3e}')
    # print(f'\tMLE : \t\t{lambda_ml:.3e}')
    print(f'\tMAP : \t\t{lambda_map:.3e}')
    print(f'sigma :')
    print(f'\tground truth : \t{sigma_gt:.3e}')
    # print(f'\tMLE : \t\t{sigma_ml:.3e}')
    print(f'\tMAP : \t\t{sigma_map:.3e}')

    fig, ax = plt.subplots(figsize=(20,6), nrows=1, ncols=3)

    sns.histplot(lambda_samples, bins=N_BINS, kde=True, label=f'lambda', ax=ax[0])
    ymax=ax[0].get_ylim()[1]*1.1
    ax[0].vlines(lambda_gt, ymin=0, ymax=ymax, color='green', label='ground truth')
    # ax[0].vlines(lambda_ml, ymin=0, ymax=ymax, color='blue', linestyles='--', label='MLE')
    ax[0].vlines(lambda_map, ymin=0, ymax=ymax, color='black', linestyles='--', label='MAP')
    ax[0].set_title(f'lambda')
    ax[0].legend()
    ax[0].grid()

    sns.histplot(sigma_samples, bins=N_BINS, kde=True, label=f'sigma', ax=ax[1])
    ymax=ax[1].get_ylim()[1]*1.1
    ax[1].vlines(sigma_gt, ymin=0, ymax=ymax, color='green', label='ground truth')
    # ax[1].vlines(sigma_ml, ymin=0, ymax=ymax, color='blue', linestyles='--', label='MLE')
    ax[1].vlines(sigma_map, ymin=0, ymax=ymax, color='black', linestyles='--', label='MAP')
    ax[1].set_title(f'sigma')
    ax[1].legend()
    ax[1].grid()

    ax[2].scatter(lambda_samples, sigma_samples, marker='.', alpha=0.5)
    ax[2].scatter(lambda_gt, sigma_gt, marker='x', s=50.0, color='green', label='ground truth')
    # ax[2].scatter(lambda_ml, sigma_ml, marker='x', s=50.0, color='blue', label='MLE')
    ax[2].scatter(lambda_map, sigma_map, marker='x', s=50.0, color='black', label='MAP')
    ax[2].set_title(f'lambda v sigma')
    ax[2].set_xlabel('lambda')
    ax[2].set_ylabel('sigma')
    ax[2].legend()
    ax[2].grid()

    return fig, ax