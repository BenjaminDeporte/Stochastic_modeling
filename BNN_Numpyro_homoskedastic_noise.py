#------------------------------------------------------------------
#
#  BNN CSR Regression avec Numpyro
#
#------------------------------------------------------------------

#------------------------------------------------------------------
# IMPORTS
#------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import summary
from jax import vmap

# from numpyro.infer.util import r_hat, effective_sample_size

#------------------------------------------------------------------
# BNN NumpPyro Model
#------------------------------------------------------------------

#------------------------------------------------------------------
# Network architecture
#------------------------------------------------------------------

architecture = {
    "D_X" : 1,
    "D_Y" : 1,
    "N_LAYERS" : 4,  # number of hidden layers size D_H neurons
    "D_H" : 64,
}

#------------------------------------------------------------------
# deterministic forward
#------------------------------------------------------------------

def forward(X, weights):
    # Deterministic forward pass
    #
    # Inputs:
    #   X : jnp.array (N, D_X)
    #   weights = {
    #       "input": {
    #           "W": W1,
    #           "B": B1
    #       },
    #       "hidden": [
    #           {"W": W_hidden_1, "B": B_hidden_1},
    #           {"W": W_hidden_2, "B": B_hidden_2},
    #           ...
    #       ],
    #       "mu": {
    #           "W": W_mu,
    #           "B": B_mu
    #       }
    #   }
    
    # activation function
    activation = jnp.tanh
    
    # get number of hidden layers
    n_layers = len(weights.get('hidden'))
    
    # forward pass
    X = activation(X @ weights["input"]["W"] + weights["input"]["B"])
    for l in range(n_layers):
        X = activation(X @ weights["hidden"][l]["W"] + weights["hidden"][l]["B"])
        
    W_mu = weights["mu"]["W"]
    B_mu = weights["mu"]["B"]
    mu = X @ W_mu + B_mu
    
    return mu # (N, D_Y)

#-------------------------------------------------------------------
# probabilistic model declaration for NN weights and observation noise
# NB : the probabilistic model depends on the architecture that is passed
#-------------------------------------------------------------------

# def NumpyroBNN(X, D_Y=1, y=None, D_H=32):
def NumpyroBNN(X, architecture, y=None):

    """
    BNN Numpyro model definition for MCMC sampling
    Inputs:
        X : jnp.array (N, D_X) : features
        y (optional) : jnp.array (N, D_Y) : target
        architecture = {
            "D_X" : 1,
            "D_Y" : 1,
            "N_LAYERS" : 2,  # number of hidden layers size D_H neurons
            "D_H" : 32,
        }
    """
    
    # dimensions for the MLP
    N = X.shape[0]  # number of points
    # D_X = X.shape[1]  # features dimension
    D_X = architecture.get("D_X", None)
    D_Y = architecture.get("D_Y", None)
    D_H = architecture.get("D_H", None)
    n_layers = architecture.get("N_LAYERS", None)
    
    # checks
    if D_X != X.shape[1]:
        raise NameError(f'Features dimension mismatch in NumpyroBNN - inputs with dim {X.shape[1]} vs architecture D_X {D_X}')
    
    # enforce y.shape = (N, 1) (vector) if y.shape = (N, ) is provided
    # target is the new reshaped y
    if y is not None:
        if y.ndim == 1:
            target = jnp.reshape(y, (-1,1))
        else:
            target = y
        assert D_Y == target.shape[1], "the shape of y and D_Y do not match"
        # D_Y = target.shape[1]
    else:
        target = None
    # NB  -there is a potential bug here if the model is trained with y=None (D_Y=1) and inference is run with D_Y > 1
    
    # priors    
    WEIGHT_PRIOR_STD = 1.0
    BIAS_PRIOR_STD = 0.1
    LOGSIG_PRIOR_MEAN = -2.0
    LOGSIG_PRIOR_STD = 0.1
    
    weights = {}
    # -- first layer : X (N, D_X) @ W1 (D_X, D_H) + B1 (D_H) => X2 (N, D_H)
    W1 = numpyro.sample("W1", dist.Normal(0.0, WEIGHT_PRIOR_STD/jnp.sqrt(D_X)).expand([D_X, D_H]).to_event(2))
    B1 = numpyro.sample("B1", dist.Normal(0.0, BIAS_PRIOR_STD).expand([D_H]).to_event(1))
    weights["input"] = {}
    weights["input"]["W"] = W1
    weights["input"]["B"] = B1
    
    # next layers
    weights["hidden"] = []
    for l in range(n_layers):
        W = numpyro.sample(f"W_hidden_{l+1}", dist.Normal( 0.0, WEIGHT_PRIOR_STD/jnp.sqrt(D_H)).expand([D_H, D_H]).to_event(2))
        B = numpyro.sample(f"B_hidden_{l+1}", dist.Normal( 0.0, BIAS_PRIOR_STD).expand([D_H]).to_event(1))
        weights["hidden"].append({"W" : W, "B" : B})
        
    # last layer
    W_mu = numpyro.sample("W_mu", dist.Normal(0.0, WEIGHT_PRIOR_STD/jnp.sqrt(D_H)).expand([D_H, D_Y]).to_event(2))
    B_mu = numpyro.sample("B_mu", dist.Normal(0.0, BIAS_PRIOR_STD).expand([D_Y]).to_event(1))
    weights["mu"] = {"W":W_mu, "B":B_mu}

    # homoskedatic noise
    log_sigma = numpyro.sample("log_sigma", dist.Normal(LOGSIG_PRIOR_MEAN, LOGSIG_PRIOR_STD).expand([D_Y]).to_event(1))
    sigma = jnp.exp(log_sigma) + 1e-6 # (D_Y,)
    
    # compute the forward pass
    mu = forward(X, weights)   # (N, D_Y)
    
    # likelihood (NO plate)
    # numpyro.sample(
    #     "ys",
    #     dist.Normal(mu, sigma).to_event(1), # batch = N, event = D_Y
    #     obs=target
    # )
    
    # equivalently with plate:
    # each plate dimension must correspond to exactly one unused batch dimension. 
    # sigma = jnp.broadcast_to(sigma, mu.shape) # manual broadcasting, unecessary
    # mu, sigma : batch = (N, D_Y), event = ()
    with numpyro.plate("data", N):
        # inside plate : remaining batch = (D_Y) (took N out), event = ()
        # to_event(1) to get to : batch =(), event =(D_Y)
        numpyro.sample("ys", dist.Normal(mu, sigma).to_event(1), obs=target)


#---------------------------------------------------------------------------------

def execute_main():
    
    #------------------------------------------------------------------
    # SET UP
    #------------------------------------------------------------------

    def seed_everything(seed=42):
        """
        Set seed for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    seed_everything()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        # if using a GPU (CUDA) and num_chains > 1
        # torch.multiprocessing.set_start_method('spawn', force=True)
    else:
        device = torch.device('cpu')
        
    # NB Warning : setting the default_device to CUDA creates a device conflict
    # when using a DataLoader, as it uses a CPU-generator for shuffling
    torch.set_default_device(device)
    print(f"Using {device}")

    torch.set_default_dtype(torch.float32)

    if device.type == 'cuda':
        print('GPU Name:', torch.cuda.get_device_name(0))
        print('Total GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
        
    #------------------------------------------------------------------
    # LOAD DATA - simple sine wave
    #------------------------------------------------------------------
    
    N_POINTS = 250
    NOISE = 0.15
    SCALE = 1.0
    # -- train / reconstruction
    X = np.linspace(0.0,1.0,N_POINTS)
    y = SCALE * X * np.sin(16*np.pi*X) + np.random.normal(0.0,NOISE,size=N_POINTS)
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    
    # -- test / forecast
    X_test = np.linspace(0.0,2.0,2*N_POINTS)
    y_test = SCALE * X_test * np.sin(16*np.pi*X_test) + np.random.normal(0.0,NOISE,size=2*N_POINTS)
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    #------------------------------------------------------------------
    # PRIOR PREDICTIVE CHECKS
    #------------------------------------------------------------------
    # NB the sampling from the numpyro model is only possible
    # within a MCMC, Predictive or handlers.seed(...) context
    # that provide a RNG key
    # then, NumpyroBNN will actually sample stuff
    # this is a different behavior from Pyro
    
    NS = 20
    predictive = Predictive(
        NumpyroBNN,
        num_samples=NS
    )
    
    # produces the first key
    rng_key = random.PRNGKey(68)
    
    samples = predictive(
        rng_key,
        X=X_test,
        architecture=architecture,
        y=None
    )
    mu_prior = samples["ys"]
    
    print(f"Prior empirical STD : {mu_prior.std(axis=0).mean()}")
    
    # fig, ax = plt.subplots(figsize=(12,6))
    # for i in range(NS):
    #     ax.plot(mu_prior[i].squeeze(), color='blue', alpha=0.2)
    # ax.grid()
    # # ax.legend()
    # fig.suptitle(f'Priors check')
    # plt.show()
    
    #------------------------------------------------------------------------
    # RUNNING MCMC !!
    #------------------------------------------------------------------------
    
    N_WARMUP = 300
    N_SAMPLES = 100
    
    print(f'\nTesting BNN and MCMC on a sine wave toy set - noise level = {NOISE} - WARM_UPs {N_WARMUP}, N_SAMPLES {N_SAMPLES}')
    assert architecture is not None, "Must provide an architecture for the probabilistic model !"
    print(f'Model architecture : {architecture} - HOMOSKEDASTIC noise')
    
    kernel = NUTS(NumpyroBNN)
    MCMC_runner = MCMC(
        sampler=kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=1   # on GPU, only sequential runs of the different chains
    )
    MCMC_runner.run(
        random.PRNGKey(0),
        # arguments passed on to the instantiated model
        X = jnp.array(X),
        y = jnp.array(y),
        architecture = architecture
    )
    
    #------------------------------------------------------------------------------
    #  Diags
    #------------------------------------------------------------------------------
    
    # look at n_eff (number of effective samples) and R_hat
    print(f'MCMC quality summary')
    MCMC_runner.print_summary()
    
    #------------------------------------------------------------------------------
    # Checking reconstruction
    #------------------------------------------------------------------------------
    
    posterior_samples = MCMC_runner.get_samples()

    # extract all posteriors into a dictionary structure
    sampled_weights = {}
    sampled_weights["input"] = {"W": posterior_samples["W1"] , "B": posterior_samples["B1"]}
    sampled_weights["hidden"] = []
    n_layers = architecture["N_LAYERS"]
    for l in range(n_layers):
        sampled_weights["hidden"].append({"W":posterior_samples[f'W_hidden_{l+1}'],"B":posterior_samples[f'B_hidden_{l+1}']})
    sampled_weights["mu"] = {"W":posterior_samples["W_mu"], "B":posterior_samples["B_mu"]}

    # mapping the exact structure of sampled_weights with 0's for vmap
    in_axes_tree = jax.tree_util.tree_map(lambda _: 0, sampled_weights)

    # use vmap to run deterministic forward pass predictions on each of the sampled neural nets
    mu_preds = vmap(forward, in_axes=(None, in_axes_tree))(X_test, sampled_weights)
    log_sigma = posterior_samples["log_sigma"]
    sigma_preds = jnp.exp(log_sigma)

    # compute macros and display
    print(f'Running predictions on X_test with posterior samples (N_WARMUP {N_WARMUP}, N_SAMPLES {N_SAMPLES})')
    print(f'\tInputs X_test {X_test.shape}')
    print(f'\tMu samples : {mu_preds.shape}')
    print(f'\tSigma samples : {sigma_preds.shape}')
    
    y_mean = np.mean(mu_preds, axis=0)
    y_low_ci = (y_mean - 1.96 * jnp.mean(sigma_preds, axis=0)).squeeze()
    y_high_ci = (y_mean + 1.96 * jnp.mean(sigma_preds, axis=0)).squeeze()
    
    fig,ax = plt.subplots(figsize=(12,6))
    ax.scatter(X, y, marker='.', color='green', label='ground truth (training)')
    ax.scatter(X_test, y_test, marker='.', color='orange', label='test set')
    ax.scatter(X_test, y_mean, marker='.', color='blue', label='moyenne predictions')
    ax.fill_between(
        x=X_test.squeeze(),
        y1=y_low_ci,
        y2=y_high_ci,
        color='blue',
        alpha=0.2,
        label='95% CI'
    )
    ax.legend()
    ax.grid()
    plt.show()



#----------------------------------------------------------------------------------

if __name__ == "__main__":
    matplotlib.use("QtAgg")  # or "qtagg" with recent matplotlib
    print("Running main")
    execute_main()