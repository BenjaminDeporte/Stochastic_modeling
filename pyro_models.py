import pyro
import pyro.distributions as dist
import torch

def pyro_model(t, X=None):
    # Pyro model to learn parameters
    # inputs:
    #   t (torch.tensor) : (N_POINTS,) : this is the tensor of time steps
    #   X (torch.tensor, optional) : (N_POINTS,) : the tensor of values at times t
    
    # number of points
    N = len(t)
    
    # priors
    # NB : we posit log_Normal distributions on lambda and sigma for positivity
    log_lambda = pyro.sample("log_lambda", dist.Normal(0., 1.))  
    log_sigma = pyro.sample("log_sigma", dist.Normal(0.,1.))
    
    # this is the prior for x_0 and also the first value for the likelihood calculation
    
    # compute likelihood
    lambda_ = torch.exp(log_lambda)
    sigma = torch.exp(log_sigma)
    # x_0 prior
    x0_sd = sigma / torch.sqrt(2 * lambda_)
    x_current = pyro.sample("x_0", dist.Normal(0.,x0_sd), obs=X[0])
    # for transition
    a = torch.exp(-lambda_ * (t[1:] - t[:-1]))   # shape (N-1,)
    sigma_2 = sigma**2 / (2 * lambda_) * (1 - torch.exp(-2 * lambda_ * (t[1:] - t[:-1])))  # shape (N-1, )
    # for numerical stability
    min_eps = 1e-8
    
    # likelihood
    for i in pyro.markov(range(1,N)):
        x_next = pyro.sample(
            f"x_{i}", 
            dist.Normal( a[i-1] * x_current , torch.sqrt(sigma_2[i-1])+min_eps ), obs=X[i])  # no observation noise on top of the latent
        x_current = x_next
        
        
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from numpyro.contrib.control_flow import scan
# from numpyro.infer import MCMC, NUTS

def numpyro_model(t, X=None):
    """
    OU process in NumPyro / JAX.
    Inputs:
        t : jnp.array of shape (N,)
        X : jnp.array of shape (N,), optional observations
    """
    N = t.shape[0]

    # priors on log parameters
    log_lambda = numpyro.sample("log_lambda", ndist.Normal(0., 1.))
    log_sigma  = numpyro.sample("log_sigma",  ndist.Normal(0., 1.))

    lambda_ = jnp.exp(log_lambda)
    sigma   = jnp.exp(log_sigma)

    dt = t[1:] - t[:-1]
    a  = jnp.exp(-lambda_ * dt)
    sigma2 = sigma**2 / (2 * lambda_) * (1 - jnp.exp(-2 * lambda_ * dt))
    scale = jnp.sqrt(sigma2 + 1e-8)

    # x0 prior
    x0_sd = sigma / jnp.sqrt(2 * lambda_)
    x0 = numpyro.sample("x_0", ndist.Normal(0., x0_sd), obs=None if X is None else X[0])

    # use scan for sequential transitions (more JAX-friendly)
    def ou_step(x_prev, inputs):
        a_i, scale_i, obs_i = inputs
        x_next = numpyro.sample("x_next", ndist.Normal(a_i * x_prev, scale_i), obs=obs_i)
        return x_next, x_next

    # prepare inputs for scan
    obs_vals = None if X is None else X[1:]
    inputs = (a, scale, obs_vals) if X is not None else (a, scale, [None]*(N-1))

    # scan along time
    _, xs = scan(ou_step, x0, inputs)