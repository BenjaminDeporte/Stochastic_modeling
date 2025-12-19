import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
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
    log_lambda = numpyro.sample("log_lambda", dist.Normal(0., 1.))
    log_sigma  = numpyro.sample("log_sigma",  dist.Normal(0., 1.))

    lambda_ = jnp.exp(log_lambda)
    sigma   = jnp.exp(log_sigma)

    dt = t[1:] - t[:-1]
    a  = jnp.exp(-lambda_ * dt)
    sigma2 = sigma**2 / (2 * lambda_) * (1 - jnp.exp(-2 * lambda_ * dt))
    scale = jnp.sqrt(sigma2 + 1e-8)

    # x0 prior
    x0_sd = sigma / jnp.sqrt(2 * lambda_)
    x0 = numpyro.sample("x_0", dist.Normal(0., x0_sd), obs=None if X is None else X[0])

    # use scan for sequential transitions (more JAX-friendly)
    def ou_step(x_prev, inputs):
        a_i, scale_i, obs_i = inputs
        x_next = numpyro.sample("x_next", dist.Normal(a_i * x_prev, scale_i), obs=obs_i)
        return x_next, x_next

    # prepare inputs for scan
    obs_vals = None if X is None else X[1:]
    inputs = (a, scale, obs_vals) if X is not None else (a, scale, [None]*(N-1))

    # scan along time
    _, xs = scan(ou_step, x0, inputs)
    
    
    
def ou_gp_model(t, X):
    log_lambda = numpyro.sample("log_lambda", dist.Normal(0., 1.))
    log_sigma  = numpyro.sample("log_sigma", dist.Normal(0., 1.))

    lam = jnp.exp(log_lambda)
    sig = jnp.exp(log_sigma)

    # pairwise |t_i - t_j|
    dt = jnp.abs(t[:, None] - t[None, :])

    cov = (sig**2 / (2 * lam)) * jnp.exp(-lam * dt)

    numpyro.sample(
        "X",
        dist.MultivariateNormal(
            loc=jnp.zeros_like(t),
            covariance_matrix=cov
        ),
        obs=X
    )