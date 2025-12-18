# separate class definition to run mutli chain MCMC

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroModuleList

# Bayesian Neural Network
class BayesianMLP(PyroModule):
    """
    The usual basic MLP class, but the Bayesian way
    """
    def __init__(self, n_layers=0, n_hidden_units=16, activation=nn.Tanh(), input_dim=1, output_dim=1, device='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.device = device    
        
        # set tensors explicitly on the device to enable multi-chain for MCMC
        loc_prior = torch.tensor(0., device=self.device)
        scale_prior = torch.tensor(1., device=self.device)
        noise_loc = torch.tensor(-2.0, device=self.device)
        noise_scale = torch.tensor(0.1, device=self.device)
        
        self.layers = PyroModuleList([])
        layer_input_dim = self.input_dim
        for _ in range(self.n_layers+1):
            bayesian_fc = PyroModule[nn.Linear](layer_input_dim, self.n_hidden_units)
            bayesian_fc.weight = PyroSample(dist.Normal(loc_prior,scale_prior).expand(bayesian_fc.weight.shape).to_event(bayesian_fc.weight.dim()))
            bayesian_fc.bias = PyroSample(dist.Normal(loc_prior, scale_prior).expand(bayesian_fc.bias.shape).to_event(bayesian_fc.bias.dim()))
            self.layers.append(bayesian_fc)
            layer_input_dim = self.n_hidden_units
            
        # last layer
        self.last_layer = PyroModule[nn.Linear](self.n_hidden_units, self.output_dim) 
        self.last_layer.weight = PyroSample(dist.Normal(loc_prior,scale_prior).expand(self.last_layer.weight.shape).to_event(self.last_layer.weight.dim()))
        self.last_layer.bias = PyroSample(dist.Normal(loc_prior,scale_prior).expand(self.last_layer.bias.shape).to_event(self.last_layer.bias.dim()))     
        
        # noise term
        self.log_sigma = PyroSample(dist.Normal(noise_loc,noise_scale))
        
        # In Pyro, when you use PyroSample as an attribute of a PyroModule, 
        # the attribute name itself (self.log_sigma in this case) becomes the name of the stochastic site in the model's trace. 
        # Therefore, you should not pass the name as a string argument to PyroSample.   
        
    def forward(self,x, y=None, **kwargs):
        x = x.reshape(-1,1)
        for layer in self.layers:
            x = self.activation(layer(x))
        mu = self.last_layer(x).squeeze()
        
        sigma = torch.exp(self.log_sigma)
        
        with pyro.plate("data", len(x)):
            pyro.sample("observations", dist.Normal(mu, sigma), obs=y)
        return mu
    
    def __repr__(self):
        msg = f'object MLP - input dim = {self.input_dim}, output_dim = {self.output_dim}, num layers = {self.n_layers}, hidden units = {self.n_hidden_units}\n'
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                msg += f"name = {name}, module = {module}\n"
    
        return msg