import torch
from torch import nn
import torch.nn.functional as F


class ICNNMap(nn.Module):
    """
    A neural network encoding an Input Convex Neural Network (ICNN).
    Its gradient can be accessed through the forwardGradient method.
    """
    
    def __init__(self, hidden_dim, depth,n_timesteps=None, final_time=None, **kwargs):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.n_timesteps = n_timesteps
        self.final_time = final_time
        self.dt = final_time / n_timesteps if final_time is not None else None
        self.dimension = 2
        self.current_time = 0
        self.X_t = []
        
        # First layer (fully connected)
        self.input_layer = nn.Linear(self.dimension, hidden_dim, bias=True)
        
        # Hidden layers (constraining non-negative weights for convexity)
        self.hidden_layers = nn.ModuleList()
        self.w_z_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            self.w_z_layers.append(nn.Linear(self.dimension, hidden_dim, bias=False))
            
        # Output layer (summing to ensure convexity)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        
        # Enforce non-negative weights
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            layer.weight.data.clamp_(0)
        nn.init.normal_(self.output_layer.weight, mean=0, std=0.1)
        self.output_layer.weight.data.clamp_(0)
        
    def forwardPotential(self, x):
        z = F.softplus(self.input_layer(x))
        for hidden_layer, w_z in zip(self.hidden_layers, self.w_z_layers):
          w = hidden_layer.weight.data  # Ensure non-negative weights
          w = w.clamp_(0)
          hidden_layer.weight.data = w
          z = F.softplus(hidden_layer(z) + w_z(x))  # Convex combination
        self.output_layer.weight.data.clamp_(0)
        output = self.output_layer(z)
        return output
    
    def forward(self, samples, t=None, detach=False, save=False):
        if t is not None: assert t == self.current_time
        if t==0:
            res = samples
        else:
            y = samples.detach().clone().requires_grad_(True)
            potential = self.forwardPotential(y)
            res = torch.autograd.grad(outputs=potential, inputs=y,
                                        grad_outputs=torch.ones_like(potential),
                                        create_graph=True)[0]
        if save:
            self.X_t.append(res)
        if detach:
            return res.detach().clone()
        return res
    
    def evaluate_up_to(self, sample, t, **kwargs):
        print('warning, evaluating disregarding of the sample')
        assert len(self.X_t) >= t
        return self.X_t[:t]