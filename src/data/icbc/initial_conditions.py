# This file is adapted from DeepXDE: A library for scientific machine learning and physics-informed learning.
# Original source: https://github.com/lululxvi/deepxde
# 
# Original author: Lu Lu

"""Initial conditions."""

__all__ = ["IC"]
import torch

from src.data.utils import compute_derivatives, to_tensor
from .boundary_conditions import BC

class IC(BC):
    """Initial conditions for PDEs."""
    def __init__(self, geom, func, derivative_order=0):
        super().__init__(geom)
        self.func = func
        self.derivative_order = derivative_order
        
    def error(self, xy, predictions):
        # Zeroth order initial condition, y(t0) = func(x)
        if self.derivative_order == 0:
            return predictions - self(to_tensor(xy))
        # First order initial condition, dy/dt(t0) = func(x)
        elif self.derivative_order == 1:
            assert isinstance(predictions, tuple) and len(predictions) == 2
            x, y = to_tensor(xy[:, :1], requires_grad=True), to_tensor(xy[:, 1:2], requires_grad=True)
            values = self(torch.cat([x, y], dim=-1))
            dvalues = self.time_derivative(x, y, values)
            return predictions[0] - values, predictions[1] - dvalues
        else:
            raise ValueError('Unsupported derivative order for initial conditions.')

    def time_derivative(self, x, y, outputs, concat_normals=False):
        dudy = compute_derivatives(
            in_var_map={'x': x, 'y': y}, 
            out_var_map={'u': outputs},
            derivatives={'u': [[('y', 1)]]},
        )['uy']
        if concat_normals:
            return torch.cat([dudy, torch.zeros_like(dudy), torch.ones_like(dudy)], dim=-1)
        return dudy

    def __call__(self, xy, concat_normals=False):
        if self.derivative_order == 0:
            return self.func(to_tensor(xy))
        elif self.derivative_order == 1:
            x, y = to_tensor(xy[:, :1], requires_grad=True), to_tensor(xy[:, 1:2], requires_grad=True)
            values = self.func(torch.cat([x, y], dim=-1))
            return values.detach().numpy(), self.time_derivative(x, y, values, concat_normals).detach().numpy()
        else:
            raise ValueError('Unsupported derivative order for initial conditions.')