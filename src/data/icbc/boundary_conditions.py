# This file is adapted from DeepXDE: A library for scientific machine learning and physics-informed learning.
# Original source: https://github.com/lululxvi/deepxde
# 
# Original author: Lu Lu

"""Boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "NeumannBC",
    "OperatorBC",
    "PeriodicBC",
    "PointSetBC",
    "PointSetOperatorBC",
    "RobinBC",
]

from abc import ABC, abstractmethod

import torch
from src.data.utils import compute_derivatives, to_tensor

class BC(ABC):
    """Boundary condition base class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom):
        self.geom = geom
        self.boundary_normal = self.geom.boundary_normal

    def filter(self, xy):
        return xy[self.geom.on_boundary(xy, self.geom.geom.on_boundary(xy))]

    def collocation_points(self, xy):
        return self.filter(xy)

    def normal_derivative(self, x, y, outputs, concat_normals=False):
        n = to_tensor(self.boundary_normal(torch.cat([x, y], dim=-1).detach().cpu().numpy())).to(x.device)
        d = compute_derivatives(
            in_var_map={'x': x, 'y': y}, 
            out_var_map={'u': outputs},
            derivatives={'u': [[('x', 1)], [('y', 1)]]},
        )
        d = torch.cat([d['ux'], d['uy']], dim=-1)
        n_d = (d * n).sum(axis=-1, keepdim=True)
        if concat_normals:
            return torch.cat([n_d, n], dim=-1)
        return n_d

    @abstractmethod
    def error(self, xy, outputs):
        """Returns the loss."""


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func):
        super().__init__(geom)
        self.func = func

    def __call__(self, xy):
        return self.func(to_tensor(xy)).detach().cpu().numpy()

    def error(self, xy, outputs):
        return outputs - self(xy)


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func):
        super().__init__(geom)
        self.func = func

    def __call__(self, xy, concat_normals=False):
        x, y = to_tensor(xy[:, :1], requires_grad=True), to_tensor(xy[:, 1:2], requires_grad=True)
        outputs = self.func(torch.cat([x, y], dim=-1))
        return self.normal_derivative(x, y, outputs, concat_normals).detach().cpu().numpy()

    def error(self, xy, outputs):
        x, y = to_tensor(xy[:, :1], requires_grad=True), to_tensor(xy[:, 1:2], requires_grad=True)
        values = self.func(torch.cat([x, y], dim=-1))
        return self.normal_derivative(x, y, outputs) - values