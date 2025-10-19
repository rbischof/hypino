# This file is adapted from DeepXDE: A library for scientific machine learning and physics-informed learning.
# Original source: https://github.com/lululxvi/deepxde
# 
# Original author: Lu Lu

"""Initial conditions and boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "PeriodicBC",
    "OperatorBC",
    "PointSetBC",
    "PointSetOperatorBC",
    "IC",
]

from .boundary_conditions import (
    BC,
    DirichletBC,
    NeumannBC,
)
from .initial_conditions import IC
