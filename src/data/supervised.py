import torch
import numpy as np

from src.data.icbc.initial_conditions import IC
from src.data.icbc.boundary_conditions import DirichletBC, NeumannBC
from src.data.datasets import RandomFunctionDataset
from src.data.geometry.timedomain import GeometryXTime
from src.data.utils import (
    ALL_DERIVATIVES,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    NUM_POINTS_BOUNDARY,
    NUM_POINTS_BOUNDARY_DENSE,
    NUM_POINTS_DOMAIN,
    RandomFunction,
    compile_diff_operator,
    classify_pde,
    compute_derivatives,
    encode_pde_str,
    generate_diff_operator,
    generate_random_domain,
    to_tensor,
    resize,
    WeightedInterpolator,
)


class SupervisedRandFunDataset(RandomFunctionDataset):
    """Dataset that generates random PDEs on‑the‑fly."""

    def __init__(
        self,
        size: int,
        grid_size: int = 224,
        init_grid_size: int = 224,
        max_abs_val: float = 10.0,
        min_std_val: float = 0.1,
        max_num_inner_boundaries: int = 4,
    ) -> None:
        super().__init__(size, grid_size, init_grid_size, max_abs_val, min_std_val, max_num_inner_boundaries)

    @staticmethod
    def _tensor_invalid(t: torch.Tensor, *, max_abs: float | None = None, min_std: float | None = None, max_val: float | None = None) -> bool:
        """Return True if t violates any numeric constraint."""
        if torch.isnan(t).any() or torch.isinf(t).any():
            return True
        if max_abs is not None and t.abs().amax() > max_abs:
            return True
        if max_val is not None and t.abs().amax() > max_val:
            return True
        if min_std is not None and t.std() < min_std:
            return True
        return False

    @staticmethod
    def _array_invalid(a: np.ndarray, *, max_abs: float | None = None, max_val: float | None = None) -> bool:
        """Same as :py:meth:`_tensor_invalid` but for numpy arrays."""
        if np.isnan(a).any() or np.isinf(a).any():
            return True
        if max_abs is not None and np.abs(a).max() > max_abs:
            return True
        if max_val is not None and np.abs(a).max() > max_val:
            return True
        return False

    def _is_sample_valid(
        self,
        *,
        u_domain: torch.Tensor,
        u_bound: torch.Tensor,
        f_domain: torch.Tensor,
        f_bound: torch.Tensor,
        f_grid: torch.Tensor,
        dir_boundc: np.ndarray,
        neu_boundc: np.ndarray,
    ) -> bool:
        """Return True if all sanity checks pass (identical to the old logic)."""
        # 1. Solution values / variation
        if self._tensor_invalid(u_domain, max_abs=self.max_abs_val, min_std=self.min_std_val):
            return False
        if self._tensor_invalid(u_bound, max_abs=self.max_abs_val):
            return False

        # 2. Forcing terms must stay within a generous finite range
        big_val = 1e4
        if self._tensor_invalid(f_domain, max_val=big_val):
            return False
        if self._tensor_invalid(f_bound, max_val=big_val):
            return False
        if self._tensor_invalid(f_grid, max_val=big_val):
            return False

        # 3. Boundary condition coefficients (numpy arrays)
        if self._array_invalid(dir_boundc, max_abs=self.max_abs_val):
            return False
        if self._array_invalid(neu_boundc, max_abs=1e2):
            return False

        return True

    def __getitem__(self, idx: int):  # noqa: C901  # (complex but unavoidable)
        """Generate one random PDE problem and associated learning sample."""

        # We loop until a numerically conformant sample is produced
        while True:
            # 1)  Define a random PDE operator (symbolic representation)
            diff_op_str, derivatives = generate_diff_operator()
            diff_op = compile_diff_operator(diff_op_str)
            pde_coeffs_dict = encode_pde_str(diff_op_str)
            diff_op_type = classify_pde(pde_coeffs_dict)

            # 2)  Draw a random analytic function u(x, y)
            u_fun = RandomFunction()

            # Prepare a (init_grid_size x init_grid_size) evaluation grid
            x_grid_np, y_grid_np = np.meshgrid(
                np.linspace(X_MIN, X_MAX, self.init_grid_size),
                np.linspace(Y_MAX, Y_MIN, self.init_grid_size),
            )
            x_grid = to_tensor(x_grid_np, requires_grad=True).reshape(-1, 1)
            y_grid = to_tensor(y_grid_np, requires_grad=True).reshape(-1, 1)
            xy_grid = torch.cat([x_grid, y_grid], dim=-1)

            # Evaluate u and its derivatives on the dense grid
            u_grid = u_fun(xy_grid)
            if u_grid.abs().amax() > self.max_abs_val:
                # Quick early‑exit (performance optimisation)
                continue

            ders_grid = compute_derivatives(
                in_var_map={"x": x_grid, "y": y_grid},
                out_var_map={"u": u_grid},
                derivatives=ALL_DERIVATIVES,
            )
            if any(d.abs().amax() > 1e4 for d in ders_grid.values()):
                continue

            f_grid = diff_op(ders_grid).reshape(self.init_grid_size, self.init_grid_size)
            if f_grid.abs().amax() > 1e4:
                continue

            # 3)  Random geometry (outer domain + optional inner holes)
            geom, domain, inner_boundaries = generate_random_domain(
                bbox=[X_MIN, Y_MIN, X_MAX, Y_MAX],
                elliptic=(diff_op_type == "Elliptic"),
                max_num_inner_boundaries=self.max_num_inner_boundaries,
            )

            # Points in the interior -------------------------------------------
            xy_domain = geom.random_points(NUM_POINTS_DOMAIN)
            x_domain = to_tensor(xy_domain[:, :1], requires_grad=True)
            y_domain = to_tensor(xy_domain[:, 1:2], requires_grad=True)
            u_domain = u_fun(torch.cat([x_domain, y_domain], dim=-1))
            if u_domain.abs().amax() > self.max_abs_val:
                continue
            ders_domain = compute_derivatives(
                in_var_map={"x": x_domain, "y": y_domain},
                out_var_map={"u": u_domain},
                derivatives=ALL_DERIVATIVES,
            )
            f_domain = diff_op(ders_domain).detach()
            if f_domain.abs().amax() > 1e4:
                continue

            # Boundary points --------------------------------------------------
            if isinstance(domain, GeometryXTime):
                xy_bound = np.concatenate(
                    [
                        geom.random_boundary_points(3 * NUM_POINTS_BOUNDARY_DENSE // 4),
                        domain.random_initial_points(NUM_POINTS_BOUNDARY_DENSE - 3 * NUM_POINTS_BOUNDARY_DENSE // 4),
                    ],
                    axis=0,
                )
            else:
                xy_bound = geom.random_boundary_points(NUM_POINTS_BOUNDARY_DENSE)

            # Prepare arrays for boundary conditions
            dir_boundc = np.zeros((len(xy_bound), 1))  # Dirichlet target value
            dir_bound = np.zeros((len(xy_bound), 1))   # 1 if Dirichlet active
            neu_boundc = np.zeros((len(xy_bound), 3))  # Neumann (value + normals)
            neu_bound = np.zeros((len(xy_bound), 1))   # 1 if Neumann active

            # --- Outer boundary ---------------------------------------------
            on_bc = domain.on_boundary(xy_bound)
            bc = DirichletBC(domain, u_fun)
            dir_boundc[on_bc] = bc(xy_bound[on_bc])
            dir_bound[on_bc] = 1.0

            # Neumann on outer boundary (25 % probability)
            if np.random.uniform() < 0.25:
                bc = NeumannBC(domain, u_fun)
                neu_boundc[on_bc] = bc(xy_bound[on_bc], concat_normals=True)
                neu_bound[on_bc] = 1.0

            # --- Initial conditions (time‑dependent domains) -----------------
            if diff_op_type == "First-Order":
                if isinstance(domain, GeometryXTime):
                    on_ic = domain.on_initial(xy_bound)
                    if np.random.uniform() < 0.75:
                        ic = IC(domain, u_fun, derivative_order=0)
                        dir_boundc[on_ic] = ic(xy_bound[on_ic])
                        dir_bound[on_ic] = 1.0
                    else:
                        ic = IC(domain, u_fun, derivative_order=1)
                        dir_boundc[on_ic], neu_boundc[on_ic] = ic(xy_bound[on_ic], concat_normals=True)
                        dir_bound[on_ic] = 1.0
                        neu_bound[on_ic] = 1.0
            elif diff_op_type != "Elliptic":
                on_ic = domain.on_initial(xy_bound)
                if diff_op_type == "Parabolic":
                    ic = IC(domain, u_fun, derivative_order=0)
                    dir_boundc[on_ic] = ic(xy_bound[on_ic])
                    dir_bound[on_ic] = 1.0
                elif diff_op_type == "Hyperbolic":
                    ic = IC(domain, u_fun, derivative_order=1)
                    dir_boundc[on_ic], neu_boundc[on_ic] = ic(xy_bound[on_ic], concat_normals=True)
                    dir_bound[on_ic] = 1.0
                    neu_bound[on_ic] = 1.0

            # --- Inner boundaries -------------------------------------------
            for inner_bound in inner_boundaries:
                on_bc = inner_bound.on_boundary(xy_bound)
                use_dirichlet = False
                if np.random.uniform() < 0.5:
                    bc = NeumannBC(inner_bound, u_fun)
                    neu_boundc[on_bc] = bc(xy_bound[on_bc], concat_normals=True)
                    neu_bound[on_bc] = 1.0
                else:
                    use_dirichlet = True
                if use_dirichlet or np.random.uniform() < 0.5:
                    bc = DirichletBC(inner_bound, u_fun)
                    dir_boundc[on_bc] = bc(xy_bound[on_bc])
                    dir_bound[on_bc] = 1.0

            # --- Sub‑sample boundary points for training ---------------------
            bound_point_ixs = np.arange(len(xy_bound))
            np.random.shuffle(bound_point_ixs)
            bound_point_ixs = bound_point_ixs[:NUM_POINTS_BOUNDARY]
            x_bound = to_tensor(xy_bound[bound_point_ixs, :1], requires_grad=True)
            y_bound = to_tensor(xy_bound[bound_point_ixs, 1:2], requires_grad=True)
            u_bound = u_fun(torch.cat([x_bound, y_bound], dim=-1))
            ders_bound = compute_derivatives(
                in_var_map={"x": x_bound, "y": y_bound},
                out_var_map={"u": u_bound},
                derivatives=ALL_DERIVATIVES,
            )
            f_bound = diff_op(ders_bound).detach()

            if not self._is_sample_valid(
                u_domain=u_domain,
                u_bound=u_bound,
                f_domain=f_domain,
                f_bound=f_bound,
                f_grid=f_grid,
                dir_boundc=dir_boundc,
                neu_boundc=neu_boundc,
            ):
                # One of the constraints failed → restart the generation loop
                continue

            # If we reach this line, all checks have passed and the sample is
            # ready to be packed and returned.
            break

        # 5)  Prepare (grid‑based) network inputs ---------------------------#

        # Separate normal vectors from Neumann values for convenience
        neu_bound_normals = neu_boundc[:, 1:]
        neu_boundc_val = neu_boundc[:, :1]

        # Resize the evaluation grid to the network input resolution
        xy_grid_big = np.stack(
            resize([
                xy_grid[:, 0].reshape(self.init_grid_size, self.init_grid_size),
                xy_grid[:, 1].reshape(self.init_grid_size, self.init_grid_size),
            ], factor=self.grid_size / self.init_grid_size),
            axis=-1,
        ).reshape(-1, 2)

        interpolator = WeightedInterpolator(xy_bound, xy_grid_big)
        dir_bound_grid = interpolator(dir_bound).reshape(self.grid_size, self.grid_size)
        neu_bound_grid = interpolator(neu_bound).reshape(self.grid_size, self.grid_size)
        dir_boundc_grid = interpolator(dir_boundc).reshape(self.grid_size, self.grid_size)
        neu_boundc_grid = interpolator(neu_boundc_val).reshape(self.grid_size, self.grid_size)
        f_grid_big = resize([f_grid], factor=self.grid_size / self.init_grid_size)[0]

        # Concatenate all "image"‑like channels expected by the network
        mat_inputs = np.concatenate(
            [
                np.expand_dims(dir_bound_grid, axis=0),
                np.expand_dims(neu_bound_grid, axis=0),
                np.expand_dims(dir_boundc_grid, axis=0),
                np.expand_dims(neu_boundc_grid, axis=0),
                np.expand_dims(f_grid_big, axis=0),
            ],
            axis=0,
        )

        # 6)  Return sample --------------------------------------------------#

        return {
            "pde_str": diff_op_str,
            "pde_derivatives": derivatives,
            "pde_coeffs": to_tensor([c for c in pde_coeffs_dict.values()]),
            "mat_inputs": to_tensor(mat_inputs),
            "xy_domain": to_tensor(xy_domain),
            "xy_boundary": to_tensor(xy_bound)[bound_point_ixs],
            "dir_boundary": to_tensor(dir_bound)[bound_point_ixs],
            "dir_boundary_c": to_tensor(dir_boundc)[bound_point_ixs],
            "neu_boundary": to_tensor(neu_bound)[bound_point_ixs],
            "neu_boundary_c": to_tensor(neu_boundc_val)[bound_point_ixs],
            "neu_bound_normals": to_tensor(neu_bound_normals)[bound_point_ixs],
            "u_domain": u_domain.detach(),
            "u_boundary": u_bound.detach(),
            "f_domain": f_domain.detach(),
            "f_boundary": f_bound.detach(),
            "u_grid": u_grid.reshape(self.init_grid_size, self.init_grid_size).detach(),
            "f_grid": f_grid.reshape(self.init_grid_size, self.init_grid_size).detach(),
            "ders_domain": {k: v.detach() for k, v in ders_domain.items()},
        }
