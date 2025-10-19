import numpy as np

from src.data.icbc.initial_conditions import IC
from src.data.icbc.boundary_conditions import DirichletBC, NeumannBC
from src.data.datasets import RandomFunctionDataset
from src.data.geometry.timedomain import GeometryXTime
from src.data.utils import (
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    NUM_POINTS_BOUNDARY,
    NUM_POINTS_BOUNDARY_DENSE,
    NUM_POINTS_DOMAIN,
    RandomFunction,
    classify_pde,
    encode_pde_str,
    generate_diff_operator,
    generate_random_domain,
    to_tensor,
    resize,
    WeightedInterpolator,
    boundary_condition_constraint,
)


class UnsupervisedRandFunDataset(RandomFunctionDataset):
    """Generate random unsupervised PDEs on‑the‑fly"""

    def __init__(
        self,
        size: int,
        grid_size: int = 224,
        init_grid_size: int = 224,
        max_abs_val: float = 10.0,
        max_num_inner_boundaries: int = 4,
    ) -> None:
        super().__init__(
            size,
            grid_size,
            init_grid_size,
            max_abs_val,
            max_num_inner_boundaries=max_num_inner_boundaries,
        )

    @staticmethod
    def _array_invalid(a: np.ndarray, *, max_abs: float | None = None, max_val: float | None = None) -> bool:
        """Return True if a violates numeric constraints."""
        if np.isnan(a).any() or np.isinf(a).any():
            return True
        if max_abs is not None and np.abs(a).max() > max_abs:
            return True
        if max_val is not None and np.abs(a).max() > max_val:
            return True
        return False

    def _is_sample_valid(self, *, f_domain: np.ndarray, dir_boundc: np.ndarray, neu_boundc: np.ndarray) -> bool:
        """Aggregate all sanity checks (mirrors original boolean guard)."""
        BIG_F_VAL = 1e4
        if self._array_invalid(f_domain, max_val=BIG_F_VAL):
            return False
        if self._array_invalid(dir_boundc, max_abs=self.max_abs_val):
            return False
        if self._array_invalid(neu_boundc, max_abs=1e2):
            return False
        return True

    def __getitem__(self, idx: int):  # noqa: C901 – (complexity is unavoidable)
        """Create a single dataset entry (unsupervised setting)."""

        # 0) Fixed random PDE operator & BC constraints (same every retry)
        diff_op_str, derivatives = generate_diff_operator()
        pde_coeffs_dict = encode_pde_str(diff_op_str)
        diff_op_type = classify_pde(pde_coeffs_dict)
        x_constr, y_constr = boundary_condition_constraint(pde_coeffs_dict)

        # Pre‑compute the dense spatial grid (no gradients needed)
        x_grid_np, y_grid_np = np.meshgrid(
            np.linspace(X_MIN, X_MAX, self.init_grid_size),
            np.linspace(Y_MAX, Y_MIN, self.init_grid_size),
        )
        xy_grid = np.concatenate([
            x_grid_np.reshape(-1, 1),
            y_grid_np.reshape(-1, 1),
        ], axis=-1)

        # Retry loop until numerical sanity passes
        while True:
            # 1)  Decide admissible boundary function library -------------#
            if x_constr == y_constr == "linear":
                num_terms_range = (1, 2)
                variables = ["x", "y"]
                bc_f_lib = [lambda x: x * 0, lambda x: x]
            elif x_constr == "linear":
                num_terms_range = (1, 2)
                variables = ["x"]
                bc_f_lib = [lambda x: x * 0, lambda x: x]
            elif y_constr == "linear":
                num_terms_range = (1, 2)
                variables = ["y"]
                bc_f_lib = [lambda x: x * 0, lambda x: x]
            elif x_constr == y_constr == "const":
                num_terms_range = (1, 2)
                variables = ["x", "y"]
                bc_f_lib = [lambda x: x * 0]
            elif x_constr == "const":
                num_terms_range = (1, 2)
                variables = ["x"]
                bc_f_lib = [lambda x: x * 0]
            elif y_constr == "const":
                num_terms_range = (1, 2)
                variables = ["y"]
                bc_f_lib = [lambda x: x * 0]
            else:  # "full" constraint
                num_terms_range = (0, 0)  # i.e. constant zero
                variables = ["x", "y"]
                bc_f_lib = [lambda x: x * 0]

            # 2)  Random geometry and interior points --------------------#
            geom, domain, inner_boundaries = generate_random_domain(
                bbox=[X_MIN, Y_MIN, X_MAX, Y_MAX],
                elliptic=(diff_op_type == "Elliptic"),
                max_num_inner_boundaries=self.max_num_inner_boundaries,
            )

            xy_domain = geom.random_points(NUM_POINTS_DOMAIN)

            # Dense boundary sampling (outer + initial if time domain) ----
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

            # 3)  Forcing term f(x, y) -----------------------------------#
            s_shift = np.random.normal(1) * 10  # base constant offset
            f_domain = np.zeros((len(xy_domain), 1)) + s_shift
            f_grid = np.zeros((len(xy_grid), 1)) + s_shift

            # 4)  Boundary conditions ------------------------------------#
            dir_boundc = np.zeros((len(xy_bound), 1))  # target value
            dir_bound = np.zeros((len(xy_bound), 1))   # mask (1 if active)
            neu_boundc = np.zeros((len(xy_bound), 3))  # value + normals
            neu_bound = np.zeros((len(xy_bound), 1))   # mask

            # --- Outer boundary ----------------------------------------
            outer_b_fun = RandomFunction(
                function_lib=bc_f_lib,
                num_terms_range=num_terms_range,
                cde_range=(-self.max_abs_val, self.max_abs_val),
                variables=variables,
            )

            on_bc = domain.on_boundary(xy_bound)
            bc = DirichletBC(domain, outer_b_fun)
            dir_boundc[on_bc] = bc(xy_bound[on_bc])
            dir_bound[on_bc] = 1.0

            if np.random.uniform() < 0.25:
                bc = NeumannBC(domain, outer_b_fun)
                neu_boundc[on_bc] = bc(xy_bound[on_bc], concat_normals=True)
                neu_bound[on_bc] = 1.0

            # --- Initial conditions (time‑dependent PDEs) --------------
            if diff_op_type == "First-Order":
                if isinstance(domain, GeometryXTime):
                    on_ic = domain.on_initial(xy_bound)
                    if np.random.uniform() < 0.75:
                        ic = IC(domain, outer_b_fun, derivative_order=0)
                        dir_boundc[on_ic] = ic(xy_bound[on_ic])
                        dir_bound[on_ic] = 1.0
                    else:
                        ic = IC(domain, outer_b_fun, derivative_order=1)
                        dir_boundc[on_ic], neu_boundc[on_ic] = ic(xy_bound[on_ic], concat_normals=True)
                        dir_bound[on_ic] = 1.0
                        neu_bound[on_ic] = 1.0
            elif diff_op_type != "Elliptic":
                on_ic = domain.on_initial(xy_bound)
                if diff_op_type == "Parabolic":
                    ic = IC(domain, outer_b_fun, derivative_order=0)
                    dir_boundc[on_ic] = ic(xy_bound[on_ic])
                    dir_bound[on_ic] = 1.0
                elif diff_op_type == "Hyperbolic":
                    ic = IC(domain, outer_b_fun, derivative_order=1)
                    dir_boundc[on_ic], neu_boundc[on_ic] = ic(xy_bound[on_ic], concat_normals=True)
                    dir_bound[on_ic] = 1.0
                    neu_bound[on_ic] = 1.0

            # --- Inner boundaries --------------------------------------
            for inner_bound in inner_boundaries:
                inner_b_fun = RandomFunction(
                    function_lib=bc_f_lib,
                    num_terms_range=num_terms_range,
                    cde_range=(-self.max_abs_val, self.max_abs_val),
                    variables=variables,
                )
                on_bc = inner_bound.on_boundary(xy_bound)

                if np.random.uniform() < 0.5:  # Neumann preferred half the time
                    bc = NeumannBC(inner_bound, inner_b_fun)
                    neu_boundc[on_bc] = bc(xy_bound[on_bc], concat_normals=True)
                    neu_bound[on_bc] = 1.0
                else:
                    # Dirichlet mandatory or additionally selected
                    bc = DirichletBC(inner_bound, inner_b_fun)
                    dir_boundc[on_bc] = bc(xy_bound[on_bc])
                    dir_bound[on_bc] = 1.0

            # 5)  Validation – identical to original conditions ----------#
            if not self._is_sample_valid(
                f_domain=f_domain,
                dir_boundc=dir_boundc,
                neu_boundc=neu_boundc,
            ):
                continue  # regenerate everything

            # If we reach here, the sample is numerically sound
            break

        # 6)  Prepare grid‑based network inputs -----------------------------#

        neu_bound_normals = neu_boundc[:, 1:]
        neu_boundc_val = neu_boundc[:, :1]

        # Resize dense grid coordinates to (grid_size x grid_size)
        xy_grid_big = np.stack(
            resize([
                x_grid_np.reshape(self.init_grid_size, self.init_grid_size),
                y_grid_np.reshape(self.init_grid_size, self.init_grid_size),
            ], factor=self.grid_size / self.init_grid_size),
            axis=-1,
        ).reshape(-1, 2)

        interpolator = WeightedInterpolator(xy_bound, xy_grid_big)
        dir_bound_grid = interpolator(dir_bound).reshape(self.grid_size, self.grid_size)
        neu_bound_grid = interpolator(neu_bound).reshape(self.grid_size, self.grid_size)
        dir_boundc_grid = interpolator(dir_boundc).reshape(self.grid_size, self.grid_size)
        neu_boundc_grid = interpolator(neu_boundc_val).reshape(self.grid_size, self.grid_size)

        # Resize f_grid to match input resolution
        f_grid = f_grid.reshape(self.init_grid_size, self.init_grid_size)
        f_grid_big = resize([f_grid], factor=self.grid_size / self.init_grid_size)[0]

        # Stack as (C x H x W)
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

        # 7)  Boundary sub‑sampling for training ---------------------------#
        bound_point_ixs = np.arange(len(xy_bound))
        np.random.shuffle(bound_point_ixs)
        bound_point_ixs = bound_point_ixs[:NUM_POINTS_BOUNDARY]

        # 8)  Final sample dict (unchanged API) ----------------------------#
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
            "f_domain": to_tensor(f_domain),
            "f_boundary": to_tensor(np.zeros((NUM_POINTS_BOUNDARY, 1)) + s_shift),
            "f_grid": to_tensor(f_grid),
        }
