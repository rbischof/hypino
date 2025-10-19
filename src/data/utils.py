import re
import torch
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from scipy.spatial import cKDTree
from sympy import expand, simplify
from torch.utils.data import default_collate
from src.data.geometry.csg import CSGDifference
from src.data.geometry.geometry_1d import Interval
from src.data.geometry.timedomain import GeometryXTime, TimeDomain
from src.data.geometry.geometry_2d import Disk, Polygon, Rectangle, Triangle

PDE_TERMS = sorted(
    list(set(['u', 'ux', 'uy', 'uxx', 'uyy'])),
    key=lambda x: (len(x), x)
)

ALL_DERIVATIVES = {'u': [[('x', 1)], [('y', 1)], [('x', 2)], [('y', 2)]]}

PDE_ALL_TERMS = [sp.symbols(term) for term in PDE_TERMS]

NUM_POINTS_DOMAIN = 2048
NUM_POINTS_BOUNDARY = NUM_POINTS_DOMAIN
NUM_POINTS_SOURCES = NUM_POINTS_DOMAIN // 4
NUM_POINTS_BOUNDARY_DENSE = NUM_POINTS_BOUNDARY * 3

X_MIN, Y_MIN = -1, -1
X_MAX, Y_MAX =  1,  1
WIDTH = X_MAX - X_MIN
HEIGHT = Y_MAX - Y_MIN

SAMPLER = 'pseudo'
EPS = 1e-4

class RandomFunction:
    """
    Samples a random analytical solution u(x,y) via the Method of Manufactured Solutions (MMS)
    """

    def __init__(self, 
                 function_lib=[
                        lambda x: x,
                        torch.sin,
                        torch.cos,
                        torch.tanh,
                        torch.sigmoid,
                        lambda x: 1.0 / (1.0 + x**2)
                    ],
                 num_terms_range=(6, 10), 
                 a_range=(-10.0, 10.0), 
                 cde_range=(-2*np.pi, 2*np.pi),
                 variables=['x', 'y']):
        """
        Args:
            num_terms_range: tuple (min_terms, max_terms), number of iterations n ∼ Uniform{min,…,max}
            a_range: range for a, b coefficients when non-zero
            cde_range: range for c, d, e offsets and multipliers
            variables: list of variables to use
        """
        
        # Term count
        self.n = np.random.randint(num_terms_range[0], num_terms_range[1] + 1)
        
        # Basis library ψ ∈ { x, sin, cos, tanh, sigmoid, (1+x^2)^{-1} }
        self.basis = function_lib

        self.variables = ['x', 'y']
        
        # Pre-sample all parameters
        self.terms = []
        for i in range(self.n):
            psi = np.random.choice(self.basis)
            # a,b each ∈ {0 with prob .5, else Uniform[a_range]}
            a = 0.0 if np.random.rand() < 0.5 or 'x' not in variables else np.random.uniform(*a_range)
            b = 0.0 if np.random.rand() < 0.5 or 'y' not in variables else np.random.uniform(*a_range)
            c = np.random.uniform(*cde_range)
            d = np.random.uniform(*cde_range) if i < self.n - 1 else 1
            e = np.random.uniform(*cde_range) if i < self.n - 1 else 0
            rule = np.random.choice(['add', 'mult', 'compose'])
            self.terms.append({'psi': psi, 'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'rule': rule})

    def __call__(self, xy):
        """
        Evaluate the sampled solution at points xy.

        Args:
            xy: tensor of shape (..., 2) or (batch_size, 2)
        Returns:
            u: tensor of shape (..., 1) or (batch_size, 1) with u(x,y)
        """
        if not isinstance(xy, torch.Tensor):
            xy = torch.tensor(xy, dtype=torch.float32)
        x, y = xy[..., 0:1], xy[..., 1:2]
        
        # Initialize u(x,y) ≡ 0
        u = (x + y) * 0
        
        for term in self.terms:
            psi = term['psi']
            a, b, c, d, e = (term[k] for k in ('a','b','c','d','e'))
            rule = term['rule']
            
            if rule in ('add', 'mult'):
                # compute ψ(a x + b y + c)
                arg = a * x + b * y + c
                val = d * psi(arg) + e
                if rule == 'add':
                    u = u + val
                else:  # 'mult'
                    u = u * val
            else:  # 'compose'
                # u ← d · ψ(u) + e
                u = d * psi(u) + e
        
        return u
    

def generate_diff_operator():
    """
    Sample a differential operator L as in the paper:
      - pick n ∈ {1,2,3} uniformly
      - choose n distinct terms from ['u','ux','uy','uxx','uyy']
      - draw each coefficient from Uniform([-2,2])
    Returns:
      pde_formula: str (expanded & simplified symbolic sum)
      info: dict with key 'u' mapping to a list of [(var, order),…] for each chosen derivative
    """

    # 1) sample number of terms n ∈ {1,2,3}
    n = random.choice([1, 2, 3])

    # 2) sample n distinct terms WITHOUT replacement
    chosen = random.sample(PDE_TERMS, k=n)

    # 3) assign each a random coefficient in [-2,2]
    coeffs = {t: np.random.uniform(-2, 2) for t in chosen}

    # 4) build symbolic formula
    term_strs = [f"{coeffs[t]} * {t}" for t in chosen]
    pde_formula = str(expand(simplify(" + ".join(term_strs))))

    # 5) extract derivative orders exactly like original
    derivatives = [
        [(v, term[1:].count(v)) for v in set(term[1:]) if term[1:].count(v) > 0]
        for term in chosen
        if len(term) > 1
    ]

    return pde_formula, {'u': derivatives}

def generate_random_shape(bbox=[-1, -1, 1, 1], shape_types=['Triangle', 'Disk', 'Rectangle', 'Polygon'], size_denominator=4):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    xmin, xmax = bbox[:2], bbox[2:]

    shape_type = random.choice(shape_types)
    center = np.random.uniform(low=xmin, high=xmax, size=2)  # A central point for constraining sizes
    max_offset = min(width, height) / size_denominator

    if shape_type == 'Disk':
        radius = np.random.uniform(low=1, high=max_offset)
        return Disk(center, radius / 2)

    elif shape_type == 'Triangle':
        vertices = [center + np.random.uniform(low=-max_offset, high=max_offset, size=2) for _ in range(3)]
        return Triangle(*vertices)

    elif shape_type == 'Polygon':
        num_vertices = random.randint(4, 6)
        vertices = [center + np.random.uniform(low=-max_offset, high=max_offset, size=2) for _ in range(num_vertices)]
        return Polygon(vertices)

    elif shape_type == 'Rectangle':
        rect_xmin = np.random.uniform(low=xmin, high=xmax, size=2)
        rect_xmax = np.random.uniform(low=rect_xmin + 1e-6, high=rect_xmin + max_offset, size=2)
        return Rectangle(rect_xmin, rect_xmax)
    
    else:
        raise ValueError(f'Shape type {shape_type} is unknown')
    
def generate_random_domain(bbox=[-1, -1, 1, 1], max_num_inner_boundaries=4, elliptic=False,
                           shape_types=['Triangle', 'Disk', 'Rectangle']):
    if elliptic:
        domain = Rectangle(bbox[:2], bbox[2:])
    else:
        spatial = Interval(bbox[0], bbox[2])
        time = TimeDomain(bbox[1], bbox[3])
        domain = GeometryXTime(spatial, time)
    geom = domain
    
    inner_boundaries = []
    if max_num_inner_boundaries > 0:
        inner_boundary_count = np.random.randint(max_num_inner_boundaries)
        
        for _ in range(inner_boundary_count):
            inner_boundaries.append(generate_random_shape(bbox, shape_types))
        
        for boundary in inner_boundaries:
            geom = CSGDifference(geom, boundary)

    return geom, domain, inner_boundaries

def compile_diff_operator(fun_str):
    derivative_terms = sorted(
        [term for term in PDE_TERMS if re.search(r'\b' + re.escape(term) + r'\b', fun_str)], 
        key=lambda x: (len(x), x)
    )
    
    # Create a lambda function that takes dynamic arguments
    func = eval(f"lambda {', '.join(derivative_terms)}: {fun_str}", {"torch": torch, "np": np})
    def fun(pred):
        return func(**{t: pred[t] for t in derivative_terms})
    return fun

def encode_pde_str(pde_str):
    # Function to encode the formula into an array of coefficients
    pde_expr = sp.sympify(pde_str)
    
    # Collect all terms with respect to the PDE_ALL_TERMS using sympy
    collected_expr = sp.collect(pde_expr, PDE_ALL_TERMS, evaluate=False)
    
    # Initialize the coefficient array with zeros
    coefficients = {str(k): 0.0 for k in PDE_ALL_TERMS}

    # Loop through PDE_ALL_TERMS and extract their coefficients
    for i, term in enumerate(PDE_ALL_TERMS):
        if term in collected_expr:
            coefficients[str(term)] = float(collected_expr[term])

    return coefficients

def classify_pde(coeffs):
    # Initialize coefficients for the linear second-order terms
    A = coeffs.get('uxx', 0)
    B = coeffs.get('uxy', 0) + coeffs.get('uyx', 0)  # uxy and uyx are the same
    C = coeffs.get('uyy', 0)

    # Track if any second-order terms are present
    second_order_present = any(
        term for term in coeffs if 'xx' in term or 'xy' in term or 'yx' in term or 'yy' in term
    )

    # If no second-order terms are present, classify as first-order
    if not second_order_present:
        return "First-Order"
    
    # Apply local linearization for nonlinear second-order terms
    for term, coeff in coeffs.items():
        if coeff != 0 and '*' in term:
            if 'uxx' in term:
                A += coeff
            if 'uxy' in term or 'uyx' in term:
                B += coeff
            if 'uyy' in term:
                C += coeff

    # Calculate the discriminant for the principal part
    discriminant = B**2 - 4*A*C

    # Classify based on the discriminant
    if discriminant < 0:
        return "Elliptic"
    elif discriminant == 0:
        return "Parabolic"
    else:
        return "Hyperbolic"
    
def to_tensor(arr, requires_grad=False, dtype=torch.float32):
    if isinstance(arr, torch.Tensor):
        tensor = arr.clone().detach()
        if requires_grad:
            tensor.requires_grad_()
        return tensor.type(dtype)
    else:
        return torch.tensor(arr, requires_grad=requires_grad, dtype=dtype)

def compute_derivatives(in_var_map, out_var_map, derivatives):

    outputs = out_var_map

    # Compute derivatives for each output variable
    for out_var, derivatives in derivatives.items():
        for derivative in derivatives:
            grads = outputs[out_var]
            der_var_name = []
            for in_var_name, order in derivative:
                in_var = in_var_map[in_var_name]
                
                for i in range(order):
                    der_var_name.append(in_var_name)
                    diff_expression = f'{out_var}{"".join(sorted(der_var_name))}'
                    if diff_expression in outputs:
                        grads = outputs[diff_expression]
                    else:
                        grads = torch.autograd.grad(
                            grads, in_var, torch.ones_like(grads),
                            create_graph=True, materialize_grads=True,
                        )[0].requires_grad_()
            
                        outputs[diff_expression] = grads
    return outputs

def resize(mats, factor):
    for i in range(len(mats)):
        if torch.is_tensor(mats[i]):
            mats[i] = mats[i].detach().cpu().numpy()
    if not np.isclose(factor, 1):
        for i in range(len(mats)):
            mats[i] = zoom(mats[i], factor, order=1)
    return mats

class WeightedInterpolator:
    def __init__(self, xy_coords, new_xy_coords, scale_factor=.1, thresh=0.8):
        """
        Precomputes neighbor distances and indices for interpolation.

        Parameters:
            xy_coords: array-like, shape (N, 2) - Original coordinates
            new_xy_coords: array-like, shape (M, 2) - New coordinates for interpolation
            scale_factor: float - Controls the weighting decay based on distance
        """
        # Convert inputs to NumPy arrays
        if torch.is_tensor(xy_coords):
            self.xy_coords_np = xy_coords.detach().cpu().numpy()
        else:
            self.xy_coords_np = xy_coords
        if torch.is_tensor(new_xy_coords):
            self.new_xy_coords_np = new_xy_coords.detach().cpu().numpy()
        else:
            self.new_xy_coords_np = new_xy_coords
        
        self.scale_factor = scale_factor
        self.thresh = thresh
        
        # Build KDTree and precompute distances and indices
        tree = cKDTree(self.xy_coords_np)
        self.distances, self.indices = tree.query(self.new_xy_coords_np, k=1)
        
        # Compute weights as a function of distances
        self.weights = np.exp(-self.distances / self.scale_factor)

    def __call__(self, u_values, default_value=0):
        """
        Performs weighted interpolation using precomputed neighbors.

        Parameters:
            u_values: array-like, shape (N, D) - Values associated with the original coordinates
            default_value: float - Default value to use if no neighbors are found

        Returns:
            interpolated_values: array-like, shape (M, D) - Interpolated values at new coordinates
        """
        # Convert u_values to NumPy array if necessary
        if torch.is_tensor(u_values):
            u_values_np = u_values.detach().cpu().numpy()
        else:
            u_values_np = u_values
        
        # Initialize interpolated values with the default value
        interpolated_values_np = np.full((self.new_xy_coords_np.shape[0], u_values_np.shape[1]), default_value, dtype=np.float64)
        
        # Assign weighted values to each new point
        for i, idx in enumerate(self.indices):
            weight = self.weights[i]
            if idx < len(u_values_np):  # Ensure the index is valid
                interpolated_values_np[i] = float(weight > self.thresh) * u_values_np[idx]
        
        return interpolated_values_np
    
def boundary_condition_constraint(pde_coeffs_dict):
    x_constr = 'none'
    y_constr = 'none'

    if not np.isclose(pde_coeffs_dict['u'], 0.):
        return '0', '0'
    if not np.isclose(pde_coeffs_dict['ux'], 0.):
        x_constr = 'const'
    if not np.isclose(pde_coeffs_dict['uy'], 0.):
        y_constr = 'const'
    if not np.isclose(pde_coeffs_dict['uxx'], 0.) and x_constr != 'const':
        x_constr = 'linear'
    if not np.isclose(pde_coeffs_dict['uyy'], 0.) and y_constr != 'const':
        y_constr = 'linear'
    return x_constr, y_constr

def custom_collate(batch, custom_features=['pde_str', 'pde_derivatives', 'ders_domain', 'u_domain', 'u_grid', 'ders_domain', 'domain_mask']):
    collated_data = {}

    # Handle regular features
    for key in batch[0]:
        if key not in custom_features:
            collated_data[key] = default_collate([item[key] for item in batch if key in item])

    for custom_key in custom_features:
        # Collect values or None if missing
        collated_data[custom_key] = [item.get(custom_key, None) for item in batch]

    return collated_data

def plot_grids(fields, titles=None, mask=None, cmap='cividis', vmin=None, vmax=None, dpi=150, save_path=None):
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 20,
        'axes.linewidth': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': False,
    })
    
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=dpi)
    if n == 1:
        axes = [axes]

    for i, (field, ax) in enumerate(zip(fields, axes)):
        if field is None:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=24, transform=ax.transAxes)
        else:
            data = field * mask if mask is not None else field
            nrows, ncols = data.shape
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

            x_ticks = [0, ncols // 2, ncols - 1]
            x_labels = [-1, 0, 1]
            y_ticks = [0, nrows // 2, nrows - 1]
            y_labels = [1, 0, -1]

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if titles:
                ax.set_title(titles[i])
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()