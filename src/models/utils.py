import os
import math
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from safetensors.torch import save_file
from src.data.utils import ALL_DERIVATIVES, compile_diff_operator, compute_derivatives, plot_grids, to_tensor

DEFAULT_PINN_CONFIGS = {
    'input_dim': 2, 'output_dim': 1, 
    'lat_dim': 32, 'num_latent_layers': 3, 
    'activation': 'tanh',
}

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def plot_progress(xy, xy_bound, u_true, u_pred, u_bound_true, u_bound_pred, f_true, f_pred, train_losses, val_losses, lr_values,
                        path, steps, mode, ix, title):
        id = f"{steps}_{mode}_{ix}"
        xy_np = torch.cat([xy_bound, xy], dim=0).detach().cpu().numpy() if xy_bound is not None else xy.detach().cpu().numpy()
        xy_domain_np = xy.detach().cpu().numpy()
        
        u_pred_np = torch.cat([u_bound_pred, u_pred], dim=0).detach().cpu().numpy() \
                    if u_bound_pred is not None else u_pred.detach().cpu().numpy()
        if u_true is not None:
            u_true_np = torch.cat([u_bound_true, u_true], dim=0).reshape(u_pred_np.shape).detach().cpu().numpy() \
                        if u_bound_true is not None else u_true.reshape(u_pred_np.shape).detach().cpu().numpy()
        else:
            u_true_np = torch.cat([u_bound_true.reshape(u_pred.shape), u_pred], dim=0).detach().cpu().numpy() \
                        if u_bound_true is not None else u_pred.detach().cpu().numpy()
        f_pred_np = f_pred.detach().cpu().numpy()
        f_true_np = f_true.detach().cpu().numpy().reshape(f_pred_np.shape)
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plotting ground truth u
        sc = axs[0, 0].scatter(xy_np[:, 0], xy_np[:, 1], c=u_true_np, cmap='viridis')
        plt.colorbar(sc, ax=axs[0, 0])
        axs[0, 0].set_title('Ground Truth u')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        
        # Plotting predicted u
        sc = axs[0, 1].scatter(xy_np[:, 0], xy_np[:, 1], c=u_pred_np, cmap='viridis')
        plt.colorbar(sc, ax=axs[0, 1])
        axs[0, 1].set_title('Predicted u')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        
        # Plotting L1 distance u
        u_l1_distance = np.abs(u_true_np - u_pred_np)
        sc = axs[0, 2].scatter(xy_np[:, 0], xy_np[:, 1], c=u_l1_distance, cmap='viridis')
        plt.colorbar(sc, ax=axs[0, 2])
        axs[0, 2].set_title('L1 Distance u')
        axs[0, 2].set_xlabel('x')
        axs[0, 2].set_ylabel('y')
        
        # Plotting ground truth f
        sc = axs[1, 0].scatter(xy_domain_np[:, 0], xy_domain_np[:, 1], c=f_true_np, cmap='viridis')
        plt.colorbar(sc, ax=axs[1, 0])
        axs[1, 0].set_title('Ground Truth f')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')
        
        # Plotting predicted f
        sc = axs[1, 1].scatter(xy_domain_np[:, 0], xy_domain_np[:, 1], c=f_pred_np, cmap='viridis')
        plt.colorbar(sc, ax=axs[1, 1])
        axs[1, 1].set_title('Predicted f')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')
        
        # Plotting L1 distance f
        f_l1_distance = np.abs(f_true_np - f_pred_np)
        sc = axs[1, 2].scatter(xy_domain_np[:, 0], xy_domain_np[:, 1], c=f_l1_distance, cmap='viridis')
        plt.colorbar(sc, ax=axs[1, 2])
        axs[1, 2].set_title('L1 Distance f')
        axs[1, 2].set_xlabel('x')
        axs[1, 2].set_ylabel('y')
        
        fig.suptitle(title)
        plt.tight_layout()
        os.makedirs(f'{path}/plots', exist_ok=True)
        plt.savefig(f'{path}/plots/vis_{id}')
        plt.close(fig)

        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.grid()
        plt.savefig(f'{path}/plots/train_loss')
        plt.close()

        plt.plot(val_losses)
        plt.title('Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.grid()
        plt.savefig(f'{path}/plots/val_loss')
        plt.close()

        plt.plot(lr_values)
        plt.title('Learning Rate')
        plt.ylabel('LR')
        plt.xlabel('Epochs')
        plt.grid()
        plt.savefig(f'{path}/plots/learning_rate')
        plt.close()

class SinPositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4")

        self.channels = channels

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, D, W, H]
        Returns:
            Tensor with positional encoding added
        """
        B, D, W, H = x.size()
        device = x.device

        channels_per_dim = self.channels // 2

        div_term = torch.exp(
            torch.arange(0, channels_per_dim, 2, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / channels_per_dim)
        )

        pos_w = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)
        pos_h = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)

        pe = torch.zeros(1, self.channels, W, H, device=device)

        # Positional encoding for width dimension
        pe[0, 0:channels_per_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, H)
        pe[0, 1:channels_per_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, H)

        # Positional encoding for height dimension
        pe[0, channels_per_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(1).repeat(1, W, 1)
        pe[0, channels_per_dim+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(1).repeat(1, W, 1)

        return x + pe
    
class FourierMapping(nn.Module):
    def __init__(self, input_dim, num_features, scale=.1, forward_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.scale = scale
        self.forward_input = forward_input
        self.output_dim = input_dim * 2 * num_features + (input_dim if forward_input else 0)
        
        # Create fixed frequencies
        freq_bands = 2.0 ** torch.linspace(0, num_features - 1, steps=num_features) * scale
        self.frequencies = torch.nn.Parameter(freq_bands.unsqueeze(0).repeat(input_dim, 1), requires_grad=False)
        
    def forward(self, x):
        # Compute the sine and cosine Fourier features
        sin_features = torch.sin(2 * torch.pi * x.unsqueeze(-1) * self.frequencies).flatten(-2, -1)
        cos_features = torch.cos(2 * torch.pi * x.unsqueeze(-1) * self.frequencies).flatten(-2, -1)
        
        # Optionally concatenate the original input
        if self.forward_input:
            fourier_features = torch.cat([sin_features, cos_features, x], dim=-1)
        else:
            fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        return fourier_features
    
class FourierMapping2d(FourierMapping):
    def __init__(self, input_dim, num_features, scale=.1, forward_input=True):
        super().__init__(input_dim, num_features, scale, forward_input)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
class FiLM(nn.Module):
    def __init__(self, in_channels, conditioning_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(conditioning_dim, in_channels)
        self.beta = nn.Linear(conditioning_dim, in_channels)
    
    def forward(self, x, conditioning):
        gamma = self.gamma(conditioning).unsqueeze(1).unsqueeze(1)
        beta = self.beta(conditioning).unsqueeze(1).unsqueeze(1)
        return gamma * x + beta
    
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        
        # Initialize learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x has shape (B, P, C)
        batch_size = x.size(0)
        
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # Shape (B, num_queries, C)
        
        attn_output, _ = self.attention(queries, x, x)
        
        return attn_output
    
class StatsLayer(nn.Module):
    def __init__(self, input_dim, lat_dim, num_fourier_features=5, activation=nn.GELU):
        super().__init__()
        self.lat_dim = lat_dim
        self.activation = activation
        self.four_features = FourierMapping(input_dim * 4, num_fourier_features, forward_input=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.four_features.output_dim, self.lat_dim),
            self.activation(),
        )

    def forward(self, mat_inputs):
        mat_stats = torch.cat([
            mat_inputs.mean(dim=(2, 3)),
            mat_inputs.amax(dim=(2, 3)),
            mat_inputs.amin(dim=(2, 3)),
            mat_inputs.std(dim=(2, 3)),
        ], dim=-1).detach()
        mat_stats = self.four_features(mat_stats)
        mat_stats = self.ffn(mat_stats)
        return mat_stats
    
class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, lat_dim, num_latent_layers, activation='tanh'):
        super().__init__()
        self.activation = getattr(nn.functional, activation) if isinstance(activation, str) else activation
        self.num_latent_layers = num_latent_layers
        self.fourier_transform = FourierMapping(input_dim=input_dim, num_features=5)

        self.u_map = nn.Linear(self.fourier_transform.output_dim, lat_dim)
        self.v_map = nn.Linear(self.fourier_transform.output_dim, lat_dim)
        
        self.lat_maps = nn.ModuleList(
            [nn.Linear(self.fourier_transform.output_dim, lat_dim)] 
            + [nn.Linear(lat_dim, lat_dim) for _ in range(num_latent_layers)] 
            + [nn.Linear(lat_dim, output_dim)]
        )

    def forward(self, inputs):
        x = self.fourier_transform(inputs)
        u = torch.tanh(self.u_map(x))
        v = torch.tanh(self.v_map(x))
        x = self.activation(self.lat_maps[0](x))
        for layer in self.lat_maps[1:-1]:
            x = u * x + v * (1 - x)
            x = self.activation(layer(x))
        return self.lat_maps[-1](x)

    @classmethod
    def from_parameters(cls, weights, biases, config=DEFAULT_PINN_CONFIGS, remove_batch_dim=False):
        if remove_batch_dim:
            weights = [w[0] for w in weights]
            biases = [b[0] for b in biases]

        net = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            lat_dim=config['lat_dim'],
            num_latent_layers=config['num_latent_layers'],
            activation=config.get('activation', 'tanh')
        )

        with torch.no_grad():
            net.u_map.weight.copy_(weights[0].T.contiguous())
            net.u_map.bias.copy_(biases[0].contiguous())

            net.v_map.weight.copy_(weights[1].T.contiguous())
            net.v_map.bias.copy_(biases[1].contiguous())

            for i, layer in enumerate(net.lat_maps):
                layer.weight.copy_(weights[i + 2].T.contiguous())
                layer.bias.copy_(biases[i + 2].contiguous())

        return net
    
class Ensemble(nn.Module):
    def __init__(self, num_nets: int = 0, weights=None, fnn_config=DEFAULT_PINN_CONFIGS):
        super().__init__()
        self.cfg = dict(fnn_config)

        if num_nets > 0:
            self.nets = nn.ModuleList([
                FNN(
                    input_dim=self.cfg['input_dim'],
                    output_dim=self.cfg['output_dim'],
                    lat_dim=self.cfg['lat_dim'],
                    num_latent_layers=self.cfg['num_latent_layers'],
                    activation=self.cfg.get('activation', 'tanh'),
                )
                for _ in range(num_nets)
            ])
            w = torch.ones(num_nets, dtype=torch.float32) if weights is None else torch.as_tensor(weights, dtype=torch.float32)
            self.weights = nn.Parameter(w, requires_grad=False)
        else:
            self.nets = nn.ModuleList()
            self.weights = None

    def forward(self, x):
        if not self.nets:
            raise RuntimeError("Ensemble is empty – no networks to evaluate.")
        return torch.stack([net(x) * w for net, w in zip(self.nets, self.weights)], dim=-1).sum(dim=-1)

    def add_net(self, net: nn.Module, weight: float = 1.0):
        device = next(net.parameters()).device
        self.nets.append(net)

        if self.weights is None:
            new_w = torch.tensor([weight], dtype=torch.float32, device=device)
        else:
            new_w = torch.cat([self.weights.detach(), torch.tensor([weight], device=device)], dim=0)
        self.weights = nn.Parameter(new_w, requires_grad=False)

    @property
    def num_nets(self):
        return len(self.nets)

class Adam_LBFGS(torch.optim.Optimizer):
    """
    Switches from Adam to LBFGS after *switch_step* optimisation steps.
    Cosine LR decay is applied only during the Adam phase, down to min_lr.
    """

    def __init__(
        self,
        params,
        switch_step: int = 1_000,
        adam_hyperparams={"lr": 1e-3, "betas": (0.9, 0.99), "weight_decay": 0.0},
        lbfgs_hyperparams={"lr": 0.1, "max_iter": 20, "history_size": 60, "line_search_fn": "strong_wolfe", 
                     "tolerance_grad": 1e-8, "tolerance_change":1e-10},
        min_lr: float = 1e-8,
    ):
        self._params = list(params)
        self.switch_step = int(switch_step)
        self.min_lr = float(min_lr)

        self.adam = torch.optim.AdamW(self._params, **adam_hyperparams)
        # Remember each group's base LR to decay from
        self._adam_base_lrs = [g["lr"] for g in self.adam.param_groups]

        self.lbfgs = torch.optim.LBFGS(self._params, **lbfgs_hyperparams)

        defaults = {}
        super().__init__(self._params, defaults)
        self.state["step"] = 0
        self.state["using_lbfgs"] = False

    def _apply_adam_cosine_decay(self):
        """
        Cosine LR decay per param group:
          lr(t) = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * t/T))
        with t in [0, T], T = switch_step.
        """
        # Clamp progress to [0, 1]
        t = min(self.state["step"], self.switch_step)
        if self.switch_step <= 0:
            progress = 1.0
        else:
            progress = t / self.switch_step

        for g, base_lr in zip(self.adam.param_groups, self._adam_base_lrs):
            decayed = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            # Numerically guard against going below min_lr due to fp errors
            g["lr"] = max(decayed, self.min_lr)

    def step(self, closure):  # type: ignore[override]
        """closure() should *zero* grads, compute loss, call backward, then return loss"""
        self.state["step"] += 1

        # Phase I: Adam
        if not self.state["using_lbfgs"]:
            # Update LR via cosine decay before the Adam step
            self._apply_adam_cosine_decay()

            loss = closure()  # gradients computed in closure
            self.adam.step()

            if self.state["step"] >= self.switch_step:
                print(f"[Adam_LBFGS] Switching to LBFGS at optimiser step {self.state['step']}")
                self.state["using_lbfgs"] = True
            return loss

        # Phase II: LBFGS
        loss = self.lbfgs.step(closure)
        return loss

    # Convenience
    def using_lbfgs(self):
        return self.state["using_lbfgs"]


def finetuning(
    pde,
    net,
    num_adam_iterations,
    num_lbfgs_iterations,
    num_collocation_points=None,
    eval_every=500,
    loss_weights=None,
    plot_path=None,
    adam_hyperparams=None,
    lbfgs_hyperparams=None,
    boundary_oversample='balanced',  # None | 'balanced' | float(0,1)
):
    """
    Fine-tunes a neural PDE solver (e.g. from HyPINO) using Adam + L-BFGS.

    boundary_oversample:
        - None:       uniform sampling over all points
        - 'balanced': 50% boundary, 50% interior
        - float p∈(0,1): fraction of boundary points per batch (e.g. 0.3 → 30%)
    """
    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    device = next(net.parameters()).device
    net = net.to(device)
    loss_weights = loss_weights or {'F': 0.1, 'D': 10, 'N': 5}

    D_mask, N_mask, g_D, g_N, f_tgt = [t.flatten().to(device) for t in pde['mat_inputs']]
    N_normals = pde['neu_normals'].reshape(-1, 2).to(device) if 'neu_normals' in pde else torch.zeros((len(D_mask), 2)).to(device)
    O_mask = pde.get('domain_mask', torch.ones_like(D_mask)).flatten().to(device)
    diff_op = compile_diff_operator(pde['pde_str'])

    grid_res = int(D_mask.shape[0] ** 0.5)
    xs, ys = np.linspace(-1, 1, grid_res), np.linspace(1, -1, grid_res)
    x_all, y_all = np.meshgrid(xs, ys)
    x_all = to_tensor(x_all).reshape(-1, 1).to(device)
    y_all = to_tensor(y_all).reshape(-1, 1).to(device)
    n_total = x_all.shape[0]
    num_collocation_points = min(num_collocation_points or n_total, n_total)

    # ----------------------------------------------------------------------
    # Sampling helpers
    # ----------------------------------------------------------------------
    with torch.no_grad():
        boundary_mask = (D_mask > 0) | (N_mask > 0)
        interior_mask = (~boundary_mask) & (O_mask > 0)
        boundary_idx = torch.nonzero(boundary_mask, as_tuple=False).flatten()
        interior_idx = torch.nonzero(interior_mask, as_tuple=False).flatten()

    def masked_mse(residual, mask):
        """Mean squared error normalized by active mask."""
        res = residual.squeeze(-1)
        return ((mask * res**2).sum() / mask.sum().clamp_min(1.0))

    def pick_indices(M):
        """Sample indices with optional boundary oversampling."""
        if (
            boundary_oversample is None
            or len(boundary_idx) == 0
            or len(interior_idx) == 0
        ):
            return torch.randperm(n_total, device=device)[:M]

        p = 0.5 if boundary_oversample == "balanced" else float(boundary_oversample)
        p = min(max(p, 0.0), 1.0)
        n_b = int(M * p)
        n_i = M - n_b

        def sample(idx, k):
            if k <= len(idx):
                return idx[torch.randperm(len(idx), device=device)[:k]]
            return idx[torch.randint(0, len(idx), (k,), device=device)]

        sel_b = sample(boundary_idx, n_b)
        sel_i = sample(interior_idx, n_i)
        all_idx = torch.cat([sel_b, sel_i])
        return all_idx[torch.randperm(len(all_idx), device=device)]

    # ----------------------------------------------------------------------
    # Optimizers
    # ----------------------------------------------------------------------
    adam_hyperparams = adam_hyperparams or {"lr": 1e-3, "betas": (0.9, 0.99)}
    lbfgs_hyperparams = lbfgs_hyperparams or {
        "lr": 0.1,
        "max_iter": 20,
        "history_size": 60,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": 1e-8,
        "tolerance_change": 1e-10,
    }

    optimizer = Adam_LBFGS(
        net.parameters(),
        switch_step=num_adam_iterations,
        adam_hyperparams=adam_hyperparams,
        lbfgs_hyperparams=lbfgs_hyperparams,
    )

    metrics = {"total": None, "F": None, "D": None, "N": None}
    loss_history = {"total": [], "F": [], "D": [], "N": []}
    num_iterations = num_adam_iterations + num_lbfgs_iterations

    # ----------------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------------
    for i in range(num_iterations):
        idx = pick_indices(num_collocation_points)
        x, y = x_all[idx], y_all[idx]
        Dm, Nm, gD, gN, fT, NmNormals, Om = (
            D_mask[idx],
            N_mask[idx],
            g_D[idx],
            g_N[idx],
            f_tgt[idx],
            N_normals[idx],
            O_mask[idx],
        )

        def closure():
            optimizer.zero_grad()
            xy = torch.cat([x.requires_grad_(), y.requires_grad_()], dim=-1)
            u = net(xy)

            D_loss = masked_mse(u.squeeze(-1) - gD, Dm)

            derivs = compute_derivatives(
                {"x": x, "y": y},
                {"u": u},
                derivatives=ALL_DERIVATIVES,
            )

            f_pred = diff_op(derivs).squeeze(-1)
            F_loss = masked_mse(f_pred - fT, Om)

            neumann_pred = (
                torch.cat([derivs["ux"], derivs["uy"]], dim=1) * NmNormals
            ).sum(dim=1)
            N_loss = masked_mse(neumann_pred - gN, Nm)

            total_loss = loss_weights["F"] * F_loss + loss_weights["D"] * D_loss + loss_weights["N"] * N_loss

            total_loss.backward()
            metrics["total"] = total_loss
            metrics["F"] = F_loss
            metrics["D"] = D_loss
            metrics["N"] = N_loss
            return total_loss

        optimizer.step(closure)
        metrics = {k: float(v.detach()) for k, v in metrics.items()}

        if i % eval_every == 0 or i == num_iterations - 1:
            for k in loss_history:
                loss_history[k].append(metrics[k])
            print(
                f"[{i:05d}] "
                f"Loss={metrics['total']:.4e} "
                f"(F={metrics['F']:.2e}, D={metrics['D']:.2e}, N={metrics['N']:.2e})"
            )

            # -------------------- Evaluation (always) --------------------
            u_full, f_full = torch.empty(n_total, device="cpu"), torch.empty(n_total, device="cpu")

            for start in range(0, n_total, num_collocation_points):
                end = min(start + num_collocation_points, n_total)
                x, y = x_all[start:end], y_all[start:end]
                xy = torch.cat([x.requires_grad_(), y.requires_grad_()], dim=-1).requires_grad_()
                u_b = net(xy)
                derivs = compute_derivatives(
                    {"x": x, "y": y}, {"u": u_b}, ALL_DERIVATIVES
                )
                f_b = diff_op(derivs).squeeze(-1)
                u_full[start:end] = u_b.detach().squeeze(-1).cpu()
                f_full[start:end] = f_b.detach().cpu()

            u_grid = u_full.numpy().reshape(grid_res, grid_res)
            f_grid = f_full.numpy().reshape(grid_res, grid_res)
            f_target = f_tgt.detach().cpu().numpy().reshape(grid_res, grid_res)
            mask_grid = O_mask.detach().cpu().numpy().reshape(grid_res, grid_res)

            fields = [u_grid, f_grid, (f_grid - f_target) * mask_grid]
            titles = [
                r"$u_{\mathrm{pred}}$",
                r"$f_{\mathrm{pred}}$",
                r"Residual $f_{\mathrm{pred}}-f$",
            ]
            plot_grids(
                fields,
                titles=titles,
                cmap="cividis",
                dpi=120,
                save_path=f"{plot_path}/step_{i:05d}.png" if plot_path else None,
            )

    return loss_history


def convert_ckpt_to_safetensors(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Clone all tensors to break shared memory
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.clone()

    safetensors_path = '.'.join(ckpt_path.split('.')[:-1]) + ".safetensors"
    save_file(state_dict, safetensors_path)
    print("Converted .ckpt to .safetensors")
    return safetensors_path


