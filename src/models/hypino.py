import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import pytorch_lightning as pl

from typing import List
from datetime import datetime
from functools import partial
from safetensors.torch import load_file
from torch.nn.functional import huber_loss
from timm.models.swin_transformer import _create_swin_transformer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from src.data.utils import ALL_DERIVATIVES, compile_diff_operator, compute_derivatives, plot_grids, to_tensor
from src.models.utils import (plot_progress, SinPositionalEncoding2D, FourierMapping, FourierMapping2d, 
                              FiLM, AttentionPooling, StatsLayer, FNN, Ensemble)

torch.set_float32_matmul_precision('medium')
mp.set_start_method('spawn', force=True)

class ModulationMFNN(nn.Module):
    def __init__(self, input_dim, output_dim, lat_dim, num_latent_layers, activation):
        super().__init__()
        self.activation = getattr(nn.functional, activation)
        self.num_latent_layers = num_latent_layers
        self.fourier_transform = FourierMapping(input_dim=input_dim, num_features=5, scale=0.1)

        self.in_map = nn.Linear(self.fourier_transform.output_dim, lat_dim)
        self.out_map = nn.Linear(lat_dim, output_dim)

    def forward(self, inputs, weights, biases):
        x = self.fourier_transform(inputs)
        u = torch.tanh(torch.bmm(x, weights[0]) + biases[0].unsqueeze(1))
        v = torch.tanh(torch.bmm(x, weights[1]) + biases[1].unsqueeze(1))
        x = self.activation(torch.bmm(x, weights[2]) + biases[2].unsqueeze(1))

        for i in range(3, 3 + self.num_latent_layers):
            x = u * x + v * (1 - x)
            x = self.activation(torch.bmm(x, weights[i]) + biases[i].unsqueeze(1))

        return torch.bmm(x, weights[-1]) + biases[-1].unsqueeze(1)

class HyperNetEncoder(nn.Module):
    def __init__(self, grid_size=224, timm_backbone='', num_pde_coeffs=5, mat_embed_dim=128, coeff_emb_dim=256, stats_emb_dim=32, patch_size=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_pde_coeffs = num_pde_coeffs

        self.mat_embed_dim = mat_embed_dim
        
        self.fourm_bc = FourierMapping2d(2, 5)
        self.fourm_s  = FourierMapping2d(1, 5)
        
        self.proj_b1  = nn.Sequential(
            nn.Conv2d(2, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mat_embed_dim // 2, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj_b2  = nn.Sequential(
            nn.Conv2d(2, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mat_embed_dim // 2, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )
        self.proj_bc = nn.Sequential(
            nn.Conv2d(self.fourm_bc.output_dim, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mat_embed_dim // 2, mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj_s  = nn.Sequential(
            nn.Conv2d(self.fourm_s.output_dim,  mat_embed_dim - mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mat_embed_dim - mat_embed_dim // 2,  mat_embed_dim - mat_embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pos_enc = SinPositionalEncoding2D(mat_embed_dim)
        self.proj_norm = nn.Sequential(nn.Conv2d(mat_embed_dim, mat_embed_dim, kernel_size=1), nn.GroupNorm(max(mat_embed_dim // 32, 1), mat_embed_dim))

        model_args = dict(patch_size=patch_size, window_size=7, embed_dim=mat_embed_dim, depths=(2, 2, 10, 2), num_heads=(4, 8, 8, 16), mlp_ratio=2)
        self.mat_encoder =  _create_swin_transformer(
            timm_backbone, pretrained=False, **dict(model_args, num_classes=0, in_chans=35))

        self.num_features = [self.mat_encoder.feature_info[i]['num_chs'] for i in range(len(self.mat_encoder.feature_info))]

        self.coeff_emb_dim = coeff_emb_dim
        self.coeff_fourier = FourierMapping(num_pde_coeffs, num_features=5)
        self.coeffs_embedding = nn.Sequential(
            self.coeff_fourier,
            nn.Linear(self.coeff_fourier.output_dim, self.coeff_emb_dim),
            nn.GELU(),
        )

        self.cond_layers = nn.ModuleList([FiLM(in_channels=self.num_features[i], conditioning_dim=coeff_emb_dim + stats_emb_dim) 
                                          for i in range(len(self.mat_encoder.layers))])

        self.stats_layer = StatsLayer(input_dim=3, lat_dim=stats_emb_dim)

    def forward(self, pde_coeffs, mat_inputs):
        # log transform the source function, which can be very large
        mat_inputs = torch.cat([mat_inputs[:, :-1], torch.sign(mat_inputs[:, -1:]) * torch.log(1 + torch.abs(mat_inputs[:, -1:]))], dim=1)

        skip_connections = []
        stats_emb = self.stats_layer(mat_inputs[:, 2:])
        coeff_enc = self.coeffs_embedding(pde_coeffs)
        cond_emb = torch.cat([coeff_enc, stats_emb], dim=-1)

        b1_map  = self.proj_b1(mat_inputs[:, :2])
        b2_map  = self.proj_b2(mat_inputs[:, :2])
        bc_map = self.proj_bc(self.fourm_bc(mat_inputs[:, 2:4]))
        s_map  = self.proj_s(self.fourm_s(mat_inputs[:, 4:]))
        mat_emb = self.proj_norm(self.pos_enc(torch.cat([b1_map + b2_map * bc_map, s_map], dim=1))).permute(0, 2, 3, 1)

        for i in range(len(self.mat_encoder.layers)):
            mat_emb = self.mat_encoder.layers[i](mat_emb)
            mat_emb = self.cond_layers[i](mat_emb, cond_emb)
            skip_connections.append(mat_emb)

        return skip_connections

class HyperNetDecoder(nn.Module):
    def __init__(self, num_features, target_net_in_dim, target_net_out_dim, target_net_lat_dim, target_net_n_layers, target_net_activation):
        super().__init__()
        self.num_features = num_features
        self.target_net_in_dim = target_net_in_dim
        self.target_net_lat_dim = target_net_lat_dim
        self.target_net_out_dim = target_net_out_dim
        self.target_net_n_layers = target_net_n_layers
        self.target_net_activation = target_net_activation

        self.target_net = ModulationMFNN(
            input_dim=target_net_in_dim,
            output_dim=target_net_out_dim,
            lat_dim=target_net_lat_dim,
            num_latent_layers=target_net_n_layers, 
            activation=target_net_activation,
        )

        self.attention_pooling = nn.ModuleList([
            AttentionPooling(self.num_features[i], num_heads=[4, 8, 8, 16][i], num_queries=2 * (target_net_n_layers + 4))
            for i in range(4)
        ])

        pool_dim = sum(self.num_features)

        self.weight_maps = nn.ModuleList(
            [self._create_ffn(pool_dim, self.target_net.fourier_transform.output_dim * target_net_lat_dim * 2, 
                              self.target_net.fourier_transform.output_dim * target_net_lat_dim, norm_heads=4) for _ in range(3)] + \
            [self._create_ffn(pool_dim, pool_dim, target_net_lat_dim**2, norm_heads=16) for _ in range(target_net_n_layers)] + \
            [self._create_ffn(pool_dim, target_net_lat_dim * target_net_out_dim * 2, target_net_lat_dim * target_net_out_dim, norm_heads=1)]
        )

        self.bias_maps = nn.ModuleList(
            [self._create_ffn(pool_dim, target_net_lat_dim * 2, target_net_lat_dim, norm_heads=4) for _ in range(target_net_n_layers + 3)] + \
            [self._create_ffn(pool_dim, 32, target_net_out_dim, norm_heads=1)]
        )

    def _create_ffn(self, in_dim, lat_dim, out_dim, act=nn.GELU, out_act=nn.Identity, norm_heads=8):
        out_layer = nn.Linear(lat_dim, out_dim)
        return nn.Sequential(
            nn.Linear(in_dim, lat_dim),
            act(),
            nn.GroupNorm(norm_heads, lat_dim) if norm_heads > 0 else nn.Identity(),
            out_layer,
            out_act(),
        )
        
    def forward(self, skip_connections, xy=None):
        batch_size = len(skip_connections[0])

        comp_features = []
        for att, skip in zip(self.attention_pooling, skip_connections):
            comp_features.append(att(skip.flatten(1, 2)))
        comp_features = torch.cat(comp_features, dim=-1)

        weights = [self.weight_maps[i](comp_features[:, i]).reshape(-1, self.target_net.fourier_transform.output_dim, self.target_net_lat_dim)
                     for i in range(3)] + \
                  [self.weight_maps[i](comp_features[:, i]).reshape(-1, self.target_net_lat_dim, self.target_net_lat_dim) 
                     for i in range(3, self.target_net_n_layers + 3)] + \
                  [self.weight_maps[-1](comp_features[:, self.target_net_n_layers + 3]).reshape(-1, self.target_net_lat_dim, self.target_net_out_dim)]

        biases = [self.bias_maps[i](comp_features[:, self.target_net_n_layers + 4 + i]).reshape(-1, self.target_net_lat_dim) 
                                for i in range(self.target_net_n_layers + 3)] + \
                 [self.bias_maps[-1](comp_features[:, -1]).reshape(-1, self.target_net_out_dim)]

        if xy is not None:
            return self.target_net(xy, weights=weights, biases=biases), weights, biases

        pinns = [partial(self.target_net, weights=[w[i:i+1] for w in weights], 
                         biases=[b[i:i+1] for b in biases]) for i in range(batch_size)]

        return pinns, weights, biases

class HyPINO(pl.LightningModule):
    def __init__(self, grid_size=224, num_pde_coeffs=5, mat_emb_dim=128, coeff_emb_dim=96,
                 target_net_in_dim=2, target_net_out_dim=1, target_net_lat_dim=32, target_net_n_layers=3, 
                 target_net_activation='tanh'):
        super().__init__()
        self.encoder = HyperNetEncoder(grid_size, num_pde_coeffs=num_pde_coeffs, mat_embed_dim=mat_emb_dim, coeff_emb_dim=coeff_emb_dim)
        self.decoder = HyperNetDecoder(num_features=self.encoder.num_features, target_net_in_dim=target_net_in_dim, 
                                       target_net_out_dim=target_net_out_dim, target_net_lat_dim=target_net_lat_dim, 
                                       target_net_n_layers=target_net_n_layers, target_net_activation=target_net_activation)
        
        self.train_losses = []
        self.val_losses = []
        self.train_losses_temp = []
        self.val_losses_temp = []
        self.lr_values = []

    def forward(self, pde_coeffs, mat_inputs, xy=None, return_weights=False):
        pool = self.encoder(pde_coeffs, mat_inputs)
        if return_weights:
            return self.decoder(pool, xy=xy)
        else:
            return self.decoder(pool, xy=xy)[0]

    def _step(self, batch, batch_idx, mode):
        with torch.enable_grad():
            pde_strings = batch['pde_str']
            pde_coeffs = batch['pde_coeffs']
            mat_inputs = batch['mat_inputs']
            u_grid = [u[::4, ::4] if u is not None else None for u in batch['u_grid']]
            f_grid = batch['f_grid'][:, ::4, ::4]

            if mode == 'train':
                domain_coords = batch['xy_domain']
                boundary_coords = batch['xy_boundary']
                dir_boundary, dir_boundary_c = batch['dir_boundary'], batch['dir_boundary_c']
                neu_boundary, neu_boundary_c = batch['neu_boundary'], batch['neu_boundary_c']
                neu_bound_normals = batch['neu_bound_normals']
                f_domain = batch['f_domain']
                ders_domain = batch['ders_domain']
                xy, f = domain_coords, f_domain
                u = batch['u_domain']
            else:
                xy_grid = batch['xy_grid'][:, :, ::4, ::4]
                xy = xy_grid.reshape(xy_grid.size(0), 2, -1).permute(0, 2, 1)
                f = f_grid.reshape(f_grid.size(0), -1, 1)
                u = u_grid

            xy.requires_grad = True
            xis, yis = [xy[j, :, :1] for j in range(len(xy))], [xy[j, :, 1:2] for j in range(len(xy))]
            x, y = torch.stack(xis, dim=0), torch.stack(yis, dim=0)
            xy = torch.cat([x, y], dim=-1)
            
            u_pred = self(pde_coeffs, mat_inputs, xy)

            if mode == 'train':
                boundary_coords.requires_grad = True
                xis_boundary = [boundary_coords[j, :, :1] for j in range(len(boundary_coords))]
                yis_boundary = [boundary_coords[j, :, 1:2] for j in range(len(boundary_coords))]
                x_boundary, y_boundary = torch.stack(xis_boundary, dim=0), torch.stack(yis_boundary, dim=0)
                boundary_coords = torch.cat([x_boundary, y_boundary], dim=-1)
                u_pred_boundary = self(pde_coeffs, mat_inputs, boundary_coords)

            # Initialize loss accumulators
            f_loss, der_loss, u_loss, dir_loss, neu_loss = 0., 0., 0., 0., 0.
            der_counts, u_counts = 0, 0

            for i, pde in enumerate(pde_strings):
                if u[i] is not None:
                    u_loss += huber_loss(u_pred[i], u[i].reshape(-1, 1))
                    u_counts += 1
                diff_op = compile_diff_operator(pde)
                pred_derivatives = compute_derivatives(
                    in_var_map={'x': xis[i], 'y': yis[i]},
                    out_var_map={'u': u_pred[i]},
                    derivatives=ALL_DERIVATIVES
                )
                f_pred = diff_op(pred_derivatives)

                # Domain loss
                f_loss += huber_loss(f_pred, f[i])

                if mode == 'train':
                    # Derivative losses
                    if ders_domain[i] is not None:
                        for key in pred_derivatives:
                            temp_der_loss = huber_loss(pred_derivatives[key], ders_domain[i][key])
                            if key != 'u':
                                der_loss += temp_der_loss
                                der_counts += 1

                    # Dirichlet boundary loss
                    dir_loss += huber_loss(u_pred_boundary[i] * dir_boundary[i], dir_boundary_c[i])

                    # Boundary derivatives
                    boundary_derivatives = compute_derivatives(
                        in_var_map={'x': xis_boundary[i], 'y': yis_boundary[i]},
                        out_var_map={'u': u_pred_boundary[i]},
                        derivatives={'u': [[('x', 1)], [('y', 1)]]},
                    )

                    neu_boundary_pred = (
                        torch.cat([boundary_derivatives['ux'], boundary_derivatives['uy']], dim=-1) * neu_bound_normals[i]
                    ).sum(dim=-1, keepdim=True) * neu_boundary[i]

                    # Neumann boundary loss
                    neu_loss += huber_loss(neu_boundary_pred, neu_boundary_c[i])

                # Debugging and Visualization
                if mode == 'train' and self.steps % 10000 == 1 and i < min(10, len(xy)):
                    assert len(xy) == len(u) == len(u_pred) == len(f)
                    plot_progress(xy[i], boundary_coords[i], u[i], u_pred[i], dir_boundary_c[i], u_pred_boundary[i] * dir_boundary[i], 
                                    f[i], f_pred, self.train_losses, self.val_losses, self.lr_values,
                                    self.run_path, self.steps, mode, i, pde)
                if mode != 'train' and i < min(10, len(xy)):
                    assert len(xy) == len(u) == len(u_pred) == len(f)
                    plot_progress(xy[i], None, u[i], u_pred[i], None, None, 
                                    f[i], f_pred, self.train_losses, self.val_losses, self.lr_values,
                                    self.run_path, self.steps, mode, i, pde)
                    
        f_loss /= len(pde_strings)
        der_loss /= max(der_counts, 1)
        u_loss /= max(u_counts, 1)
        dir_loss /= len(pde_strings)
        neu_loss /= len(pde_strings)

        loss = f_loss * 0.25 + der_loss * 0.25 + u_loss * 1 + dir_loss * 10 + neu_loss * 1

        if loss is not None and torch.isnan(loss).any():
            print("NaN loss detected. Stopping training.")
            self.trainer.should_stop = True

        return {
            mode + '_loss': loss,
            mode + '_f_loss': f_loss,
            mode + '_der_loss': der_loss,
            mode + '_u_loss': u_loss,
            mode + '_dir_loss': dir_loss,
            mode + '_neu_loss': neu_loss,
            'lr': self.optimizer.param_groups[0]["lr"],
        }

    def training_step(self, batch, batch_idx):
        self.steps += 1
        losses = self._step(batch, batch_idx, mode='train')
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=self.batch_size)
        if torch.is_tensor(losses['train_loss']):
            self.train_losses_temp.append(losses['train_loss'].item())
        return losses['train_loss']

    def test_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx, mode='test')
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=self.batch_size)
        return losses['test_loss']

    def validation_step(self, batch, batch_idx):
        losses = self._step(batch, batch_idx, mode='val')
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=self.batch_size)
        if torch.is_tensor(losses['val_loss']):
            self.val_losses_temp.append(losses['val_loss'].item())
        return losses['val_loss']
        
    def on_train_epoch_end(self):
        new_weight_decay = self.weight_decay_lambda(self.current_epoch)
        print('new weight decay:', new_weight_decay)
        for group in self.optimizers().param_groups:
            group["weight_decay"] = new_weight_decay
        self.train_losses.append(np.array(self.train_losses_temp).mean())
        print('')
        print(f'Epoch {self.current_epoch}, train_loss: {self.train_losses[-1]}')
        print('')
        self.train_losses_temp = []
        
    def on_validation_epoch_end(self):
        self.val_losses.append(np.array(self.val_losses_temp).mean())
        self.lr_values.append(self.optimizer.param_groups[0]["lr"])
        print('')
        print(f'Epoch {self.current_epoch}, val_loss:   {self.val_losses[-1]}, lr: {self.optimizer.param_groups[0]["lr"]}')
        print('')
        self.val_losses_temp = []

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), betas=(0.9, 0.999), weight_decay=self.initial_weight_decay, lr=self.lr)
        
        def lr_lambda(current_step: int, min_factor=0.001):
            warmup_steps = self.trainer.estimated_stepping_batches * 0.01
            if current_step < warmup_steps:
                progress = current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (self.trainer.estimated_stepping_batches - warmup_steps)
            return (1 - min_factor) * 0.5 * (1 + np.cos(int(current_step < warmup_steps) * torch.pi + torch.pi * progress)) + min_factor

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.weight_decay_lambda = lambda epoch: self.initial_weight_decay + epoch / self.trainer.max_epochs * \
            (self.final_weight_decay - self.initial_weight_decay)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'step', # or 'epoch'
            }
        }
        
    def fit(self, datamodule, epochs=200, lr=1e-5, name='default', accumulate_grad_batches=1,
            initial_weight_decay=1e-4, final_weight_decay=1e-3, checkpoint=None, devices=-1):
        self.epochs = epochs
        self.batch_size = datamodule.batch_size
        self.initial_weight_decay = initial_weight_decay
        self.final_weight_decay = final_weight_decay

        self.steps = 0
        self.lr = lr
        self.run_name = name + '_' + datetime.now().strftime('%m%d%H%M')
        self.run_path = f'runs/{self.run_name}'
        os.makedirs(self.run_path, exist_ok=True)
        
        best_checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f'{self.run_path}/checkpoints',
            filename=name + '-{train_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{self.run_path}/checkpoints',
            filename=name + '-{epoch}',
        )

        progbar = TQDMProgressBar(refresh_rate=1)

        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=[progbar, checkpoint_callback, best_checkpoint_callback],
            accumulate_grad_batches=accumulate_grad_batches,
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=True),
            devices=devices,
        )

        self.trainer.fit(self, datamodule, ckpt_path=checkpoint)

    @classmethod
    def load_from_safetensors(cls, path, map_location=None, **kwargs):
        model = cls(**kwargs)
        state_dict = load_file(path)
        model.load_state_dict(state_dict, strict=False)
        if map_location is not None:
            model.to(map_location)
        model.eval()
        return model
    
    def iterative_refinement(self,
                            sample: dict,
                            num_iter: int = 10,
                            ensemble_weighting: List[int] = None,
                            plot_progress: bool = False,
                            output_dir: str = None):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if ensemble_weighting is None:
            ensemble_weighting = [1.] * num_iter
        else:
            assert len(ensemble_weighting) == num_iter, \
            f'the length of `ensemble_weighting` should be equal to `num_iter`: {num_iter}, \
                but was {len(ensemble_weighting)}'

        # --- Unpack ---
        gs         = self.encoder.grid_size
        pde_coeffs = sample['pde_coeffs'].unsqueeze(0)
        mat_inputs = sample['mat_inputs'].unsqueeze(0)
        pde_str    = sample['pde_str']

        # aliases
        D_mask = sample['mat_inputs'][0]
        N_mask = sample['mat_inputs'][1]
        g_D    = sample['mat_inputs'][2]
        g_N    = sample['mat_inputs'][3]
        f_tgt  = sample['mat_inputs'][4]

        # optional domain mask
        O_mask = sample['domain_mask'] if 'domain_mask' in sample else torch.ones_like(sample['mat_inputs'][0])

        # optional neumann normals
        N_normals = sample['neu_normals'] if 'neu_normals' in sample else torch.zeros_like(sample['mat_inputs'][:1])

        f_tgt_np = f_tgt.detach().cpu().numpy().reshape(gs, gs)

        # Prepare a (gs x gs) evaluation grid
        x_grid, y_grid = np.meshgrid(
            np.linspace(-1,  1, gs),
            np.linspace( 1, -1, gs),
        )
        x = to_tensor(x_grid, requires_grad=True).reshape(-1, 1).to(self.device)
        y = to_tensor(y_grid, requires_grad=True).reshape(-1, 1).to(self.device)
        xy = torch.cat([x, y], dim=-1)

        # initialize (empty) ensemble
        ensemble = Ensemble(num_nets=0)

        # initial net/preds
        _, init_w, init_b = self(pde_coeffs, mat_inputs, return_weights=True)
        init_net = FNN.from_parameters(init_w, init_b, {
            'input_dim': self.decoder.target_net_in_dim, 'output_dim': self.decoder.target_net_out_dim, 
            'lat_dim': self.decoder.target_net_lat_dim, 'num_latent_layers': self.decoder.target_net_n_layers, 
            'activation': self.decoder.target_net_activation,
        }, remove_batch_dim=True).to(self.device)

        # add initial net to ensemble
        ensemble.add_net(init_net)

        u_pred = ensemble(xy)

        diff_op = compile_diff_operator(pde_str)
        pred_derivs = compute_derivatives(
            in_var_map={'x': x, 'y': y},
            out_var_map={'u': u_pred},
            derivatives=ALL_DERIVATIVES
        )
        f_pred = diff_op(pred_derivs)           # (gs*gs,1)
        neumann_pred = (torch.stack([
            pred_derivs['ux'].detach().view(gs, gs),
            pred_derivs['uy'].detach().view(gs, gs),
        ], dim=0) * N_normals).sum(dim=0)  # (gs,gs)

        # --- Build mat_corr (step 0) and compute residual metrics from it
        u_pred_grid = u_pred.view(gs, gs)
        mat_corr = torch.stack([
            D_mask,                                                  # [0]
            N_mask,                                                  # [1]
            (g_D - u_pred_grid.detach()) * D_mask,                     # [2] = -(r_D)
            (g_N - neumann_pred.detach()) * N_mask,                    # [3] = -(r_N)
            (f_tgt - f_pred.view(gs, gs).detach()) * O_mask,                    # [4] = -(r_f)
        ], dim=0).unsqueeze(0)  # shape (1,5,gs,gs)

        # MSEs
        eps = 1e-12
        mse_D0 = (mat_corr[0,2]**2).sum() / (D_mask.sum() + eps)
        mse_N0 = (mat_corr[0,3]**2).sum() / (N_mask.sum() + eps)
        mse_f0 = (mat_corr[0,4]**2).mean()
        print(f"[init] MSE_D={mse_D0.item():.4e}  MSE_N={mse_N0.item():.4e}  MSE_f={mse_f0.item():.4e}")

        # plots step 0:
        if plot_progress:
            f_pred_np = (f_tgt - mat_corr[0,4]).detach().cpu().numpy()
            plot_grids(
                fields=[
                    u_pred_grid.detach().cpu().numpy(),
                    f_pred_np.reshape(gs, gs),
                    f_tgt_np,
                    np.abs(f_pred_np.reshape(gs, gs) - f_tgt_np),
                    mat_corr[0,2].detach().cpu().numpy(),
                    mat_corr[0,3].detach().cpu().numpy(),
                ],
                titles=['u_pred', 'f_pred', 'f_target', 'abs diff f', 'abs diff D', 'abs diff N'],
                save_path=os.path.join(output_dir, f'step_00.png') if output_dir is not None else None
            )

        for i in range(0, num_iter):
            _, new_w, new_b = self(pde_coeffs, mat_corr, return_weights=True)
            new_net = FNN.from_parameters(new_w, new_b, {
                'input_dim': self.decoder.target_net_in_dim, 'output_dim': self.decoder.target_net_out_dim, 
                'lat_dim': self.decoder.target_net_lat_dim, 'num_latent_layers': self.decoder.target_net_n_layers, 
                'activation': self.decoder.target_net_activation,
            }, remove_batch_dim=True).to(self.device)

            # update ensemble
            ensemble.add_net(new_net, ensemble_weighting[i])

            # predict with updated ensemble
            u_pred = ensemble(xy)
            
            pred_derivs = compute_derivatives(
                in_var_map={'x': x, 'y': y},
                out_var_map={'u': u_pred},
                derivatives=ALL_DERIVATIVES
            )
            f_pred = diff_op(pred_derivs)
            neumann_pred = (torch.stack([
                pred_derivs['ux'].detach().view(gs, gs),
                pred_derivs['uy'].detach().view(gs, gs),
            ], dim=0) * N_normals).sum(dim=0)

            # rebuild mat_corr for next iteration & for metrics/plots
            u_pred_grid = u_pred.view(gs, gs)
            mat_corr = torch.stack([
                D_mask,
                N_mask,
                (g_D - u_pred_grid.detach()) * D_mask,
                (g_N - neumann_pred.detach()) * N_mask,
                (f_tgt - f_pred.view(gs, gs).detach()) * O_mask,
            ], dim=0).unsqueeze(0)

            # residual metrics from mat_corr
            mse_D = (mat_corr[0,2]**2).sum() / (D_mask.sum() + eps)
            mse_N = (mat_corr[0,3]**2).sum() / (N_mask.sum() + eps)
            mse_f = (mat_corr[0,4]**2).mean()
            print(f"[iter {i}] MSE_D={mse_D.item():.4e}  MSE_N={mse_N.item():.4e}  MSE_f={mse_f.item():.4e}")

            if plot_progress:
                delta = new_net(xy).detach().cpu().numpy().reshape(gs, gs)
                f_pred_np = (f_tgt - mat_corr[0,4]).detach().cpu().numpy()
                plot_grids(
                    fields=[
                        u_pred_grid.detach().cpu().numpy(),
                        delta,
                        f_pred_np.reshape(gs, gs),
                        f_tgt_np,
                        np.abs(f_pred_np.reshape(gs, gs) - f_tgt_np),
                        mat_corr[0,2].detach().cpu().numpy(),
                        mat_corr[0,3].detach().cpu().numpy(),
                    ],
                    titles=['u_pred', 'delta', 'f_pred', 'f_target', 'abs diff f', 'abs diff D', 'abs diff N'],
                    save_path=os.path.join(output_dir, f'step_{i:02d}.png') if output_dir is not None else None
                )

        return ensemble