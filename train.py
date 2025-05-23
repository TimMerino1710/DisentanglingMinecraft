from pathlib import Path
from functools import partial
from collections import defaultdict
from collections import OrderedDict

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vision_utils
import lpips
from torchinfo import summary
from torchvision.datasets import LSUN
from torch.utils.data import DataLoader, random_split
import os
from einops import rearrange

from models3d import BiomeClassifier
from data_utils import BlockBiomeConverter, MinecraftDataset, get_minecraft_dataloaders
from visualization_utils import MinecraftVisualizerPyVista, display_minecraft_pyvista
import torch
import numpy as np
import copy
import time
import random
from log_utils import log, log_stats, save_model, save_stats, save_images, save_maps, \
                            display_images, set_up_visdom, config_log, start_training_log, log_hparams_to_json
import visdom
from fq_models import FQModel, VQLossDualCodebook


class HparamsBase(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

class HparamsFQGAN(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        
        if self.dataset == 'minecraft':
            # Existing parameters
            self.batch_size = 8
            self.img_size = 24
            self.n_channels = 42
            self.nf = 64
            self.ndf = 64
            self.res_blocks = 2
            self.latent_shape = [1, 6, 6, 6]
            
            # Structure consistency parameters
            self.with_struct_consistency = False  # Enable structure consistency regularization
            self.struct_consistency_weight = 2.0  # Weight for structure consistency loss
            self.struct_consistency_threshold = 0.72  # Similarity threshold

            # cycle consistency parameters
            self.with_cycle_consistency = True
            self.cycle_consistency_type = 'post_quant_conv'
            self.cycle_consistency_weight = 0.0
            self.disc_gumbel_for_cycle_input = True
            self.cycle_start_step = 0
            
            self.struct_codebook_size = 32  # Size of each codebook
            self.style_codebook_size = 16  # Size of each codebook
            self.emb_dim = 8  # Embedding dimension
            self.z_channels = 8  # Bottleneck channels
            self.ch_mult = [1, 2, 4]  # Channel multipliers for progressive downsampling
            self.num_resolutions = len(self.ch_mult)
            self.attn_resolutions = [6]  # Resolutions at which to apply attention
            
            # Loss weights and parameters
            self.disc_type = 'conv'
            self.disc_weight_max = 1.0  # Weight for discriminator loss
            self.disc_weight = 0.5
            self.disc_weight_min = 0.0  # Weight for discriminator loss
            self.disc_adaptive_weight = False  # Enable adaptive weighting
            self.reconstruction_weight = 1.0  # Weight for reconstruction loss
            self.codebook_weight = 1.0  # Weight for codebook loss
            self.biome_weight = 1.0  # Weight for biome feature prediction
            self.disentanglement_ratio = 0.5  # Weight for disentanglement loss
            
            # Codebook specific parameters
            self.quantizer_type = 'ema'
            self.beta = 0.5  # Commitment loss coefficient
            self.entropy_loss_ratio = 0.2
            self.codebook_l2_norm = True  
            self.codebook_show_usage = True  
            self.ema_decay = 0.99
            
            # Training parameters
            self.lr = 1e-4  # Learning rate
            self.beta1 = 0.9  # Adam beta1
            self.beta2 = 0.95  # Adam beta2
            self.disc_layers = 1  # Number of discriminator layers
            self.train_steps = 15000
            self.disc_start_step = 15000  # Step to start discriminator training

            self.start_step = 0
            
            self.transformer_dim = self.emb_dim  # Make transformer dim match embedding dim
            self.num_heads = 8  # Number of attention heads
            
            # Feature prediction parameters
            self.with_biome_supervision = False  # Enable biome feature prediction
            self.with_disentanglement = True  # Enable disentanglement loss
            
            self.steps_per_log = 150
            self.steps_per_checkpoint = 1000
            self.steps_per_display_output = 500
            self.steps_per_save_output = 500
            self.steps_per_validation = 150
            self.val_samples_to_save = 16
            self.val_samples_to_display = 4
            self.visdom_port = 8097
            
            # Two stage decoder stuff
            self.binary_reconstruction_weight = 3
            self.two_stage_decoder = True
            self.multiply_decoder = False
            self.use_dumb_decoder = False
            self.combine_method = 'concat'
            self.detach_binary_recon = True

            # Weighted recon loss:
            self.block_weighting = True
            self.weighted_block_amount = 3.0
            self.weighted_block_indices = []
             
            self.num_biomes = 11  # Number of biome classes
            self.biome_feat_dim = 256  # Dimension of biome features
            self.biome_classifier_path = 'best_biome_classifier_airprocessed.pt'

            self.disc_gumbel = True
            self.gumbel_tau = 1
            self.gumbel_hard = True
            self.disc_argmax_ste = False

            self.padding_mode = 'reflect'

            self.weight_decay = 0.0

            self.air_class_index = 0
            
        else:
            raise KeyError(f'Defaults not defined for dataset: {self.dataset}')
        
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = HparamsFQGAN(dataset='minecraft')
    H.log_dir = 'logdir'
    H.load_dir = 'loaddir'

    vis = visdom.Visdom(port=H.visdom_port)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')

    start_training_log(H)
    log_hparams_to_json(H, H.log_dir)

    data_path = '24_newdataset_processed_cleaned3.pt'
    mappings_path = '.24_newdataset_mappings3.pt'
    # visualizer = MinecraftVisualizer()
    visualizer = MinecraftVisualizerPyVista()
    train_loader, val_loader = get_minecraft_dataloaders(
        data_path,
        batch_size=H.batch_size,
        num_workers=0,
        val_split=0.1,
        augment=True,
        vert=True
        # save_val_path=f'../../text2env/data/{H.log_dir}_valset.pt'
    )
    block_converter = BlockBiomeConverter.load_mappings(mappings_path)
    air_idx = block_converter.get_air_block_index()
    water_idx = block_converter.get_water_block_index()
    print(f'setting air idx in hparams to {air_idx}')
    H.air_class_index = air_idx

    log_idxs = block_converter.get_blockid_indices([131, 132])
    print(f'log indices: {log_idxs}')
    H.weighted_block_indices = log_idxs

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    # Initialize model and loss
    vqgan = FQModel(H).cuda()
    vq_loss = VQLossDualCodebook(H).cuda()

    train_iterator = cycle(train_loader)
    if val_loader is not None:
        val_iterator = cycle(val_loader)

    # Setup optimizers 
    optimizer_g = torch.optim.Adam(
        vqgan.parameters(), 
        lr=H.lr,
        betas=(H.beta1, H.beta2),
        weight_decay=H.weight_decay
    )
    optimizer_d = torch.optim.Adam(
        vq_loss.discriminator.parameters(),
        lr=H.lr,
        betas=(H.beta1, H.beta2),
        weight_decay=H.weight_decay
    )

    print(f'Using weight decay {H.weight_decay}')
    # Initialize loss tracking
    plotted_loss_terms = [
        'rec_loss', 'binary_rec_loss', 'style_loss', 'struct_loss', 
        'biome_feat_loss', 'disent_loss', 'g_loss', 'd_loss', 'struct_consistency_loss', 'cycle_consistency_loss'
    ]
    loss_arrays = {term: np.array([]) for term in plotted_loss_terms}
    codebook_usage = {
        'style': np.array([]),
        'struct': np.array([])
    }

    # Training loop
    for step in range(0, H.train_steps):
        step_start_time = time.time()
        batch = next(train_iterator)
        
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        x = x.cuda()

        
        
        binary_target = (torch.argmax(x, dim=1) != air_idx).float().cuda()
        if H.two_stage_decoder:
            recons, binary_out, codebook_loss_style, codebook_loss_struct, disentangle_loss, biome_feat, struct_consistency, cycle_consistency = vqgan(x)
        else:
            recons, codebook_loss_style, codebook_loss_struct, disentangle_loss, biome_feat, struct_consistency. cycle_consistency = vqgan(x)
            binary_out = None

        
        # Generator/Encoder update
        optimizer_g.zero_grad()
        # Calculate generator losses
        loss_dict = vq_loss(
            codebook_loss_style, codebook_loss_struct,
            x, recons, disentangle_loss, biome_feat,
            optimizer_idx=0,
            global_step=step,
            binary_out=binary_out,
            binary_target=binary_target,
            struct_consistency_loss = struct_consistency,
            cycle_consistency_loss=cycle_consistency
        )
            
            
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(vqgan.parameters(), max_norm=1.0)
        optimizer_g.step()

        
    d_loss_dict = {}

    # Discriminator update
    if step >= H.disc_start_step:
        optimizer_d.zero_grad()
        
        # Calculate discriminator losses
        d_loss_dict = vq_loss(
            codebook_loss_style, codebook_loss_struct,
            x, recons, disentangle_loss, biome_feat,
            optimizer_idx=1,
            global_step=step,
            binary_out=binary_out,
            binary_target=binary_target
        )

        # print(f"Final d_loss: {d_loss_dict}")        d_loss_dict['d_loss'].backward()
        torch.nn.utils.clip_grad_norm_(vq_loss.discriminator.parameters(), max_norm=1.0)
        optimizer_d.step()

    # Logging
    if step % H.steps_per_log == 0:
        print(f"\nStep {step}:")
        for term in plotted_loss_terms:
            if term in loss_dict:
                value = loss_dict[term].item() if torch.is_tensor(loss_dict[term]) else loss_dict[term]
                loss_arrays[term] = np.append(loss_arrays[term], value)
                print(f"{term}: {value:.4f}")
            elif term == 'd_loss':
                if step >= H.disc_start_step:
                    value = d_loss_dict['d_loss'].item() if torch.is_tensor(d_loss_dict['d_loss']) else d_loss_dict['d_loss']
                else:
                    value = 0.0
                loss_arrays[term] = np.append(loss_arrays[term], value)
                print(f"{term}: {value:.4f}")

        # Update codebook usage tracking
        style_usage = loss_dict['codebook_usage_style'] if torch.is_tensor(loss_dict['codebook_usage_style']) else loss_dict['codebook_usage_style']
        struct_usage = loss_dict['codebook_usage_struct'] if torch.is_tensor(loss_dict['codebook_usage_struct']) else loss_dict['codebook_usage_struct']
        
        codebook_usage['style'] = np.append(codebook_usage['style'], style_usage)
        codebook_usage['struct'] = np.append(codebook_usage['struct'], struct_usage)
        print(f"Codebook Usage - Style: {style_usage:.2f}%, Structure: {struct_usage:.2f}%")
        # Plot in Visdom
        x_axis = list(range(0, step+1, H.steps_per_log))
        
        # Individual loss plots
        for term in plotted_loss_terms:
            if len(loss_arrays[term]) > 0:
                vis.line(
                    loss_arrays[term],
                    x_axis,
                    win=f'{term}_plot',
                    opts=dict(title=f'{term} Loss')
                )

        # Combined loss plot
        if len(x_axis) > 1:
            vis.line(
                Y=np.column_stack([loss_arrays[term] for term in plotted_loss_terms if len(loss_arrays[term]) > 0]),
                X=np.column_stack([x_axis for _ in plotted_loss_terms if len(loss_arrays[_]) > 0]),
                win='all_losses',
                opts=dict(title='All Losses', legend=[t for t in plotted_loss_terms if len(loss_arrays[t]) > 0])
            )

        # Codebook usage plot
        vis.line(
            Y=np.column_stack([codebook_usage['style'], codebook_usage['struct']]),
            X=np.column_stack([x_axis, x_axis]),
            win='codebook_usage',
            opts=dict(title='Codebook Usage', legend=['Style Codebook', 'Structure Codebook'])
        )

    # Visualization of reconstructions
    if step % H.steps_per_display_output == 0 or step == H.train_steps - 1:
        print("Rendering...")
        # Convert indices back to original block IDs for visualization
        orig_blocks = block_converter.convert_to_original_blocks(x)
        recon_blocks = block_converter.convert_to_original_blocks(recons)
        
        if step % H.steps_per_save_output == 0:
            log_dir = f"../model_logs/{H.log_dir}/images"
            os.makedirs(log_dir, exist_ok=True)

            display_minecraft_pyvista(vis, visualizer, orig_blocks, win_name='Original Maps', title=f'Original Maps step {step}', save_path=f"{log_dir}/orig_{step}.png", nrow=8)
            display_minecraft_pyvista(vis, visualizer, recon_blocks, win_name='Reconstructed Maps', title=f'Reconstructed Maps step {step}', save_path=f"{log_dir}/recon_{step}.png", nrow=8)
        else:
            display_minecraft_pyvista(vis, visualizer, orig_blocks, win_name='Original Maps', title=f'Original Maps step {step}', nrow=8)
            display_minecraft_pyvista(vis, visualizer, recon_blocks, win_name='Reconstructed Maps', title=f'Reconstructed Maps step {step}', nrow=8)
        print("Done Rendering")

    # Save checkpoints
    if (step % H.steps_per_checkpoint == 0 and step > 0) or step == H.train_steps - 1:
        save_model(vqgan, 'fqgan', step, H.log_dir)
        save_model(optimizer_g, 'optimizer_g', step, H.log_dir)
        save_model(optimizer_d, 'optimizer_d', step, H.log_dir)

        train_stats = {
            'losses': loss_arrays,
            'codebook_usage': codebook_usage,
            'steps_per_log': H.steps_per_log,
        }
        save_stats(H, train_stats, step)


if __name__ == "__main__":
    main()