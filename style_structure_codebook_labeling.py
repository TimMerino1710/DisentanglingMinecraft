import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sampler_utils import retrieve_autoencoder_components_state_dicts, latent_ids_to_onehot3d, get_latent_loaders
from models3d import VQAutoEncoder, Generator
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch.distributions as dists
from tqdm import tqdm
import gc
from models3d import BiomeClassifier

from data_utils import BlockBiomeConverter, MinecraftDataset, get_minecraft_dataloaders
import os
from log_utils import log, load_stats, load_model
import copy
from fq_models import FQModel, HparamsFQGAN
from visualization_utils import MinecraftVisualizerPyVista



CHECKPOINT_STEPS = [12000] 
MODEL_BASE_PATH = "model_path" 
OUTPUT_DIR = "output_path"
SHARP_MAD_THRESHOLD = 0.3
SHARP_ENTROPY_THRESHOLD = 0.5
CONS_VAR_THRESHOLD = 0.15
MIN_CHUNKS_THRESHOLD = 10
DISTANCE_METRIC = 'mae'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# FQGAN helper functions
# Loads hparams from hparams.json file in saved model directory
def load_hparams_from_json(log_dir):
    import json
    import os
    json_path = os.path.join(log_dir, 'hparams.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No hparams.json file found in {log_dir}")
    
    with open(json_path, 'r') as f:
        hparams = json.load(f)

    return hparams

# turns loaded hparams json into propery hyperparams object
def dict_to_vcqgan_hparams(hparams_dict, dataset=None):
    # Determine which hyperparameter class to use based on the dataset
    if dataset == None:
        dataset = hparams_dict.get('dataset', 'MNIST')  # Default to MNIST if not specified
    
    vq_hyper = HparamsFQGAN(dataset)
    # Set attributes from the dictionary
    for key, value in hparams_dict.items():
        setattr(vq_hyper, key, value)
    
    return vq_hyper

# Loads fqgan model weights from a given checkpoint file
def load_fqgan_from_checkpoint(H, fqgan):
    fqgan = load_model(fqgan, "fqgan", H.load_step, H.load_dir).cuda()
    fqgan.eval()
    return fqgan

# Takes a chunk or batch of chunks from the dataset, returns the encoded style and structure indices matrices
def encode_and_quantize(fqgan, terrain_chunks, device='cuda'):
    """Memory-efficient encoding function"""
    fqgan.eval()
    with torch.no_grad():
        # Move input to device
        terrain_chunks = terrain_chunks.to(device)
        
        # Get encodings
        h_style, h_struct = fqgan.encoder(terrain_chunks)
        
        # Process style path
        h_style = fqgan.quant_conv_style(h_style)
        quant_style, _, style_stats = fqgan.quantize_style(h_style)
        style_indices = style_stats[2]  # Get indices from tuple
        style_indices = style_indices.view(
            (h_style.size()[0], h_style.size()[2], h_style.size()[3], h_style.size()[4])
        )
        
        # Clear intermediate tensors
        del h_style, quant_style, style_stats
        
        # Process structure path
        h_struct = fqgan.quant_conv_struct(h_struct)
        quant_struct, _, struct_stats = fqgan.quantize_struct(h_struct)
        struct_indices = struct_stats[2]  # Get indices from tuple
        struct_indices = struct_indices.view(
            (h_struct.size()[0], h_struct.size()[2], h_struct.size()[3], h_struct.size()[4])
        )
        
        # Clear intermediate tensors
        del h_struct, quant_struct, struct_stats
        
        # Move indices to CPU to save GPU memory
        style_indices = style_indices.cpu()
        struct_indices = struct_indices.cpu()
        
        torch.cuda.empty_cache()
        
        return style_indices, struct_indices

# Takes style and structure indices, returns the reconstructed map
def decode_from_indices(style_indices, struct_indices, fqgan, device='cuda', two_stage=False):
    """Memory-efficient decoding function"""
    with torch.no_grad():
        # Move indices to device only when needed
        style_indices = style_indices.to(device)
        struct_indices = struct_indices.to(device)
        
        # Get quantized vectors
        quant_style = fqgan.quantize_style.get_codebook_entry(
            style_indices.view(-1),
            shape=[1, fqgan.embed_dim, *style_indices.shape[1:]]
        )
        quant_struct = fqgan.quantize_struct.get_codebook_entry(
            struct_indices.view(-1),
            shape=[1, fqgan.embed_dim, *struct_indices.shape[1:]]
        )
        
        # Clear indices from GPU
        del style_indices, struct_indices
        
        # Combine and decode
        quant = torch.cat([quant_struct, quant_style], dim=1)
        # quant = quant_style + quant_struct
        del quant_style, quant_struct
        
        if two_stage:
            decoded, binary_decoded = fqgan.decoder(quant)
        else:
            decoded = fqgan.decoder(quant)
        
        del quant
        
        # Convert to block IDs if one-hot encoded
        if decoded.shape[1] > 1:
            decoded = torch.argmax(decoded, dim=1)
        
        # Move result to CPU and clear GPU memory
        result = decoded.squeeze(0).cpu()
        if two_stage:
            binary_result = binary_decoded.squeeze(0).cpu()
            del decoded
            del binary_decoded
            torch.cuda.empty_cache()
            return result, binary_result
        
        del decoded
        torch.cuda.empty_cache()
        
        return result
    

def plot_positional_frequency(position_counts, struct_code, save_dir):
    """
    Generates and saves a 3D scatter plot showing the positional frequency
    of a structure code within the latent grid, using a Viridis colormap and
    Minecraft coordinate conventions (Y=Height is vertical, Y=0 is bottom).

    Args:
        position_counts (dict): Maps struct_code to 6x6x6 count tensor.
        struct_code (int): The structure code to visualize.
        save_dir (str): The directory to save the plot image.
    """
    if struct_code not in position_counts:
        print(f"Error: Code {struct_code} not found in position_counts dictionary.")
        return

    counts_tensor = position_counts[struct_code].cpu() # Ensure it's on CPU
    latent_shape = counts_tensor.shape
    if len(latent_shape) != 3:
        print(f"Error: Count tensor for code {struct_code} is not 3D (shape: {latent_shape}).")
        return

    counts_numpy = counts_tensor.numpy()
    max_count = np.max(counts_numpy)

    if max_count == 0:
        print(f"Info: Code {struct_code} never appeared. Skipping visualization.")
        # Optional: Create an empty plot placeholder if desired
        return # Stop here if code never appeared

    # --- Plotting Setup ---
    # os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid indices (I, J, K) corresponding to tensor dimensions
    # Assume I=Depth(Z_mc), J=Height(Y_mc), K=Width(X_mc)
    i_indices, j_indices, k_indices = np.indices(latent_shape)

    # Flatten coordinates and the raw counts (for color mapping)
    i_coords_mc = i_indices.flatten() # Minecraft Z coordinates
    j_coords_mc = j_indices.flatten() # Minecraft Y (Height) coordinates
    k_coords_mc = k_indices.flatten() # Minecraft X coordinates
    frequencies = counts_numpy.flatten()

    # --- Create Scatter Plot (Viridis Colormap, Correct Axis Mapping) ---
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('Greys')
    # Plot mapping:
    # Plot X-axis <- Minecraft X data (k_coords_mc)
    # Plot Y-axis <- Minecraft Z data (i_coords_mc)
    # Plot Z-axis <- Minecraft Y (Height) data (j_coords_mc) <<< VERTICAL AXIS
    scatter = ax.scatter(k_coords_mc, i_coords_mc, j_coords_mc, # Correct mapping
                         c=frequencies, cmap=cmap, # Color based on frequency counts
                         s=150, # Adjust size as needed
                         alpha=0.8, # Add some transparency
                         # vmin=0, vmax=max_count, # Optional: Explicitly set color limits
                         edgecolors='grey', linewidth=0.5)

    # --- Add Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Code Occurrence Count")

    # --- Set Labels, Ticks, Limits, Title, and Invert Z-axis ---
    ax.set_title(f"Code {struct_code} Positional Frequency")
    # Label plot axes according to the *Minecraft dimension* plotted on them
    ax.set_xlabel("Z (Latent Dim K)")
    ax.set_ylabel("X (Latent Dim I)")
    ax.set_zlabel("Y (Height, Latent Dim J)") # Vertical axis

    # Set ticks based on the dimension size
    ax.set_xticks(np.arange(latent_shape[2])) # K dimension
    ax.set_yticks(np.arange(latent_shape[0])) # I dimension
    ax.set_zticks(np.arange(latent_shape[1])) # J dimension

    # Set limits for plot axes
    ax.set_xlim(-0.5, latent_shape[2] - 0.5)
    ax.set_ylim(-0.5, latent_shape[0] - 0.5)
    ax.set_zlim(-0.5, latent_shape[1] - 0.5)

    # --- Invert the Z-axis (which represents Y-Height) ---
    # ax.invert_zaxis() # Ensures Y=0 is at the bottom
    ax.invert_yaxis() # Ensures Y=0 is at the bottom

    # Adjust view angle
    ax.view_init(elev=20., azim=-75)
    # plt.show()
    # --- Save and Close ---
    image_path = os.path.join(save_dir, f"pos_freq_code_{struct_code}.png")
    try:
        plt.savefig(image_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving positional frequency plot for code {struct_code}: {e}")
    finally:
        plt.close(fig)


def generate_all_positional_frequency_plots(position_counts, output_dir):
    print(f"Generating positional frequency plots for {len(position_counts)} codes...")
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    # Iterate through the codes present in the dictionary
    codes_to_plot = sorted(position_counts.keys())

    for i, code in enumerate(codes_to_plot):
        # Call the plotting function for the current code
        plot_positional_frequency(position_counts, code, output_dir)

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(codes_to_plot):
            print(f"Generated plot {i+1}/{len(codes_to_plot)} (Code {code})")

    print(f"Finished generating positional frequency plots. Saved to: {output_dir}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from PIL import Image


def _setup_3d_axis_for_pattern_visualization(ax, title, ticks_range=np.arange(4)):
    """Helper to consistently format 3D axes for pattern plots."""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (MC)", fontsize=8)
    ax.set_ylabel("Z (MC)", fontsize=8)
    ax.set_zlabel("Y (MC H)", fontsize=8) # MC Height

    ax.set_xticks(ticks_range)
    ax.set_yticks(ticks_range)
    ax.set_zticks(ticks_range)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(-0.5, 3.5)
    ax.invert_zaxis() # Minecraft Y=0 is typically bottom
    ax.view_init(elev=25., azim=-125) # Consistent view angle
    ax.tick_params(axis='both', which='major', labelsize=7)

def crop_pil_image(img, side_px_crop, top_px_crop):
    original_w, original_h = img.size
    left = side_px_crop // 2
    right = original_w - (side_px_crop // 2)
    top = top_px_crop // 2
    bottom = original_h - (top_px_crop // 2)
    cropped = img.crop((left, top, right, bottom))
    return cropped


def visualize_top_binary_patterns(
    binary_structure_dict: dict,
    save_dir: str,
    num_top_patterns_to_show_3d: int = 5, # For 3D visualization
    num_top_patterns_for_hist: int = 100, # For 2D histogram
    base_plot_size_3d: tuple = (4.5, 4)
):
    """
    Analyzes binary_structure_dict to:
    1. Visualize the top N (num_top_patterns_to_show_3d) most frequent unique
       4x4x4 patterns in 3D.
    2. Plot a histogram of frequencies for the top M (num_top_patterns_for_hist)
       unique patterns.

    Args:
        binary_structure_dict (dict): Maps structure codes (int) to lists of
                                      binary tensors (torch.Tensor, 0s and 1s).
                                      Tensors are expected to be 4x4x4 or 4x4x4x1.
        save_dir (str): Directory to save the output plots.
        num_top_patterns_to_show_3d (int): Number of top patterns for 3D visualization.
        num_top_patterns_for_hist (int): Number of top patterns for the 2D histogram.
        base_plot_size_3d (tuple): Base size for each 3D subplot.
    """
    os.makedirs(save_dir, exist_ok=True)

    for code, chunk_list in binary_structure_dict.items():
        if not chunk_list:
            print(f"Info: Code {code} has no associated chunks. Skipping pattern visualization.")
            continue

        pattern_counts = Counter()
        unique_pattern_tensors_cpu = {} # For 3D visualization

        for i, binary_tensor_original in enumerate(chunk_list):
            if not isinstance(binary_tensor_original, torch.Tensor):
                print(f"Warning: Item {i} for code {code} is not a tensor. Skipping.")
                continue
            binary_tensor = binary_tensor_original.squeeze()
            if binary_tensor.ndim != 3 or binary_tensor.shape != (4, 4, 4):
                print(f"Warning: Skipping non-4x4x4 tensor (shape: {binary_tensor_original.shape}) for code {code}, item {i}.")
                continue
            
            pattern_np = binary_tensor.cpu().numpy().astype(np.int16)
            pattern_key = tuple(pattern_np.flatten())

            pattern_counts[pattern_key] += 1
            if pattern_key not in unique_pattern_tensors_cpu: # Store for 3D vis if needed
                if len(unique_pattern_tensors_cpu) < num_top_patterns_to_show_3d * 2: # Optimization: only store if potentially in top N for 3D
                     unique_pattern_tensors_cpu[pattern_key] = binary_tensor.cpu()


        if not pattern_counts:
            print(f"Info: No valid patterns found for code {code} after processing. Skipping visualization.")
            continue

        total_samples_for_code = sum(pattern_counts.values())
        if total_samples_for_code == 0:
            print(f"Info: Total samples for code {code} is zero. Skipping visualization.")
            continue
            
        all_sorted_patterns = pattern_counts.most_common() 

        if not all_sorted_patterns:
            print(f"Info: No patterns to display for code {code} after sorting. Skipping plots.")
            continue

        # --- 1. Visualize Top N 3D Patterns ---
        patterns_for_3d_vis = all_sorted_patterns[:num_top_patterns_to_show_3d]
        num_patterns_actually_plotted_3d = len(patterns_for_3d_vis)

        
        pyvista_dir = os.path.join(save_dir, f"code_{code}_pyvista/")
        os.makedirs(pyvista_dir, exist_ok=True) # Ensure vis_dir exists for caching
        # --- Cache file path for collected data ---
        # nice pyvista plot
        fig_3d = plt.figure(figsize=(num_patterns_actually_plotted_3d, 4))
        for i, (pattern_key, count) in enumerate(patterns_for_3d_vis):
            if pattern_key in unique_pattern_tensors_cpu:
                    pattern_tensor_for_vis = unique_pattern_tensors_cpu[pattern_key]
            else: # Fallback: recreate from key if it wasn't stored due to optimization
                # print('not found, recreating')
                pattern_tensor_for_vis = torch.from_numpy(np.array(pattern_key, dtype=np.int16).reshape(4,4,4))
                # print(f'top pattern shape: {pyvisa_pattern.shape}, uniques: {np.unique(pyvisa_pattern)}')
            # pattern_tensor_for_vis = unique_pattern_tensors_cpu[pattern_key]
            pyvisa_pattern = np.copy(pattern_tensor_for_vis)
            # print(f'top pattern shape: {pyvisa_pattern.shape}, uniques: {np.unique(pyvisa_pattern)}')
            pyvisa_pattern[pyvisa_pattern == 1] = 217
            pyvisa_pattern[pyvisa_pattern == 0] = 5
            # pyvisa_pattern = pyvisa_pattern.unsqueeze(0)
            # print(f'top pattern shape: {pyvisa_pattern.shape}, uniques: {np.unique(pyvisa_pattern)}')
            plotter = visualizer.visualize_chunk(pyvisa_pattern, show_axis=False)
            plotter.camera.zoom(5.0)
            top_n_path = os.path.join(pyvista_dir, f"top_{i}.png")
            frame = plotter.screenshot(transparent_background=True, return_img=True)
            pil_frame = Image.fromarray(frame)
            cropped = crop_pil_image(pil_frame, side_px_crop=220, top_px_crop=120)
            ax = fig_3d.add_subplot(1, num_patterns_actually_plotted_3d, i + 1)
            ax.imshow(cropped)

            percentage = (count / total_samples_for_code) * 100
            ax.set_title(f"{percentage:.1f}%)", fontsize=14)
        plot_filename_3d = f"top_binary_patterns2_code_{code}.png"
        full_save_path_3d = os.path.join(save_dir, plot_filename_3d)
        plt.savefig(full_save_path_3d, dpi=150)
        plt.close(fig_3d)
            # cropped.save(top_n_path)

        # --- 2. Plot Histogram of Top M Pattern Frequencies ---
        patterns_for_histogram = all_sorted_patterns[:num_top_patterns_for_hist]
        num_patterns_in_hist = len(patterns_for_histogram)

        if num_patterns_in_hist > 0:
            pattern_frequencies_hist = [item[1] for item in patterns_for_histogram] 
            pattern_percentages_hist = [(freq / total_samples_for_code) * 100 for freq in pattern_frequencies_hist]
            
            # Figure width based on number of patterns in histogram, up to a max width.
            # e.g., 0.15 inches per pattern, min 10, max 20-25 inches.
            hist_fig_width = min(max(10, num_patterns_in_hist * 0.15), 25) 
            
            fig_hist, ax_hist = plt.subplots(figsize=(hist_fig_width, 6))
            
            x_ticks_labels_hist = [f"P{i+1}" for i in range(num_patterns_in_hist)] 
            bar_positions_hist = np.arange(num_patterns_in_hist)

            ax_hist.bar(bar_positions_hist, pattern_percentages_hist, color='cornflowerblue', width=0.8)
            
            ax_hist.set_xlabel("Unique Pattern Rank (Most to Least Frequent)", fontsize=12)
            ax_hist.set_ylabel("Frequency (%)", fontsize=12)
            ax_hist.set_title(f"Distribution of Top {num_patterns_in_hist} Unique Patterns for Code {code} (Total Samples: {total_samples_for_code})", fontsize=14)
            
            num_ticks_to_show_hist = min(num_patterns_in_hist, 30) 
            if num_patterns_in_hist > num_ticks_to_show_hist:
                tick_indices = np.linspace(0, num_patterns_in_hist - 1 , num_ticks_to_show_hist, dtype=int)
                valid_tick_indices = [ti for ti in tick_indices if ti < len(bar_positions_hist)] # Should always be true here
                ax_hist.set_xticks(bar_positions_hist[valid_tick_indices])
                ax_hist.set_xticklabels([x_ticks_labels_hist[i] for i in valid_tick_indices], rotation=45, ha="right")
            else:
                ax_hist.set_xticks(bar_positions_hist)
                ax_hist.set_xticklabels(x_ticks_labels_hist, rotation=45, ha="right")

            ax_hist.set_ylim(0, max(pattern_percentages_hist) * 1.1 if pattern_percentages_hist else 10) 
            ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plot_filename_hist = f"top_patterns_histogram_code_{code}.png" # Renamed for clarity
            full_save_path_hist = os.path.join(save_dir, plot_filename_hist)
            try:
                plt.savefig(full_save_path_hist, dpi=150)
                print(f"Saved top {num_patterns_in_hist} patterns histogram for code {code} to {full_save_path_hist}")
            except Exception as e:
                print(f"Error saving top patterns histogram for code {code}: {e}")
            finally:
                plt.close(fig_hist)
        else:
            print(f"Info: No patterns to plot in histogram for code {code}.")

import random
def visualize_structure_grid(structure_dict, struct_code, num_chunks=36, grid_size=8, spacing=2, save_path=None, code_type="Struct"):
    """
    Visualize a grid of 4x4x4 chunks for a specific structure code.
    
    Parameters:
    - structure_dict: Dictionary containing chunks for each structure code
    - struct_code: The structure code to visualize
    - num_chunks: Number of chunks to display (default: 64)
    - grid_size: Number of chunks per row/column (default: 8)
    - spacing: Number of blocks spacing between chunks (default: 2)
    
    Returns:
    - PyVista plotter with interactive visualization
    """
    import pyvista as pv
    import numpy as np
    import torch
    
    # Check if the structure code exists
    if struct_code not in structure_dict:
        print(f"Structure code {struct_code} not found in dictionary")
        return None
    
    # Randomly select num_chunks from the list
    available_chunks = structure_dict[struct_code]
    actual_chunks = len(available_chunks)
    
    if actual_chunks == 0:
        print(f"No chunks found for structure code {struct_code}")
        return None
    
    if actual_chunks < num_chunks:
        print(f"Only {actual_chunks} chunks available for structure code {struct_code}")
        chunks = available_chunks
    else:
        # Randomly select num_chunks from the list
        chunks = random.sample(available_chunks, num_chunks)
    
    # Setup block colors mapping (simplified from the visualizer)
    blocks_to_cols = {
            0: (0.5, 0.25, 0.0),    # light brown
            10: 'black', # bedrock
            29: "#006400", # cacutus
            38: "#B8860B",  # clay
            60: "brown",  # dirt
            92: "gold",  # gold ore
            93: "green",  # grass
            115: "brown",  # ladder...?
            119: (.02, .28, .16, 0.9),  # transparent forest green (RGBA) for leaves
            120: (.02, .28, .16, 0.9),  # leaves2
            194: "yellow",  # sand
            217: "gray",  # stone
            240: (0.0, 0.0, 1.0, 0.4),  # water
            227: (0.0, 1.0, 0.0, .3), # tall grass
            237: (0.33, 0.7, 0.33, 0.3), # vine
            40: "#2F4F4F",  # coal ore
            62: "#228B22",  # double plant
            108: "#BEBEBE",  # iron ore
            131: "saddlebrown",  # log1
            132: "saddlebrown",  #log2
            95: "lightgray",  # gravel
            243: "wheat",  # wheat
            197: "limegreen",  # sapling
            166: "orange",  #pumpkin
            167: "#FF8C00",  # pumpkin stem
            184: "#FFA07A",  # red flower
            195: "tan",  # sandstone
            250: "white",  #wool 
            251: "gold",   #yellow flower
        }
    
    # Calculate grid dimensions
    chunk_size = 4
    grid_dim = grid_size * chunk_size + (grid_size - 1) * spacing
    
    # Create plotter
    plotter = pv.Plotter(notebook=True)
    
    # Remove existing lights and add custom lighting
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(1, -1, 1), intensity=1.0, color='white'))
    plotter.add_light(pv.Light(position=(-1, 1, 0.5), intensity=0.5, color='white'))
    plotter.add_light(pv.Light(position=(-0.5, -0.5, -1), intensity=0.3, color='white'))
    plotter.add_title(f"{code_type} Code {struct_code} - Pattern Visualization", font_size=16)
    # Place each chunk in the grid
    for i in range(min(actual_chunks, num_chunks)):
        # Calculate grid position
        row = i // grid_size
        col = i % grid_size
        
        # Calculate offset in the grid
        x_offset = col * (chunk_size + spacing)
        z_offset = row * (chunk_size + spacing)
        
        # Get current chunk
        chunk = chunks[i]
        chunk = block_converter.convert_to_original_blocks(chunk)
        # Convert to numpy if needed
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().cpu().numpy()
        
        # Apply the same transformations as original visualizer
        chunk = chunk.transpose(2, 0, 1)
        chunk = np.rot90(chunk, 1, (0, 1))
        
        # Convert encoded blocks to original block IDs
        
        # Create grid for this chunk
        grid = pv.ImageData()
        grid.dimensions = np.array(chunk.shape) + 1
        grid.origin = (x_offset, z_offset, 0)  # Position in the overall grid
        grid.spacing = (1, 1, 1)  # Unit spacing
        grid.cell_data["values"] = chunk.flatten(order="F")
        
        # Plot each block type in the chunk
        mask = (chunk != 5) & (chunk != -1)
        unique_blocks = np.unique(chunk[mask])
        
        for block_id in unique_blocks:
            # Skip air blocks (0)
            if block_id == 0:
                continue
                
            threshold = grid.threshold([block_id-0.5, block_id+0.5])
            
            # Get color for this block type
            if block_id in blocks_to_cols:
                color = blocks_to_cols[int(block_id)]
                opacity = 1.0 if isinstance(color, str) or len(color) == 3 else color[3]
            else:
                # Default for unknown blocks
                color = (0.7, 0.7, 0.7)  # Gray
                opacity = 1.0
            
            # Add mesh for this block type
            plotter.add_mesh(threshold, 
                           color=color,
                           opacity=opacity,
                           show_edges=True,
                           edge_color='black',
                           line_width=0.5,
                           edge_opacity=0.3,
                           lighting=True)
    
    # Add a dummy cube to set overall bounds
    total_size = grid_size * chunk_size + (grid_size - 1) * spacing
    # outline = pv.Cube(bounds=(0, total_size, 0, chunk_size, 0, total_size))
    # plotter.add_mesh(outline, opacity=0.0)
    
    # Set camera position and bounds
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1)  # Zoom out to see the whole grid
    
    # Add grid lines or axes
    # plotter.show_bounds(
    #     grid='back',
    #     location='back', 
    #     font_size=10,
    #     bounds=[0, total_size, 0,total_size, 0, chunk_size],
    #     axes_ranges=[0, total_size, 0, total_size, 0, chunk_size]
    # )
    # Save image if path provided
    if save_path:
        plotter.screenshot(save_path)
        plotter.close()
        print(f"Saved visualization for structure code {struct_code} to {save_path}")
    return plotter

# Usage example:
# plotter = visualize_structure_grid(binary_structure_dict, 4)
# plotter.show()

# Loop to visualize and save all structure codes
def visualize_all_structure_codes(structure_dict, output_dir="structure_visualizations", num_chunks=64, code_type="Struct"):
    import os
    import math
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # vis_out_dir = os.path.join(output_dir, f"{code_type}_visualizations")
    # os.makedirs(vis_out_dir, exist_ok=True)
    
    # Get all unique structure codes
    struct_codes = sorted(structure_dict.keys())
    
    print(f"Found {len(struct_codes)} {code_type} codes to visualize")
    
    # Loop through each structure code
    for struct_code in struct_codes:
        # Define output file path
        output_path = os.path.join(output_dir, f"{code_type}_code_{struct_code}.png")
        
        # Skip if file already exists (optional, can be removed)
        if os.path.exists(output_path):
            print(f"Skipping {code_type} code {struct_code} - file already exists")
            continue
        
        print(f"Visualizing {code_type} code {struct_code}...")
        
        # Generate and save visualization
        try:
            visualize_structure_grid(
                structure_dict, 
                struct_code, 
                num_chunks=num_chunks, 
                grid_size=int(math.sqrt(num_chunks)),
                save_path=output_path,
                code_type=code_type
            )
        except Exception as e:
            print(f"Error visualizing {code_type} code {struct_code}: {str(e)}")
    
    print(f"Visualization complete. Images saved to {output_dir}/")

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# Ensure pyvista is set up for notebook display
# pv.set_jupyter_backend('trame') # or 'ipygany', 'panel', etc. depending on your setup

def visualize_probability_volume_pyvista(prob_matrix, title="Probability Volume", cmap_name='viridis', fixed_opacity=0.5):
    """
    Visualizes a 4x4x4 probability matrix using PyVista, where voxel color
    maps to probability and opacity is fixed for visibility.

    Args:
        prob_matrix (np.ndarray): A 4x4x4 numpy array of probabilities (0.0 to 1.0).
        title (str): The title for the plot.
        cmap_name (str): Name of the matplotlib colormap to use (e.g., 'viridis', 'coolwarm', 'jet').
        fixed_opacity (float): The fixed opacity value (0.0 to 1.0) for visible blocks.
    """
    if not isinstance(prob_matrix, np.ndarray) or prob_matrix.shape != (4, 4, 4):
        print(f"Error: Input must be a 4x4x4 NumPy array. Got shape {prob_matrix.shape}")
        return

    plotter = pv.Plotter(notebook=True)
    plotter.add_title(title, font_size=12)

    # Set up colormap and normalization
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=0.0, vmax=1.0) # Probabilities range from 0 to 1

    min_probability_threshold = 0.01 # Don't render blocks with near-zero probability

    # Keep track if any blocks were added
    blocks_added = False
    for x in range(4):
        for y in range(4):
            for z in range(4):
                probability = prob_matrix[x, y, z]

                # Only add a cube if the probability is above the threshold
                if probability >= min_probability_threshold:
                    # Get the corresponding color from the colormap
                    # Use matplotlib cmap directly to get RGBA, then take RGB
                    color = cmap(norm(probability))

                    # Create a cube for this voxel position
                    cube = pv.Cube(bounds=(x, x + 1, y, y + 1, z, z + 1))

                    # Add the cube mesh to the plotter
                    plotter.add_mesh(
                        cube,
                        color=color[:3], # Pass RGB tuple
                        opacity=fixed_opacity,
                        show_edges=True,
                        edge_color='grey',
                        line_width=1
                    )
                    blocks_added = True

    # Only add scalar bar if we actually plotted something
    if blocks_added:
        # Add a scalar bar manually configured
        plotter.add_scalar_bar(
            title="Probability",
            # cmap=cmap_name, # Pass the colormap name
            # Define the limits for the scalar bar manually
            # This requires creating a dummy actor or setting clim directly
            # For simplicity, we'll rely on cmap name and default limits (0-1)
            # which works since our data is normalized 0-1.
            n_labels=6, # Number of labels on the colorbar (e.g., 0.0, 0.2, ..., 1.0)
            fmt="%.2f" # Format labels to 2 decimal places
        )
        # Note: If the range wasn't 0-1, you might need plotter.update_scalar_bar_range([min, max])
        # or pass clim=[min, max] if supported by your PyVista version.

    # Add axes bounds for context
    plotter.show_bounds(
        bounds=[0, 4, 0, 4, 0, 4],
        grid='front',
        location='outer',
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        ticks='inside',
        minor_ticks=False,
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        fmt='%0.0f'
    )

    # Set camera view
    plotter.camera_position = 'iso'
    # plotter.camera.zoom(1.5)

    # Show the interactive plot
    plotter.show()

block_index_to_name = {
    0:  "AIR",
    1:  "BONE_BLOCK",
    2:  "BROWN_MUSHROOM",
    3:  "BROWN_MUSHROOM_BLOCK",
    4:  "CACTUS",
    5:  "CHEST",
    6:  "CLAY",
    7:  "COAL_ORE",
    8:  "COBBLESTONE",
    9:  "DEADBUSH",
   10:  "DIRT",
   11:  "DOUBLE_PLANT",
   12:  "EMERALD_ORE",
   13:  "FLOWING_LAVA",
   14:  "FLOWING_WATER",
   15:  "GOLD_ORE",
   16:  "GRASS",
   17:  "GRAVEL",
   18:  "IRON_ORE",
   19:  "LAPIS_ORE",
   20:  "LAVA",
   21:  "LEAVES",
   22:  "LEAVES2",
   23:  "LOG",
   24:  "LOG2",
   25:  "MOB_SPAWNER",
   26:  "MONSTER_EGG",
   27:  "MOSSY_COBBLESTONE",
   28:  "PUMPKIN",
   29:  "RED_FLOWER",
   30:  "RED_MUSHROOM_BLOCK",
   31:  "REEDS",
   32:  "SAND",
   33:  "SANDSTONE",
   34:  "SNOW_LAYER",
   35:  "STONE",
   36:  "STONE_SLAB",
   37:  "TALLGRASS",
   38:  "VINE",
   39:  "WATER",
   40:  "WATERLILY",
   41:  "YELLOW_FLOWER",
}

import torch
import itertools

def calculate_average_frequency_maps(binary_structure_dict):
    """
    Calculates the average frequency map for each structure code.

    Args:
        binary_structure_dict (dict): Dictionary mapping structure codes (int)
                                      to lists of 4x4x4 binary tensors (torch.Tensor).

    Returns:
        dict: Dictionary mapping structure codes (int) to their 4x4x4
              average frequency map (float tensor). Returns empty dict if input is empty.
              Codes with no associated chunks are skipped.
    """
    if not binary_structure_dict:
        return {}

    avg_freq_maps = {}
    for code, chunk_list in binary_structure_dict.items():
        if not chunk_list:
            print(f"Warning: Code {code} has no associated chunks. Skipping.")
            continue
        # Stack tensors along a new dimension (dim=0) and calculate the mean
        stacked_chunks = torch.stack(chunk_list).float() # Ensure float for mean calculation
        avg_freq_maps[code] = torch.mean(stacked_chunks, dim=0)
    return avg_freq_maps


def calculate_average_block_frequency_maps(
    style_chunks_dict: dict,
    num_block_types: int,
    ignore_air = False
) -> dict:
    """
    Calculates the average block‐type frequency map for each style code,
    starting from raw 3D chunks of block‐ID ints.

    Args:
        style_chunks_dict (dict):
            Mapping style_code (int) → list of torch.Tensor of shape (H, W, D),
            where each entry is an integer in [0 .. num_block_types-1].
        num_block_types (int):
            Total number C of distinct block types / channels.

    Returns:
        dict:
            Mapping style_code (int) → 1D torch.float tensor of length C,
            giving the average frequency of each block type (0…1).
    """
    if not style_chunks_dict:
        return {}

    avg_block_freq_maps = {}
    for code, chunks in style_chunks_dict.items():
        if not chunks:
            print(f"Warning: Code {code} has no associated chunks. Skipping.")
            continue

        # Stack raw index tensors → shape (N, H, W, D)
        stacked_idx = torch.stack(chunks).long()
        N, H, W, D = stacked_idx.shape

        # One‐hot encode → shape (N, H, W, D, C)
        onehot = F.one_hot(stacked_idx, num_classes=num_block_types).float()

        normalizer: float
        if ignore_air:
            # Zero out the air channel (index 0) in the one-hot tensor
            # onehot structure is (N, H, W, D, C), channel 0 is air
            onehot[..., 0] = 0.0
            
            # Normalizer is the count of all non-air blocks
            num_non_air_blocks = (stacked_idx != 0).sum().item()
            normalizer = float(num_non_air_blocks)
        else:
            # Normalizer is the total count of all blocks
            total_blocks = stacked_idx.numel()
            normalizer = float(total_blocks)

        # Sum over batch + spatial dims → (C,)
        counts = onehot.sum(dim=(0, 1, 2, 3))

        # Normalize by total voxels (N * H*W*D)
        if normalizer > 0:
            avg_block_freq_maps[code] = counts / normalizer
        else:
            # If normalizer is 0 (e.g., no blocks, or only air blocks when ignore_air=True),
            # frequencies are all zero.
            avg_block_freq_maps[code] = torch.zeros_like(counts)

    return avg_block_freq_maps

def calculate_sharpness_mad(avg_freq_maps):
    """
    Calculates sharpness using Mean Absolute Deviation from 0.5 for each code's avg freq map.

    Args:
        avg_freq_maps (dict): Dictionary mapping codes to avg freq maps (4x4x4 float tensors).

    Returns:
        tuple: (dict mapping code to sharpness score, float overall average sharpness)
               Returns ({}, 0.0) if input is empty.
    """
    if not avg_freq_maps:
        return {}, 0.0

    sharpness_scores = {}
    total_sharpness = 0.0
    for code, freq_map in avg_freq_maps.items():
        # Calculate |F_i[x,y,z] - 0.5| for all voxels and average
        mad = torch.mean(torch.abs(freq_map - 0.5))
        sharpness_scores[code] = mad.item()
        total_sharpness += sharpness_scores[code]

    average_sharpness = total_sharpness / len(sharpness_scores) if sharpness_scores else 0.0
    return sharpness_scores, average_sharpness

def calculate_sharpness_entropy(avg_freq_maps, epsilon=1e-9):
    """
    Calculates sharpness using binary entropy for each voxel in the avg freq map.

    Args:
        avg_freq_maps (dict): Dictionary mapping codes to avg freq maps (4x4x4 float tensors).
        epsilon (float): Small value to avoid log(0).

    Returns:
        tuple: (dict mapping code to sharpness entropy score, float overall average sharpness entropy)
               Returns ({}, 0.0) if input is empty.
    """
    if not avg_freq_maps:
        return {}, 0.0

    entropy_scores = {}
    total_entropy = 0.0
    for code, freq_map in avg_freq_maps.items():
        # Clamp values to avoid log(0)
        p = torch.clamp(freq_map, epsilon, 1.0 - epsilon)
        # Calculate binary entropy: -p*log2(p) - (1-p)*log2(1-p)
        voxel_entropies = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)
        avg_entropy = torch.mean(voxel_entropies)
        entropy_scores[code] = avg_entropy.item()
        total_entropy += entropy_scores[code]

    average_entropy = total_entropy / len(entropy_scores) if entropy_scores else 0.0
    return entropy_scores, average_entropy

def calculate_consistency_variance(binary_structure_dict):
    """
    Calculates consistency using the average voxel variance across chunks for each code.

    Args:
        binary_structure_dict (dict): Dictionary mapping codes to lists of 4x4x4 binary tensors.

    Returns:
        tuple: (dict mapping code to consistency variance score, float overall average consistency variance)
               Returns ({}, 0.0) if input is empty. Codes with < 2 chunks have variance 0.
    """
    if not binary_structure_dict:
        return {}, 0.0

    variance_scores = {}
    total_variance = 0.0
    num_valid_codes = 0
    for code, chunk_list in binary_structure_dict.items():
        if len(chunk_list) < 2: # Variance requires at least 2 samples
             # Assign 0 variance, but maybe handle this differently? (e.g. skip?)
            variance_scores[code] = 0.0
            # print(f"Warning: Code {code} has < 2 chunks. Assigning variance 0.") # Optional warning
            continue # Skip adding to total_variance if we want average over codes with enough data

        stacked_chunks = torch.stack(chunk_list).float()
        # Calculate variance across the chunks (dim=0) for each voxel
        voxel_variances = torch.var(stacked_chunks, dim=0, unbiased=False) # Use population variance
        avg_variance = torch.mean(voxel_variances)
        variance_scores[code] = avg_variance.item()
        total_variance += variance_scores[code]
        num_valid_codes += 1 # Only count codes where variance could be computed

    average_variance = total_variance / num_valid_codes if num_valid_codes > 0 else 0.0
    return variance_scores, average_variance


def calculate_consistency_entropy(binary_structure_dict, epsilon=1e-9):
    """
    Calculates consistency using the average entropy of voxel distributions across chunks for each code.
    This relies on calculating the probability p(voxel=1) at each position first.

    Args:
        binary_structure_dict (dict): Dictionary mapping codes to lists of 4x4x4 binary tensors.
        epsilon (float): Small value to avoid log(0).

    Returns:
        tuple: (dict mapping code to consistency entropy score, float overall average consistency entropy)
               Returns ({}, 0.0) if input is empty. Codes with no chunks are skipped.
    """
    # This metric is essentially the same as sharpness entropy applied to the avg freq maps
    # because the entropy H(p) depends only on the mean frequency p at that voxel.
    # H(p) = -p*log2(p) - (1-p)*log2(1-p), where p = mean(voxel_values_for_code)
    # So we can reuse calculate_sharpness_entropy with the average frequency maps.
    avg_freq_maps = calculate_average_frequency_maps(binary_structure_dict)
    return calculate_sharpness_entropy(avg_freq_maps, epsilon)


def calculate_uniqueness_metrics(avg_freq_maps, distance_metric='mae'):
    """
    Calculates uniqueness metrics (average and minimum pairwise distance) between avg freq maps.

    Args:
        avg_freq_maps (dict): Dictionary mapping codes to avg freq maps (4x4x4 float tensors).
        distance_metric (str): 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).

    Returns:
        tuple: (float average pairwise distance, float minimum pairwise distance)
               Returns (0.0, 0.0) if less than 2 codes exist.
    """
    codes = list(avg_freq_maps.keys())
    if len(codes) < 2:
        return 0.0, 0.0 # Cannot compare pairs if less than 2 codes

    total_distance = 0.0
    min_distance = float('inf')
    num_pairs = 0

    for code1, code2 in itertools.combinations(codes, 2):
        map1 = avg_freq_maps[code1]
        map2 = avg_freq_maps[code2]

        if distance_metric == 'mae':
            distance = torch.mean(torch.abs(map1 - map2)).item()
        elif distance_metric == 'mse':
            distance = torch.mean((map1 - map2)**2).item()
        else:
            raise ValueError("Unsupported distance_metric. Choose 'mae' or 'mse'.")

        total_distance += distance
        min_distance = min(min_distance, distance)
        num_pairs += 1

    average_distance = total_distance / num_pairs if num_pairs > 0 else 0.0
    # Handle case where min_distance wasn't updated (e.g., only one code pair and it was identical)
    if min_distance == float('inf'):
        min_distance = 0.0


    return average_distance, min_distance

def find_most_similar_codes(avg_freq_maps, distance_metric='mae', tolerance=1e-5):
    """
    Finds the pair(s) of codes with the minimum pairwise distance between their avg freq maps.

    Args:
        avg_freq_maps (dict): Dictionary mapping codes to avg freq maps (4x4x4 float tensors).
        distance_metric (str): 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).
        tolerance (float): Tolerance for considering distances equal to the minimum.

    Returns:
        tuple: (float minimum_distance, list of tuples containing the most similar code pairs [(code1, code2), ...])
               Returns (inf, []) if less than 2 codes exist.
    """
    codes = list(avg_freq_maps.keys())
    if len(codes) < 2:
        return float('inf'), []

    min_distance = float('inf')
    similar_pairs = []

    # First pass to find the minimum distance
    for code1, code2 in itertools.combinations(codes, 2):
        map1 = avg_freq_maps[code1]
        map2 = avg_freq_maps[code2]

        if distance_metric == 'mae':
            distance = torch.mean(torch.abs(map1 - map2)).item()
        elif distance_metric == 'mse':
            distance = torch.mean((map1 - map2)**2).item()
        else:
            raise ValueError("Unsupported distance_metric. Choose 'mae' or 'mse'.")
        min_distance = min(min_distance, distance)

    # Handle case where min_distance wasn't updated (e.g., only one code pair and it was identical)
    if min_distance == float('inf'):
        # This case should ideally not happen if len(codes) >= 2,
        # unless maybe all maps are identical? Check one pair.
         if len(codes) >= 2:
             code1, code2 = codes[0], codes[1]
             map1 = avg_freq_maps[code1]
             map2 = avg_freq_maps[code2]
             if distance_metric == 'mae': distance = torch.mean(torch.abs(map1 - map2)).item()
             else: distance = torch.mean((map1 - map2)**2).item()
             min_distance = distance
         else: # Should not be reachable due to initial check
             return float('inf'), []


    # Second pass to collect all pairs at (or very close to) the minimum distance
    for code1, code2 in itertools.combinations(codes, 2):
        map1 = avg_freq_maps[code1]
        map2 = avg_freq_maps[code2]

        if distance_metric == 'mae':
            distance = torch.mean(torch.abs(map1 - map2)).item()
        elif distance_metric == 'mse':
            distance = torch.mean((map1 - map2)**2).item()
        else: # Should not happen if first pass succeeded
             raise ValueError("Unsupported distance_metric.")


        if abs(distance - min_distance) < tolerance:
            similar_pairs.append(tuple(sorted((code1, code2)))) # Store sorted pairs

    # Deduplicate pairs (if any floating point issues caused near duplicates)
    similar_pairs = sorted(list(set(similar_pairs)))


    return min_distance, similar_pairs

def analyze_per_code_metrics(
    code_sharpness_mad,
    code_sharpness_entropy,
    code_consistency_var,
    avg_freq_maps, # Needed to get the list of codes that actually appeared
    num_chunks_per_code, # Add a dict mapping code -> number of chunks
    sharp_mad_threshold=0.3, # Example threshold: Lower than this might be "blurry"
    sharp_entropy_threshold=0.5, # Example threshold: Higher than this might be "blurry"
    cons_var_threshold=0.15, # Example threshold: Higher than this might be "inconsistent"
    min_chunks_threshold=10 # Example threshold: Codes used less than this might be unreliable
    ):
    """
    Prints sharpness and consistency metrics for each code and flags potential issues.

    Args:
        code_sharpness_mad (dict): Code -> Sharpness (MAD) score.
        code_sharpness_entropy (dict): Code -> Sharpness (Entropy) score.
                                       (Also used for Consistency Entropy).
        code_consistency_var (dict): Code -> Consistency (Variance) score.
        avg_freq_maps (dict): Code -> Average Frequency Map. Used to get active codes.
        num_chunks_per_code(dict): Code -> integer count of chunks assigned.
        sharp_mad_threshold (float): Threshold below which MAD is flagged low.
        sharp_entropy_threshold (float): Threshold above which Entropy is flagged high.
        cons_var_threshold (float): Threshold above which Variance is flagged high.
        min_chunks_threshold (int): Threshold below which code usage is flagged low.
    """
    print("\n--- Per-Code Analysis ---")
    codes = sorted(list(avg_freq_maps.keys()))

    if not codes:
        print("No codes found in avg_freq_maps.")
        return

    print(f"{'Code':<6} {'NumChunks':<10} {'Sharp(MAD)':<12} {'Sharp(Entr)':<12} {'Cons(Var)':<12} {'Flags':<20}")
    print("-" * 70)

    for code in codes:
        sharp_mad = code_sharpness_mad.get(code, float('nan'))
        sharp_entropy = code_sharpness_entropy.get(code, float('nan'))
        cons_var = code_consistency_var.get(code, float('nan')) # Variance might be missing if < 2 chunks
        n_chunks = num_chunks_per_code.get(code, 0)


        flags = []
        if n_chunks < min_chunks_threshold:
             flags.append(f"LOW_USAGE({n_chunks})")
        if sharp_mad < sharp_mad_threshold:
            flags.append("LOW_SHARPNESS_MAD")
        if sharp_entropy > sharp_entropy_threshold:
            flags.append("HIGH_SHARPNESS_ENT")
        if cons_var > cons_var_threshold:
            flags.append("HIGH_CONSIST_VAR")
        # Note: Consistency Entropy is same as Sharpness Entropy here

        # Handle cases where variance wasn't computed (e.g., < 2 chunks)
        cons_var_str = f"{cons_var:<12.4f}" if not torch.isnan(torch.tensor(cons_var)) else f"{'N/A':<12}" # Use torch.isnan for tensor check compatibility

        print(f"{code:<6} {n_chunks:<10} {sharp_mad:<12.4f} {sharp_entropy:<12.4f} {cons_var_str} {', '.join(flags)}")

    print("-" * 70)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import os
import torch # Assuming input is torch tensor

def plot_frequency_heatmap(prob_matrix_tensor, struct_code, save_dir):
    """
    Generates and saves a 3D frequency heatmap visualization for a structure code,
    applying Minecraft-specific orientation adjustments. Uses the 'viridis' colormap
    where color represents probability and ensures Y (Height) axis is vertical and correctly oriented.

    Args:
        prob_matrix_tensor (torch.Tensor): The 4x4x4 frequency map (PyTorch tensor).
        struct_code (int): The structure code ID.
        save_dir (str): The directory to save the heatmap image.
    """
    # --- Input Validation and Conversion ---
    if not isinstance(prob_matrix_tensor, torch.Tensor) or prob_matrix_tensor.shape != (4, 4, 4):
        print(f"Error: Input for code {struct_code} is not a 4x4x4 torch tensor.")
        return

    prob_matrix_np = prob_matrix_tensor.cpu().numpy()

    # --- Orientation Adjustment ---
    # Apply the same sequence: Transpose Z <-> X, then rotate around Z.
    prob_matrix_transposed = prob_matrix_np.transpose(2, 0, 1) # (X,Y,Z) -> (Z,X,Y)
    prob_matrix_oriented = np.rot90(prob_matrix_transposed, k=1, axes=(1, 2)) # (Z,X,Y) -> (Z,Y_mc,X_mc)

    # --- Plotting Setup ---
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(9, 8)) # Adjusted size slightly for colorbar
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid indices corresponding to the oriented matrix axes (Z_mc, Y_mc, X_mc)
    z_indices, y_indices, x_indices = np.indices(prob_matrix_oriented.shape)

    # Flatten coordinates and probability values
    x_coords_mc = x_indices.flatten() # Minecraft X coordinates
    y_coords_mc = y_indices.flatten() # Minecraft Y (Height) coordinates
    z_coords_mc = z_indices.flatten() # Minecraft Z coordinates
    probabilities = prob_matrix_oriented.flatten()

    # --- Create Scatter Plot (Viridis Colormap) ---
    # Plot mapping:
    # Plot X-axis <- Minecraft X data (x_coords_mc)
    # Plot Y-axis <- Minecraft Z data (z_coords_mc)
    # Plot Z-axis <- Minecraft Y (Height) data (y_coords_mc) <<< VERTICAL AXIS
    scatter = ax.scatter(x_coords_mc, z_coords_mc, y_coords_mc,
                         c=probabilities, # Color based on probability values
                         cmap='viridis',  # Use the viridis colormap
                         vmax=1,
                         vmin=0,
                         s=300,
                         edgecolors='k', linewidth=0.5)

    # --- Add Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Probability')

    # --- Set Labels, Ticks, Limits, Title, and INVERT Z-AXIS ---
    ax.set_title(f"Structure Code {struct_code} Frequency Map (Color = Probability)") # Updated title
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (Height)")

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4)) # Corresponds to Z data range
    ax.set_zticks(np.arange(4)) # Corresponds to Y (Height) data range

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(-0.5, 3.5)
    # --- Invert the Z-axis (which represents Y-Height) ---
    ax.invert_zaxis()
    # --- Optionally invert X too depending on preferred view ---
    # ax.invert_xaxis()

    # Adjust view angle if desired
    ax.view_init(elev=25., azim=-125)

    # --- Save and Close ---
    image_path = os.path.join(save_dir, f"heatmap_code_{struct_code}.png")
    try:
        plt.savefig(image_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving heatmap for code {struct_code}: {e}")
    finally:
        plt.close(fig)

    # === Plot 2: Thresholded Structure (at 0.5) ===
    binary_structure_oriented = prob_matrix_oriented > 0.5 # Shape (MC_Z, MC_Y_Height, MC_X)

    # For ax.voxels, data is typically (X, Y, Z) corresponding to plot axes.
    # We want: Plot X = MC_X, Plot Y = MC_Z, Plot Z = MC_Y_Height.
    # So, data_for_voxels should be (MC_X_dim, MC_Z_dim, MC_Y_Height_dim)
    # `binary_structure_oriented` is (MC_Z_dim, MC_Y_Height_dim, MC_X_dim)
    # Transpose from (idx_Z, idx_Y_H, idx_X) to (idx_X, idx_Z, idx_Y_H)
    data_for_voxels = binary_structure_oriented.transpose(2, 0, 1)

    fig_thresh = plt.figure(figsize=(8, 7)) # Slightly smaller, no colorbar needed
    ax_thresh = fig_thresh.add_subplot(111, projection='3d')

    # `facecolors` can be used to set color, `edgecolor` for borders
    # `voxels` plots True values.
    ax_thresh.voxels(data_for_voxels, edgecolor='k', facecolors='cyan')

    ax_thresh.set_title(f"Structure Code {struct_code} Thresholded (Prob > 0.5)")
    ax_thresh.set_xlabel("X (MC)")
    ax_thresh.set_ylabel("Z (MC)")
    ax_thresh.set_zlabel("Y (MC Height)")

    ax_thresh.set_xticks(np.arange(4))
    ax_thresh.set_yticks(np.arange(4)) # Corresponds to Z data range
    ax_thresh.set_zticks(np.arange(4))

    ax_thresh.set_xlim(-0.5, 3.5) # Match limits for consistency if using voxels at 0..3 ranges
    ax_thresh.set_ylim(-0.5, 3.5)
    ax_thresh.set_zlim(-0.5, 3.5)
    ax_thresh.invert_zaxis()
    # ax_thresh.invert_xaxis() # Optional

    ax_thresh.view_init(elev=25., azim=-125) # Same view angle

    threshold_image_path = os.path.join(save_dir, f"heatmap_code_{struct_code}_thresholded.png")
    try:
        plt.savefig(threshold_image_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving thresholded plot for code {struct_code}: {e}")
    finally:
        plt.close(fig_thresh)

def plot_block_frequency_heatmap(
    style_code: int,
    block_freqs,                    # torch.Tensor or array of shape (C,)
    save_dir: str,
    block_converter,                # object with index_to_block(idx) -> blockID
    block_index_to_name: dict,      # maps Minecraft blockID → block name
    cmap: str = "viridis"
):
    """
    Plot and save a 1C heatmap of blocktype frequencies for one style code.

    Args:
        style_code (int): Identifier of the style code.
        block_freqs (Tensor or ndarray): 1D of length C, values in [0,1].
        save_dir (str): Directory where the PNG will be written.
        block_converter: Object with method index_to_block(idx) returning
                         the Minecraft blockID for channel idx.
        block_index_to_name (dict): Mapping from blockID to block name.
        cmap (str): Matplotlib colormap name.
    """
    # ensure numpy array on CPU
    if isinstance(block_freqs, torch.Tensor):
        # print('converting to numpy')
        freqs = block_freqs.detach().cpu().numpy()
    else:
        print('already numpy')
        freqs = np.array(block_freqs)

    # print(freqs)
    # print(freqs.shape)
    C = freqs.shape[0]
    heatmap = freqs.reshape(C, 1)   # shape (1, C)

    # Map each channel idx → Minecraft blockID → block name
    block_ids = [i for i in range(C)]
    names = [block_index_to_name[bid] for bid in block_ids]

    # ensure output dir exists
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"block_heatmap_code_{style_code}.png")

    # plot
    plt.figure(figsize=(4, max(8, C * 0.2)))
    im = plt.imshow(
        heatmap,
        aspect='auto',
        cmap=cmap,
        vmin=0.0, vmax=1.0         # fixed scale across all codes
    )
     # ---- Bigger, labeled colorbar ----
    cbar = plt.colorbar(
        im,
        orientation='vertical',
        fraction=0.12,    # thicker bar
        pad=0.05          # more space from the heatmap
    )
    cbar.ax.tick_params(labelsize=8)               # larger tick labels
    cbar.set_label('Frequency', rotation=270, labelpad=15)

    plt.yticks(np.arange(C), names, fontsize=6)
    # only one column, no X-ticks needed
    plt.xticks([])

    plt.title(f"Style Code {style_code} Block Frequencies", pad=10)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()

    # ---- Calculate and Plot Top 5 Block Frequencies Histogram ----
    num_top_blocks_to_plot = min(5, C)

    if num_top_blocks_to_plot > 0:
        sorted_channel_indices = np.argsort(freqs)[::-1] 

        top_n_indices = sorted_channel_indices[:num_top_blocks_to_plot]
        top_n_freqs = freqs[top_n_indices]
        top_n_block_names = [block_index_to_name.get(idx, f"Unknown_idx_{idx}") for idx in top_n_indices]

        print(f"Style Code {style_code} Top 5 Blocks:\n")
        for block_name, freq in zip(top_n_block_names, top_n_freqs):
            print(f'{block_name}: {freq}')

        plt.figure(figsize=(10, 6))
        # Updated to use plt.colormaps.get_cmap()
        bar_color = plt.colormaps.get_cmap(cmap)(0.6) if cmap else 'skyblue' # Default color if cmap is None
        bars = plt.bar(np.arange(num_top_blocks_to_plot), top_n_freqs, color=bar_color)

        plt.ylabel("Frequency")
        plt.xlabel("Block Type")
        plt.title(f"Top {num_top_blocks_to_plot} Block Frequencies for Style Code {style_code}", pad=10)
        plt.xticks(np.arange(num_top_blocks_to_plot), top_n_block_names, rotation=45, ha="right")
        plt.ylim(0, 1.05) 
        
        for bar_idx, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        hist_basename = f"block_heatmap_code_{style_code}.png" 
        top5_hist_filename = f"top5_{hist_basename}"
        histogram_image_path = os.path.join(save_dir, top5_hist_filename)
        print(f'saving top 5 hist to: {histogram_image_path}')
        plt.savefig(histogram_image_path, dpi=300)
        plt.close()
    elif C > 0 : 
        print(f"Warning: num_top_blocks_to_plot is 0 for style code {style_code} but C={C}. Skipping histogram.")

    

    return image_path

def calculate_positional_frequencies(fqgan):
    latent_depth, latent_height, latent_width = 6, 6, 6 # *** Adjust if different ***
    latent_shape = (latent_depth, latent_height, latent_width)
    num_structure_codes = fqgan.struct_codebook_size
    num_style_codes = fqgan.style_codebook_size
    # Initialize count tensors for each code
    struct_position_counts = {
        code: torch.zeros(latent_shape, dtype=torch.long, device='cpu')
        for code in range(num_structure_codes)
    }
    style_position_counts = {
        code: torch.zeros(latent_shape, dtype=torch.long, device='cpu')
        for code in range(num_style_codes)
    }
    total_samples_processed = 0

    print("Collecting positional frequencies from train_loader...")
    for batch_idx, batch in enumerate(train_loader):
        # Limit batches for testing?
        # if batch_idx > 20: break

        if torch.cuda.is_available():
            batch = batch.cuda()

        # Only need the encoding part
        style_indices, struct_indices = encode_and_quantize(fqgan, batch) # Get indices for the whole batch

        # Process each sample in the batch
        for sample_idx in range(struct_indices.shape[0]): # Iterate through batch dimension
            struct_indices_sample = struct_indices[sample_idx].cpu() # Get indices for one sample, move to CPU

            # Iterate through the latent grid dimensions (D, H, W)
            for i in range(latent_depth):
                for j in range(latent_height):
                    for k in range(latent_width):
                        struct_code = struct_indices_sample[i, j, k].item()
                        if 0 <= struct_code < num_structure_codes:
                            position_counts[struct_code][i, j, k] += 1
                        else:
                            print(f"Warning: Encountered out-of-bounds code {struct_code} at ({i},{j},{k})")
            total_samples_processed += 1

        # Optional: Print progress
        if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    return position_counts

def plot_positional_frequency(position_counts, struct_code, save_dir):
    """
    Generates and saves a 3D scatter plot showing the positional frequency
    of a structure code within the latent grid, using a Viridis colormap and
    Minecraft coordinate conventions (Y=Height is vertical, Y=0 is bottom).

    Args:
        position_counts (dict): Maps struct_code to 6x6x6 count tensor.
        struct_code (int): The structure code to visualize.
        save_dir (str): The directory to save the plot image.
    """
    if struct_code not in position_counts:
        print(f"Error: Code {struct_code} not found in position_counts dictionary.")
        return

    counts_tensor = position_counts[struct_code].cpu() # Ensure it's on CPU
    latent_shape = counts_tensor.shape
    if len(latent_shape) != 3:
        print(f"Error: Count tensor for code {struct_code} is not 3D (shape: {latent_shape}).")
        return

    counts_numpy = counts_tensor.numpy()
    max_count = np.max(counts_numpy)

    if max_count == 0:
        print(f"Info: Code {struct_code} never appeared. Skipping visualization.")
        # Optional: Create an empty plot placeholder if desired
        return # Stop here if code never appeared

    # --- Plotting Setup ---
    # os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid indices (I, J, K) corresponding to tensor dimensions
    # Assume I=Depth(Z_mc), J=Height(Y_mc), K=Width(X_mc)
    i_indices, j_indices, k_indices = np.indices(latent_shape)

    # Flatten coordinates and the raw counts (for color mapping)
    i_coords_mc = i_indices.flatten() # Minecraft Z coordinates
    j_coords_mc = j_indices.flatten() # Minecraft Y (Height) coordinates
    k_coords_mc = k_indices.flatten() # Minecraft X coordinates
    frequencies = counts_numpy.flatten()

    # --- Create Scatter Plot (Viridis Colormap, Correct Axis Mapping) ---
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('Greys')
    # Plot mapping:
    # Plot X-axis <- Minecraft X data (k_coords_mc)
    # Plot Y-axis <- Minecraft Z data (i_coords_mc)
    # Plot Z-axis <- Minecraft Y (Height) data (j_coords_mc) <<< VERTICAL AXIS
    scatter = ax.scatter(k_coords_mc, i_coords_mc, j_coords_mc, # Correct mapping
                         c=frequencies, cmap=cmap, # Color based on frequency counts
                         s=150, # Adjust size as needed
                         alpha=0.8, # Add some transparency
                         # vmin=0, vmax=max_count, # Optional: Explicitly set color limits
                         edgecolors='grey', linewidth=0.5)

    # --- Add Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Code Occurrence Count")

    # --- Set Labels, Ticks, Limits, Title, and Invert Z-axis ---
    ax.set_title(f"Code {struct_code} Positional Frequency")
    # Label plot axes according to the *Minecraft dimension* plotted on them
    ax.set_xlabel("Z (Latent Dim K)")
    ax.set_ylabel("X (Latent Dim I)")
    ax.set_zlabel("Y (Height, Latent Dim J)") # Vertical axis

    # Set ticks based on the dimension size
    ax.set_xticks(np.arange(latent_shape[2])) # K dimension
    ax.set_yticks(np.arange(latent_shape[0])) # I dimension
    ax.set_zticks(np.arange(latent_shape[1])) # J dimension

    # Set limits for plot axes
    ax.set_xlim(-0.5, latent_shape[2] - 0.5)
    ax.set_ylim(-0.5, latent_shape[0] - 0.5)
    ax.set_zlim(-0.5, latent_shape[1] - 0.5)

    # --- Invert the Z-axis (which represents Y-Height) ---
    # ax.invert_zaxis() # Ensures Y=0 is at the bottom
    ax.invert_yaxis() # Ensures Y=0 is at the bottom

    # Adjust view angle
    ax.view_init(elev=20., azim=-75)
    # plt.show()
    # --- Save and Close ---
    image_path = os.path.join(save_dir, f"pos_freq_code_{struct_code}.png")
    try:
        plt.savefig(image_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving positional frequency plot for code {struct_code}: {e}")
    finally:
        plt.close(fig)


def generate_all_positional_frequency_plots(position_counts, output_dir):
    print(f"Generating positional frequency plots for {len(position_counts)} codes...")
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    # Iterate through the codes present in the dictionary
    codes_to_plot = sorted(position_counts.keys())

    for i, code in enumerate(codes_to_plot):
        # Call the plotting function for the current code
        plot_positional_frequency(position_counts, code, output_dir)

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(codes_to_plot):
            print(f"Generated plot {i+1}/{len(codes_to_plot)} (Code {code})")

    print(f"Finished generating positional frequency plots. Saved to: {output_dir}")

def load_model_for_step(step, base_path):
    """Loads the FQGAN model for a specific checkpoint step."""
    print(f"\n--- Loading Checkpoint Step: {step} ---")
    try:
        fqgan_hparams = dict_to_vcqgan_hparams(load_hparams_from_json(f"{base_path}"), 'minecraft')
        fqgan_hparams.load_step = step # Set the step to load
        fqgan = FQModel(fqgan_hparams)
        fqgan = load_fqgan_from_checkpoint(fqgan_hparams, fqgan) 
        print(f'Loaded model for step {step}')
        fqgan.eval() # Set to evaluation mode
        if torch.cuda.is_available():
             fqgan.cuda()
        return fqgan
    except Exception as e:
        print(f"Error loading model for step {step}: {e}")
        # Depending on the error, you might want to skip this step or stop
        return None # Indicate failure
    
@torch.no_grad() # Ensure no gradients are computed
def collect_data_for_model(model, loader, block_converter):
    """
    Collects structure dictionary data (original and binary) for a given model.
    Returns original structure dict, binary structure dict, and chunk counts per code.
    """
    print("Collecting data from train_loader...")
    latent_depth, latent_height, latent_width = 6, 6, 6 # *** Adjust if different ***
    latent_shape = (latent_depth, latent_height, latent_width)
    num_structure_codes = fqgan.struct_codebook_size
    num_style_codes = fqgan.style_codebook_size

     # Initialize count tensors for each code
    struct_position_counts = {
        code: torch.zeros(latent_shape, dtype=torch.long, device='cpu')
        for code in range(num_structure_codes)
    }
    style_position_counts = {
        code: torch.zeros(latent_shape, dtype=torch.long, device='cpu')
        for code in range(num_style_codes)
    }
    total_samples_processed = 0

    air_idx = block_converter.get_air_block_index()
    structure_dict_original = {} # Store original chunks here
    style_dict_original = {}
    # style_dict = {} # Keep this if you plan to analyze style later
    i = 0
    # --- (Loop through loader, encode, decode as before) ---
    for batch_idx, batch in enumerate(loader):
        # Limit number of batches?
        # if batch_idx > 20: break

        if torch.cuda.is_available():
            batch = batch.cuda()

        for sample_idx in range(len(batch)):
            sample = batch[sample_idx].unsqueeze(0)
            try:
                style_indices, struct_indices = encode_and_quantize(model, sample)
                reconstructed, _ = decode_from_indices(style_indices, struct_indices, model, two_stage=True)

                struct_indices_cpu = struct_indices.squeeze(0).cpu()
                style_indices_cpu = style_indices.squeeze(0).cpu() # If needed for style
                reconstructed_cpu = reconstructed.cpu()

                for i in range(struct_indices_cpu.shape[0]): # D
                    for j in range(struct_indices_cpu.shape[1]): # H
                        for k in range(struct_indices_cpu.shape[2]): # W
                            # style_code = style_indices_cpu[i, j, k].item() # If needed
                            struct_code = struct_indices_cpu[i, j, k].item()
                            style_code  = style_indices_cpu[i, j, k].item()
                            x_start, y_start, z_start = i * 4, j * 4, k * 4
                            x_end, y_end, z_end = x_start + 4, y_start + 4, z_start + 4
                            block_chunk = reconstructed_cpu[x_start:x_end, y_start:y_end, z_start:z_end]

                            # Store the original chunk tensor
                            if struct_code not in structure_dict_original:
                                structure_dict_original[struct_code] = []
                            if style_code not in style_dict_original:
                                style_dict_original[style_code] = []

                            structure_dict_original[struct_code].append(block_chunk)
                            style_dict_original[style_code].append(block_chunk)
                            struct_position_counts[struct_code][i, j, k] += 1
                            style_position_counts[style_code][i, j, k] += 1
                            # --- (Style dictionary population if needed) ---

            except Exception as e:
                 print(f"Error processing sample {sample_idx} in batch {batch_idx}: {e}")
                 continue

    print(f"Finished data collection. Found {len(structure_dict_original)} unique structure codes and {len(style_dict_original)} unique style codes.")

    # Convert structure chunks to binary
    print("Converting structure chunks to binary...")
    binary_structure_dict = {}
    num_chunks_per_code = {}
    for struct_code, block_list in structure_dict_original.items(): # Iterate original dict
         num_chunks_per_code[struct_code] = len(block_list) # Get count first
         if block_list:
             try:
                 # Convert to binary for metrics
                 binary_chunks = [(b.cpu() != air_idx).to(dtype=torch.int) for b in block_list]
                 binary_structure_dict[struct_code] = binary_chunks
             except Exception as e:
                 print(f"Error converting chunks to binary for code {struct_code}: {e}")
                 # Ensure binary dict entry exists even if conversion failed? Or skip?
                 # binary_structure_dict[struct_code] = [] # Or handle differently
         # else: # No need for else, count is already 0 from len() above

    # Clean up GPU memory if possible
    # del style_dict # Keep structure_dict_original
    if torch.cuda.is_available():
       torch.cuda.empty_cache()

    print(f"Finished binary conversion. Processed {len(binary_structure_dict)} codes.")
    # Return the original dictionary as well
    return structure_dict_original, style_dict_original, binary_structure_dict, num_chunks_per_code, struct_position_counts, style_position_counts

def save_checkpoint_report(step, metrics, num_chunks_per_code, similar_pairs, min_dist, output_dir):
    """Saves a text report summarizing metrics for a single checkpoint."""
    report_path = os.path.join(output_dir, f"checkpoint_{step}_report.txt")
    print(f"Saving report for step {step} to {report_path}")

    with open(report_path, 'w') as f:
        f.write(f"--- Analysis Report for Checkpoint Step: {step} ---\n\n")

        f.write("--- Model-Level Metrics ---\n")
        f.write(f"Avg Sharpness (MAD):      {metrics['avg_sharpness_mad']:.4f} (Higher is better)\n")
        f.write(f"Avg Sharpness (Entropy):  {metrics['avg_sharpness_entropy']:.4f} (Lower is better)\n")
        f.write(f"Avg Consistency (Var):    {metrics['avg_consistency_var']:.4f} (Lower is better)\n")
        # Consistency Entropy is same as Sharpness Entropy in our current impl.
        # f.write(f"Avg Consistency (Entropy):{metrics['avg_consistency_entropy']:.4f} (Lower is better)\n")
        f.write(f"Avg Pairwise Dist ({DISTANCE_METRIC.upper()}): {metrics['avg_pairwise_dist']:.4f} (Higher is better)\n")
        f.write(f"Min Pairwise Dist ({DISTANCE_METRIC.upper()}): {min_dist:.6f} (Higher is better)\n\n")


        f.write("--- Code Uniqueness Analysis ---\n")
        f.write(f"Minimum Pairwise Distance ({DISTANCE_METRIC.upper()}): {min_dist:.6f}\n")
        if similar_pairs:
            f.write("Most Similar Code Pairs (Potential Redundancy):\n")
            for pair in similar_pairs:
                f.write(f"  - Codes {pair[0]} and {pair[1]}\n")
        else:
            f.write("No highly similar code pairs found.\n")
        f.write("\n")


        f.write("--- Per-Code Analysis ---\n")
        codes = sorted(list(metrics['code_sharpness_mad'].keys())) # Use keys from one of the dicts
        if not codes:
            f.write("No codes found with sufficient data for per-code analysis.\n")
        else:
            header = f"{'Code':<6} {'NumChunks':<10} {'Sharp(MAD)':<12} {'Sharp(Entr)':<12} {'Cons(Var)':<12} {'Flags':<20}\n"
            f.write(header)
            f.write("-" * (len(header) - 1) + "\n") # Adjust separator length

            for code in codes:
                sharp_mad = metrics['code_sharpness_mad'].get(code, float('nan'))
                sharp_entropy = metrics['code_sharpness_entropy'].get(code, float('nan'))
                cons_var = metrics['code_consistency_var'].get(code, float('nan'))
                n_chunks = num_chunks_per_code.get(code, 0)

                flags = []
                if n_chunks < MIN_CHUNKS_THRESHOLD:
                     flags.append(f"LOW_USAGE({n_chunks})")
                # Use np.isnan for checking because values are Python floats now
                if not np.isnan(sharp_mad) and sharp_mad < SHARP_MAD_THRESHOLD:
                    flags.append("LOW_SHARPNESS_MAD")
                if not np.isnan(sharp_entropy) and sharp_entropy > SHARP_ENTROPY_THRESHOLD:
                    flags.append("HIGH_SHARPNESS_ENT")
                if not np.isnan(cons_var) and cons_var > CONS_VAR_THRESHOLD:
                    flags.append("HIGH_CONSIST_VAR")

                cons_var_str = f"{cons_var:<12.4f}" if not np.isnan(cons_var) else f"{'N/A':<12}"

                f.write(f"{code:<6} {n_chunks:<10} {sharp_mad:<12.4f} {sharp_entropy:<12.4f} {cons_var_str} {', '.join(flags)}\n")
            f.write("-" * (len(header) -1 ) + "\n")


def main():
    data_path = '24_newdataset_processed_cleaned3.pt'
    train_loader, val_loader = get_minecraft_dataloaders(
        data_path,
        batch_size=4,
        num_workers=0, 
        val_split=0.1
    )
    mappings_path = '24_newdataset_mappings3.pt'
    block_converter = BlockBiomeConverter.load_mappings(mappings_path)
    visualizer = MinecraftVisualizerPyVista()
    all_metrics_data = {} 
    model_name_file = os.path.join(OUTPUT_DIR, "model_name.txt")
    with open(model_name_file, "w", encoding="utf-8") as f:
        f.write(MODEL_BASE_PATH)
    for step in CHECKPOINT_STEPS:
        # 1. Load Model
        fqgan = load_model_for_step(step, MODEL_BASE_PATH)
        fqgan.to('cuda')
        if fqgan is None:
            print(f"Skipping step {step} due to loading error.")
            continue

            # Define visualization directory for this step
        vis_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{step}_visualizations")
        os.makedirs(vis_dir, exist_ok=True) # Ensure vis_dir exists for caching
        # --- Cache file path for collected data ---
        collected_data_cache_path = os.path.join(vis_dir, "collected_model_data.pt")

        try:
            cached_data = torch.load(collected_data_cache_path)
            structure_dict_original = cached_data['structure_dict_original']
            style_dict_original = cached_data['style_dict_original']
            binary_structure_dict = cached_data['binary_structure_dict']
            num_chunks_per_code = cached_data['num_chunks_per_code']
            struct_position_counts = cached_data['struct_position_counts']
            style_position_counts = cached_data['style_position_counts']
            print(f"Successfully loaded cached data for step {step}.")
        except Exception as e:
            print(f"Error loading cached data for step {step}: {e}. Re-collecting data.")
            structure_dict_original = None # Ensure it's None to trigger collection

        if structure_dict_original is None: 
            print(f"Collecting data for model at step {step}...")
            sdo, stylo, bsd, ncpc, spc, stpc = collect_data_for_model(fqgan, train_loader, block_converter)
            
            structure_dict_original = sdo
            style_dict_original = stylo
            binary_structure_dict = bsd
            num_chunks_per_code = ncpc
            struct_position_counts = spc
            style_position_counts = stpc
            
            # Save the collected data to cache
            print(f"Saving collected data to cache for step {step} at {collected_data_cache_path}")
            try:
                data_to_cache = {
                    'structure_dict_original': structure_dict_original,
                    'style_dict_original': style_dict_original,
                    'binary_structure_dict': binary_structure_dict,
                    'num_chunks_per_code': num_chunks_per_code,
                    'struct_position_counts': struct_position_counts,
                    'style_position_counts': style_position_counts,
                }
                torch.save(data_to_cache, collected_data_cache_path)
                print(f"Successfully saved data to cache for step {step}.")
            except Exception as e:
                print(f"Error saving data to cache for step {step}: {e}")

        if not binary_structure_dict: # Check binary dict as it's needed for metrics
            print(f"Skipping metrics/plotting for step {step} as no data was collected/converted.")
            del fqgan, structure_dict_original # Clean up original dict too
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        # 3. Calculate Metrics
        print(f"Calculating metrics for step {step}...")
        metrics = {}
        avg_freq_maps = calculate_average_frequency_maps(binary_structure_dict)
        block_freq_maps = calculate_average_block_frequency_maps(style_dict_original, fqgan.in_channels, ignore_air=True)

        if not avg_freq_maps:
            print(f"Skipping metrics for step {step} as no average frequency maps could be calculated (no valid codes?).")
            del fqgan, binary_structure_dict, num_chunks_per_code
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue
        
        
        # --- Visualization Section ---
        print(f"Generating visualizations for step {step}...")
        struct_vis_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{step}_visualizations/Struct_visualizations") # Combined dir
        style_vis_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{step}_visualizations/Style_visualizations")
        os.makedirs(struct_vis_dir, exist_ok=True)
        os.makedirs(style_vis_dir, exist_ok=True)
        visualize_top_binary_patterns(binary_structure_dict, struct_vis_dir, num_top_patterns_to_show_3d=5)

        visualize_all_structure_codes(structure_dict_original, struct_vis_dir)
        visualize_all_structure_codes(style_dict_original, style_vis_dir, code_type="Style")
        generate_all_positional_frequency_plots(struct_position_counts, struct_vis_dir)
        generate_all_positional_frequency_plots(style_position_counts, style_vis_dir)
        for code, freq_map_tensor in avg_freq_maps.items():
            # --- Heatmap Generation ---
            plot_frequency_heatmap(freq_map_tensor, code, struct_vis_dir) # Save to combined dir
                
        for code, block_freq_map_tensor in block_freq_maps.items():
            # --- Heatmap Generation ---
            plot_block_frequency_heatmap(code, block_freq_map_tensor, style_vis_dir, block_converter, block_index_to_name) # Save to combined dir
    

        #Clean up GPU memory before next loop iteration
        del fqgan, binary_structure_dict, num_chunks_per_code, avg_freq_maps, metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n--- Finished evaluating all checkpoints ---")

if __name__ == "__main__":
    main()