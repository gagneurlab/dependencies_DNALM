import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import argparse
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import time
from plotnine import *
import pyranges as pr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



#this function loads stored maps. please refer to the compute_and_visualize_dep_maps.ipynb to compute a map for a specific DNA LM from scratch
def create_load_matrix_function(species_of_interest, maps_base_path='samples_upstream_kazachstania_k1_interaction/'):
    # load the dataframes
    print('Loading dataframes with snp preds...')
    snp_df = pd.read_parquet(os.path.join(maps_base_path, f"{species_of_interest}/snp.parquet")).reset_index()
    id_to_path = pd.read_parquet(os.path.join(maps_base_path, f"{species_of_interest}/id_to_path.parquet"))
    snp_df = id_to_path.merge(snp_df, on="transcript_id")

    snp_df = snp_df.reset_index(drop=True) # make sure the index is unique
    assert (snp_df.index == snp_df["index"]).all() # check that the index is unique


    all_genes = snp_df['gene_id'].unique() # In pandas Uniques are returned in order of appearance. That is important later on to optimize reading files


    gene_to_idxs_snps_dataframe = snp_df.groupby('gene_id').groups # create a dictionary mapping gene_id to indices in the snp_df dataframe for very fast lookup increases speed by at least 10x
    transcripts_to_idxs_snps_dataframe = snp_df.groupby('transcript_id').groups
    file_cache = OrderedDict()
    CACHE_SIZE = 10 

    def get_ci_matrix(gene_id=None, transcript_id=None, effect_on_ref_only=False, use_logit=True, eps=1e-10):
    
        
        if transcript_id is None: 
            if gene_id not in gene_to_idxs_snps_dataframe:
                return None
            snp_subset_df = snp_df.loc[gene_to_idxs_snps_dataframe[gene_id]] 
        elif transcript_id not in transcripts_to_idxs_snps_dataframe:
                return None
        else:
            snp_subset_df = snp_df.loc[transcripts_to_idxs_snps_dataframe[transcript_id]] 

        if len(snp_subset_df) == 0:
            return None
        # get path and indices
        path = snp_subset_df.iloc[0]["path"]
        offset = int(path.split("_")[-2])
        start = snp_subset_df["index"].min() - offset
        end = snp_subset_df["index"].max() - offset

        # load matrix
        if path in file_cache:
            # Move to the end to show that it was recently accessed
            file_cache.move_to_end(path)
            interact_mat = file_cache[path]
        else:
            interact_mat = torch.load(path)
            file_cache[path] = interact_mat
            # Remove the oldest item if cache exceeds its size limit
            if len(file_cache) > CACHE_SIZE:
                file_cache.popitem(last=False)

        interact_mat = interact_mat[start:end+1]
        interact_mat = interact_mat.float()

        if use_logit: # for the logit add the small value and renormalize such that every prob in one position sums to 1
            interact_mat = interact_mat + eps 
            interact_mat = interact_mat/interact_mat.sum(axis=-1,keepdim=True)

        # prepare snp effect matrix  
        interact_mat_ref = interact_mat[0]
        interact_mat_alt = interact_mat[1:]

        #return interact_mat_alt, interact_mat_ref
        if not use_logit:
            snp_effect_unordered = torch.log(interact_mat_alt + eps) - torch.log(interact_mat_ref + eps)#.unsqueeze(0)
        else:
            snp_effect_unordered = (torch.log(interact_mat_alt) - torch.log(1 - interact_mat_alt) 
                                    - torch.log(interact_mat_ref) + torch.log(1 - interact_mat_ref))


        snp_effect = torch.zeros((4,1003,1003,4))

        snp_effect[snp_subset_df.iloc[1:]['var_nt_idx'].values, snp_subset_df.iloc[1:]['var_pos'].values] = snp_effect_unordered

        ci_matrix = torch.abs(snp_effect).max(dim=0)[0].max(dim=2)[0]

        return ci_matrix
    
    return get_ci_matrix


def plot_ci_matrix(ci_matrix, vmax=8, figsize=10):#increase figure size
    plt.rcParams['figure.figsize'] = [figsize, figsize]
    plt.imshow(ci_matrix, cmap="coolwarm", vmax=vmax)
    return plt

def plot_matrix_with_roi(ci_matrix, dna_sequence, coords, zoom_slack=20, plot_size=30, width_ratios=[1, 2], 
    vmax=5, display_values=False, annot_size=10, display_dotted_line=True, tick_label_fontsize=8):


    # Create a figure with two square subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_size, plot_size), gridspec_kw={'width_ratios': width_ratios})

    # Adjust the space between the plots
    plt.subplots_adjust(wspace=.1)  

    # Plot the first heatmap
    sns.heatmap(ci_matrix, cmap='coolwarm', vmax=vmax, ax=ax1, cbar=False,  
                xticklabels=False, yticklabels=False)  # Remove the color bar
    ax1.set_aspect('equal')

    # The donor coordinates with respect to the start codon can be higher or lower than the acceptor coordinates depeding on the strand of the start codon 100bp area considered
    x1 = coords[1]
    x0 = coords[0]

    ax1.axhline(y=x0, color='gold', linestyle='--')
    ax1.axhline(y=x1, color='gold', linestyle='--')
    ax1.axvline(x=x0, color='gold', linestyle='--')
    ax1.axvline(x=x1, color='gold', linestyle='--')

    # Add a rectangle to indicate the zoomed area

    rect_width = x1 - x0 + 2 * zoom_slack  # Width of the rectangle
    rect_height = rect_width  # Height of the rectangle, making it square

    # Create the rectangle with the corrected dimensions
    rect = Rectangle((x0 - zoom_slack, x0 - zoom_slack), 
                    rect_width, rect_height, 
                    linewidth=1, edgecolor='red', facecolor='none')

    ax1.add_patch(rect)

    # Plot the zoomed-in heatmap
    start_idx = x0-zoom_slack
    end_idx = x1+zoom_slack
    zoomed_sequence = dna_sequence[start_idx:end_idx]

    zoomed_matrix = ci_matrix[(x0-zoom_slack):(x1+zoom_slack), (x0-zoom_slack):(x1+zoom_slack)]
    if display_values:
        sns.heatmap(zoomed_matrix, cmap='coolwarm', vmax=vmax, ax=ax2, cbar=False, annot=True, fmt=".2f", annot_kws={"size": annot_size})
    else:
        sns.heatmap(zoomed_matrix, cmap='coolwarm', vmax=vmax, ax=ax2, cbar=False)
    ax2.set_aspect('equal')  # Set the aspect ratio to be equal

    if display_dotted_line:
        ax2.axhline(y=zoom_slack, color='gold', linestyle='--')    
        ax2.axhline(y=x1-(x0-zoom_slack), color='gold', linestyle='--')
        ax2.axvline(x=zoom_slack, color='gold', linestyle='--')
        ax2.axvline(x=x1-(x0-zoom_slack), color='gold', linestyle='--')

    if len(zoomed_sequence) == zoomed_matrix.shape[0] and len(zoomed_sequence) == zoomed_matrix.shape[1]:
        # Set the tick labels
        tick_positions = np.arange(len(zoomed_sequence)) + 0.5 # Center the ticks

        ax2.set_xticks(tick_positions)
        ax2.set_yticks(tick_positions)
        ax2.set_xticklabels(list(zoomed_sequence), fontsize=tick_label_fontsize, rotation=0)
        ax2.set_yticklabels(list(zoomed_sequence), fontsize=tick_label_fontsize, rotation=0)
    else:
        print("Warning: Zoomed DNA sequence length does not match the heatmap dimensions.")



    plt.show()
    
    
    



def ic_scale(pwm,background):
    odds_ratio = ((pwm+0.001)/(1.004))/(background[None,:])
    ic = ((np.log((pwm+0.001)/(1.004))/np.log(2))*pwm -\
            (np.log(background)*background/np.log(2))[None,:])
    return pwm*(np.sum(ic,axis=1)[:,None])


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(
            matplotlib.patches.Polygon(
                (np.array([1,height])[None,:]*polygon_coords
                 + np.array([left_edge,base])[None,:]),
                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
        facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+1, base], width=1.0, height=height,
        facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
        facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+1, base], width=1.0, height=height,
        facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.825, base+0.085*height],
        width=0.174, height=0.415*height,
        facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.625, base+0.35*height],
        width=0.374, height=0.15*height,
        facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
        width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
        width=1.0, height=0.2*height, facecolor=color,
        edgecolor=color, fill=True))

    
def plot_u(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.5, base+0.4*height], width=0.95, height=0.8*height,
        facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.025, base+0.4*height], width=0.95, height=0.6*height,
        facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.5, base+0.4*height], width=0.6175, height=0.52*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+0.19125, base+0.4*height], width=0.6175, height=0.6*height,
        facecolor='white', edgecolor='white', fill=True))
    """ax.add_patch(matplotlib.patches.Ellipse(
        xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
        facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=[left_edge+1, base], width=1.0, height=height,
        facecolor='white', edgecolor='white', fill=True))"""
    

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
dna_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
rna_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_u}


def plot_weights_given_ax(ax, array,          
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency="auto",
                 colors=default_colors,
                 plot_funcs=dna_plot_funcs,
                 highlight={},
                 ylabel=""):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    if (subticks_frequency=="auto"):
        subticks_frequency = 1.0 if len(array) <= 40 else int(len(array)/40)
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far,
                      left_edge=i+0.5, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(0.5-length_padding, 0.5+array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(1.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                 figsize=(20,2),
                 ax_transform_func=lambda x: x,
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    ax_transform_func(ax)
    plot_weights_given_ax(ax=ax,
        array=array,
        **kwargs)
    plt.show()
    
    
def dna_plot_weights(array, **kwargs):
  plot_weights(array=array, plot_funcs=dna_plot_funcs, **kwargs)


def rna_plot_weights(array, **kwargs):
  plot_weights(array=array, plot_funcs=rna_plot_funcs, **kwargs)