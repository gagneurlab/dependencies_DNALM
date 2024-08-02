import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import convolve2d
import math
import argparse
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
from tqdm import tqdm

species='spombe'
print(species)

if species=='spombe':
    all_maps_folder = 'dnalm_genome_wide_predictions/schizosaccharomyces_pombe_genome_wide'
    all_maps = os.listdir(all_maps_folder)
    # Regular expression pattern to extract chromosome, start, and end
    pattern = r'^([A-Za-z]+)(\d+):(\d+)_plus.npy$'

    # Extract chromosome, start, and end
    def get_chromosome_start_end(filename, attribute=None):
        match = re.match(pattern, filename)
        chromosome = match.group(1)
        start = match.group(2)
        end = match.group(3)

        if attribute=='chromosome':
            return chromosome
        elif attribute=='start':
            return start
        elif attribute=='end':
            return end
        
    map_file_df = pd.DataFrame(all_maps, columns=['filename'])
    map_file_df['Chromosome'] = map_file_df['filename'].apply(get_chromosome_start_end, attribute='chromosome')
    map_file_df['Start'] = map_file_df['filename'].apply(get_chromosome_start_end, attribute='start').astype(int)
    map_file_df['End'] = map_file_df['filename'].apply(get_chromosome_start_end, attribute='end').astype(int)
    map_file_df['filename'] = map_file_df['filename'].apply(lambda el: os.path.join(all_maps_folder, el))


else:
    all_maps_folder = 'dnalm_genome_wide_predictions/saccharomyces_cerevisiae_genome_wide'
    all_maps = os.listdir(all_maps_folder)


    def build_map_file_df(all_maps, root_folder='dnalm_genome_wide_predictions/saccharomyces_cerevisiae_genome_wide'):

        map_file_df = pd.DataFrame(all_maps, columns=['filename'])
        map_file_df['Chromosome'] = map_file_df['filename'].apply(lambda el: el.split(':')[0])
        map_file_df['Start'] = map_file_df['filename'].apply(lambda el: int(el.split(':')[1].split('-')[0]))
        map_file_df['End'] = map_file_df['filename'].apply(lambda el: int(el.split(':')[1].split('-')[1].split('_')[0]))
        map_file_df['Strand'] = map_file_df['filename'].apply(lambda el: el.split('_')[1].split('.')[0])

        map_file_df['filename'] = map_file_df['filename'].apply(lambda el: os.path.join(root_folder, el))

        return map_file_df

    map_file_df = build_map_file_df(all_maps).sort_values(['Chromosome', 'Start'])
    map_file_df = map_file_df[map_file_df['Strand']=='plus'].copy().reset_index(drop=True) #only consider plus strand

    


map_file_df['window_length'] = map_file_df['End'] - map_file_df['Start']
map_file_df = map_file_df[map_file_df['window_length']==1003].copy().reset_index(drop=True)


def get_ci_matrix(filename):

    return np.load(filename)


# print(f'GPU Model: {torch.cuda.get_device_name(0)}')


class GeneMatrix(Dataset):
    def __init__(self, map_file_df):
        """
        Args:
            gene_ids (list): List of gene IDs.
        """
        self.filenames = map_file_df['filename']
        self.chromosomes = map_file_df['Chromosome']
        self.starts = map_file_df['Start']
        self.ends = map_file_df['End'] - 3 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file = self.filenames[idx]
        ci_matrix = torch.tensor(get_ci_matrix(file)).unsqueeze(0) #add a dimension for the channel
        chromosome = self.chromosomes[idx]
        start = self.starts[idx]
        end = self.ends[idx]

        return (ci_matrix[:, :1000,:1000], chromosome, start, end) #remove last 3 positions which would belong to the "start codon" in the Species LM

dataset = GeneMatrix(map_file_df)
batch_size=1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def plot_ci_matrix(ci_matrix, plot_size=10, vmax=None):
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))

    if vmax is not None:
        sns.heatmap(ci_matrix, cmap='coolwarm', ax=ax, vmax=vmax, 
                    cbar_kws={'shrink': .4})
    else:
        sns.heatmap(ci_matrix, cmap='coolwarm', ax=ax, 
                    cbar_kws={'shrink': .4})

    ax.set_aspect('equal')
    plt.show()


# ## Analyze distance vs dependency


seq_len = 1000


def create_rowise_dist_to_diag_matrix(N):
    # Create a range vector from 0 to N-1
    range_vector = np.arange(N)

    # subtract the transpose of the range_vector from itself and take the absolute value
    matrix = range_vector - range_vector[:, None]
    return matrix


dist_matrix = create_rowise_dist_to_diag_matrix(seq_len)
dist_matrix


#create a matrix where every row contains the row index
row_matrix = np.repeat(np.arange(seq_len)[:, np.newaxis], seq_len, axis=1)
row_matrix


n_values_to_sample_per_batch = 1_000
dist_values_dfs_list = []

for i, batch in enumerate(tqdm(data_loader)):
    #print(f'Processing batch {i+1} of {len(data_loader)}...')
    maps, chromosomes, starts, ends = batch

    row_matrix_batch = np.repeat(row_matrix[np.newaxis, :, :], maps.shape[0], axis=0).flatten()
    dist_matrix_batch = np.repeat(dist_matrix[np.newaxis, :, :], maps.shape[0], axis=0).flatten()

    chromosomes_expanded = np.repeat(chromosomes, maps.shape[-2] * maps.shape[-1])
    starts_expanded = np.repeat(starts, maps.shape[-2] * maps.shape[-1])
    ends_expanded = np.repeat(ends, maps.shape[-2] * maps.shape[-1])

    maps = maps.squeeze().numpy().flatten()

    try:
        dist_values_df_batch = pd.DataFrame({'Chromosome': chromosomes_expanded, 
            'Start_window': starts_expanded, 
            'dist': dist_matrix_batch,
            'row': row_matrix_batch})
    except:
        print(f'{chromosomes_expanded.shape},{starts_expanded.shape},{dist_matrix_batch.shape},{row_matrix_batch.shape}' )

    dist_values_df_batch['Start'] = dist_values_df_batch['Start_window']+dist_values_df_batch['row']
    dist_values_df_batch['End'] = dist_values_df_batch['Start_window'] + dist_values_df_batch['row'] + dist_values_df_batch['dist']

    dist_values_df_batch = dist_values_df_batch.reset_index().rename({'index':'array_index'}, axis=1)
    unique_dependency_pos = dist_values_df_batch.loc[:, ['Chromosome', 'Start', 'End']].drop_duplicates()


    sample_pos_df = unique_dependency_pos.sample(n_values_to_sample_per_batch)
    dist_values_sample_df_batch = dist_values_df_batch.merge(sample_pos_df, on=['Chromosome', 'Start','End'])
    dist_values_sample_df_batch['value'] = maps[dist_values_sample_df_batch.array_index.values] 
    dist_values_sample_df_batch = dist_values_sample_df_batch.groupby(['Chromosome', 'Start','End']).mean() #get the mean of the values in overlapping matrices 
    dist_values_sample_df_batch = dist_values_sample_df_batch.reset_index().drop(columns=['array_index','Start_window','row'])
    dist_values_dfs_list.append(dist_values_sample_df_batch)

dist_values_df = pd.concat(dist_values_dfs_list, axis=0)

dist_values_df.to_csv(f'data/dist_values_{species}_b1_1k_all_chrs_genome_wide.tsv', index=False, sep='\t')

