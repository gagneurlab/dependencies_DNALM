
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
import random

def get_args():
    parser = argparse.ArgumentParser(description="This script computes the convolution of specific filters to the DNA LM matrices and reports the hits.")
    
    parser.add_argument('--species_of_interest', type=str)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--diagonal_mask_width', type=int, default=9)
    parser.add_argument('--n_top', type=int, default=500)
    parser.add_argument('--seq_region_length', type=int, default=1003)
    #if the stored_gene_id_ci_matrices is provided in the command line, the script will load the ci matrices from the path provided in the stored_gene_id_matrices_path
    parser.add_argument('--stored_gene_id_ci_matrices', action='store_true', default=False)
    parser.add_argument('--stored_gene_id_matrices_path', type=str, default='data')
    parser.add_argument('--gradient_based', action='store_true', default=False)

    
    args = parser.parse_args()

    
    return args


if __name__ == "__main__":
    args = get_args()
    species_of_interest = args.species_of_interest
    batch_size = args.batch_size
    mask_width = args.diagonal_mask_width
    n_top = args.n_top
    seq_region_length = args.seq_region_length
    stored_gene_id_ci_matrices = args.stored_gene_id_ci_matrices
    stored_gene_id_matrices_path = args.stored_gene_id_matrices_path
    gradient_based = args.gradient_based


# load the dataframes
print('Loading dataframes with snp preds...')
snp_df = pd.read_parquet(f"samples_upstream_kazachstania_k1_interaction/{species_of_interest}/snp.parquet").reset_index()
id_to_path = pd.read_parquet(f"samples_upstream_kazachstania_k1_interaction/{species_of_interest}/id_to_path.parquet")
snp_df = id_to_path.merge(snp_df, on="transcript_id")


snp_df = snp_df.reset_index(drop=True) # make sure the index is unique
assert (snp_df.index == snp_df["index"]).all() # check that the index is unique


all_genes = snp_df['gene_id'].unique() # In pandas Uniques are returned in order of appearance. That is important later on to optimize reading files


gene_to_idxs_snps_dataframe = snp_df.groupby('gene_id').groups # create a dictionary mapping gene_id to indices in the snp_df dataframe for very fast lookup increases speed by at least 10x
file_cache = OrderedDict()
CACHE_SIZE = 10 

def get_ci_matrix(gene_id):

    global file_cache 

    snp_subset_df = snp_df.loc[gene_to_idxs_snps_dataframe[gene_id]] #.copy()
    if len(snp_subset_df) == 0:
        return None

    if len(snp_subset_df.transcript_id.unique()) >1: # if there is more than one transcript, we only keep the first one
        snp_subset_df = snp_subset_df[snp_subset_df.transcript_id == snp_subset_df.transcript_id.unique()[0]].copy()

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

    # convert to nucleotide representation
    interact_mat_ref = interact_mat[0]
    interact_mat_alt = interact_mat[1:]

    non_zero_log_value = 1e-10 #due to the conversion to bfloat16, we need to add a small value to avoid log(0)

    snp_effect_unordered = torch.log(interact_mat_alt+non_zero_log_value) - torch.log(interact_mat_ref+non_zero_log_value)#.unsqueeze(0)

    snp_effect = torch.zeros((4,1003,1003,4))

    snp_effect[snp_subset_df.iloc[1:]['var_nt_idx'].values, snp_subset_df.iloc[1:]['var_pos'].values] = snp_effect_unordered

    ci_matrix = torch.abs(snp_effect).max(dim=0)[0].max(dim=2)[0]

    return ci_matrix



print(f'GPU Model: {torch.cuda.get_device_name(0)}')


class GeneMatrix(Dataset):
    def __init__(self, gene_ids):
        """
        Args:
            gene_ids (list): List of gene IDs.
        """
        self.gene_ids = gene_ids

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gene_id = self.gene_ids[idx]
        ci_matrix = get_ci_matrix(gene_id).unsqueeze(0) #add a dimension for the channel

        return ci_matrix

dataset = GeneMatrix(all_genes)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



def create_diagonal_filter(diagonal_width, filter_size, remove_mean=True, antidiagonal=False, negative_filter_value=-1, positive_filter_value=1):

    convolutional_filter = negative_filter_value * np.ones((filter_size, filter_size))

    # set the main antidiagonal and its adjacent diagonals to positive_filter_value
    for offset in range(-math.floor(diagonal_width/2), math.ceil(diagonal_width/2)):
        # adjust the range based on the offset to avoid going out of bounds
        if offset < 0:
            rng = np.arange(-offset, filter_size)
        else:
            rng = np.arange(filter_size - offset)
        
        convolutional_filter[rng, rng+offset] = positive_filter_value

    convolutional_filter = convolutional_filter[:,::-1] if antidiagonal else convolutional_filter # flip the filter with respect to the vertical axis to get an antidiagonal
    if remove_mean:
        convolutional_filter = convolutional_filter - convolutional_filter.mean()

    return convolutional_filter

#create a conv mask that masks the diagonal and the adjacent diagonals
def create_conv_diag_mask(mask_width = 9, filter_size = 5, seq_region_length = 1003):

    conv_mask = torch.zeros((seq_region_length - filter_size + 1, seq_region_length - filter_size + 1))

    for k in range(-mask_width // 2 + 1, mask_width // 2 + 1):
        conv_mask += torch.diag(torch.ones(seq_region_length - filter_size + 1 - abs(k)), diagonal=k)

    conv_mask = 1 - conv_mask
    conv_mask =conv_mask.to("cuda")
    return conv_mask

print('Creating filters and masks...')
# anti diagonal filter
filter_size = 5
anti_diag_filter = create_diagonal_filter(diagonal_width = 1, filter_size=filter_size, antidiagonal=True, negative_filter_value=-1)
anti_diag_filter = torch.tensor(anti_diag_filter.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
conv_anti_diag_mask = create_conv_diag_mask(mask_width = 9, filter_size = 5, seq_region_length = seq_region_length)

# diagonal filter
filter_size = 5
diag_filter = create_diagonal_filter(diagonal_width = 1, filter_size=filter_size, antidiagonal=False)
diag_filter = torch.tensor(diag_filter.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
conv_diag_mask = create_conv_diag_mask(mask_width = 9, filter_size = 5, seq_region_length = seq_region_length)

# Filter that averages the col values for each row. This is used to find the row with the highest average value so we can find hotpsots
hotspot_size = 1
hotspot_row_filter = torch.tensor((1/seq_region_length) * np.ones((1, 1, hotspot_size, seq_region_length)), dtype=torch.float32).to("cuda") 
conv_mask_hotspot_row = create_conv_diag_mask(mask_width = 9, filter_size = 1, seq_region_length = seq_region_length)

# Hotspot col
hotspot_size = 1
hotspot_col_filter = torch.tensor((1/seq_region_length) * np.ones((1, 1, seq_region_length, hotspot_size)), dtype=torch.float32).to("cuda") 
conv_mask_hotspot_col = create_conv_diag_mask(mask_width = 9, filter_size = 1, seq_region_length = seq_region_length)


def apply_conv_get_max(batch, conv_filter, conv_mask, apply_mask_first = False):

    if apply_mask_first:
        output = batch * conv_mask #mask the diagonal and adjacent diagonals

    output = F.conv2d(batch, conv_filter)

    # get the argmax of the output for different images of the vetor output of shape (n_images, 1, 999, 999)
    #pytorch doesn't have an implemmentation of an argmax across multiple dims so we need to do it in two steps
    output = output.squeeze(1)

    #torch.amax(output, (1,2), keepdim=True)

    if not apply_mask_first:
        output = output * conv_mask #mask the diagonal and adjacent diagonals

    max_conv_values = torch.amax(output, (1,2), keepdim=True)
    argmax = torch.nonzero(torch.Tensor(output == max_conv_values)).cpu()
    #this returns a vector with the indices of the max values for each image

    # Randomly select one index for each image if there are multiple max values
    selected_indices = []
    for i in range(batch.shape[0]):
        # Filter indices for each image
        indices_per_matrix = argmax[argmax[:, 0] == i]

        # Randomly select one index or use a default value
        if len(indices_per_matrix) > 1:
            random_index = random.choice(indices_per_matrix)
            selected_indices.append(random_index.unsqueeze(0))
        else:
            selected_indices.append(indices_per_matrix)

    argmax = torch.cat(selected_indices,axis=0)

    assert argmax.shape[0] == batch.shape[0] #check that we have an argmax for each image in the batch
    argmax = argmax[:,1:] #discard the first column, which corresponds to the image index

    return max_conv_values.squeeze().cpu(), argmax

print('Started convolutions...')

max_idx_hits_batches_dict = {'hotspot_col':[], 'hotspot_row':[], 'diag':[], 'anti_diag':[]}
max_hits_batches_dict = {'hotspot_col':[], 'hotspot_row':[], 'diag':[], 'anti_diag':[]}

start = time.time()

for i, batch in enumerate(data_loader):
    print(f'batch {i + 1}/{len(data_loader)}')
    batch = batch.to('cuda')  # Move batch to GPU
    
    max_hotspot_col_values, argmax_hotspot_col = apply_conv_get_max(batch, hotspot_col_filter, conv_mask_hotspot_col, apply_mask_first = True)
    max_hotspot_row_values, argmax_hotspot_row = apply_conv_get_max(batch, hotspot_row_filter, conv_mask_hotspot_row, apply_mask_first = True)
    max_diag_values, argmax_diag = apply_conv_get_max(batch, diag_filter, conv_diag_mask, apply_mask_first = False)
    max_anti_diag_values, argmax_anti_diag = apply_conv_get_max(batch, anti_diag_filter, conv_anti_diag_mask, apply_mask_first = False)
    
    max_idx_hits_batches_dict['hotspot_col'].append(argmax_hotspot_col)
    max_idx_hits_batches_dict['hotspot_row'].append(argmax_hotspot_row)
    max_idx_hits_batches_dict['diag'].append(argmax_diag)
    max_idx_hits_batches_dict['anti_diag'].append(argmax_anti_diag)

    max_hits_batches_dict['hotspot_col'].append(max_hotspot_col_values)
    max_hits_batches_dict['hotspot_row'].append(max_hotspot_row_values)
    max_hits_batches_dict['diag'].append(max_diag_values)
    max_hits_batches_dict['anti_diag'].append(max_anti_diag_values)


end = time.time()
print(f'Elapsed time: {end - start}')

# concatenate each list of tensors into a single tensor for each type of hit
max_hits_dict = {key: torch.cat(max_hits_batches_dict[key]).cpu().numpy() for key in max_hits_batches_dict}

#build a dataframe with the max hits for each type of hit, with the gene id and the position of the hit
top_hits_df = pd.DataFrame(max_hits_dict)
top_hits_df['gene_id'] = data_loader.dataset.gene_ids

# set the index name to id_in_batch to avoid confusion with the gene_id
top_hits_df.index.name = 'id_in_batch'

# concat all max_idx_hits_batches_list into a single tensor per conv filter
max_idx_hits_cat_dict = {key: torch.cat(max_idx_hits_batches_dict[key], axis=0) for key in max_idx_hits_batches_dict}

# get the indices of the top hits for each type of hit in a dataframe
max_xy_list = []
for key in max_idx_hits_cat_dict:

    conv_type_dict = {}

    conv_type_dict[f'hit_x'] = max_idx_hits_cat_dict[key][:, 0]
    conv_type_dict[f'hit_y'] = max_idx_hits_cat_dict[key][:, 1]
    conv_type_df = pd.DataFrame(conv_type_dict)
    conv_type_df['gene_id'] = data_loader.dataset.gene_ids
    conv_type_df = conv_type_df.melt(id_vars='gene_id', var_name='axis', value_name='coord')
    conv_type_df['conv_type'] = key

    max_xy_list.append(conv_type_df)

max_xy_df = pd.concat(max_xy_list, axis=0)

top_hits_df = top_hits_df.reset_index().melt(id_vars=['id_in_batch', 'gene_id'], var_name='conv_type', value_name='max_value')
top_hits_idx_df = top_hits_df.merge(max_xy_df, on=['gene_id', 'conv_type'])


if not stored_gene_id_ci_matrices:
    if not gradient_based:
        gene_transcript_df = snp_df.loc[:, ['gene_id', 'transcript_id']].drop_duplicates().copy()
    else:
        gene_transcript_df = id_to_path.loc[:, ['gene_id', 'transcript_id']].drop_duplicates().copy()

else:
    real_seqs_df = pd.read_csv(os.path.join(f'sequences/{species_of_interest}/{species_of_interest}_upstream_1000bp_with_start_cds_downstream_509bp_with_stop_longest_cds.tsv'), sep='\t')
    real_seqs_df = real_seqs_df.loc[:, ['gene_id', 'transcript_id', 'five_prime_seq']].copy().dropna()
    real_seqs_df = real_seqs_df[real_seqs_df['five_prime_seq'].apply(lambda x: len(x))==1003].reset_index(drop=True)

    assert real_seqs_df.gene_id.unique().size == real_seqs_df.shape[0]

    gene_transcript_df = real_seqs_df.loc[:,['gene_id','transcript_id']].drop_duplicates().copy()

if len(gene_transcript_df) != len(all_genes):
        print('The matrices were not computed for all genes')
        print('Number of genes with matrix ', len(all_genes))
        print('Number of genes with transcript ', len(gene_transcript_df))

top_hits_idx_df = top_hits_idx_df.merge(gene_transcript_df, on='gene_id')

#get the length of the conv output for each type of hit so we can later map the hits to the genomic positions
conv_output_length_dict = {'hotspot_col': conv_mask_hotspot_row.shape[0], 'hotspot_row': conv_mask_hotspot_col.shape[0], 'diag': conv_diag_mask.shape[0], 'anti_diag': conv_anti_diag_mask.shape[0]}
conv_output_length_df = pd.DataFrame(conv_output_length_dict.items(), columns=['conv_type', 'conv_output_length'])

top_hits_idx_df = top_hits_idx_df.merge(conv_output_length_df, on='conv_type')

print('Saving...')
if not gradient_based:
    top_hits_idx_df.to_csv(os.path.join(f'data/top_hits_{species_of_interest}.tsv'), sep='\t', index=False)
else:
    top_hits_idx_df.to_csv(os.path.join(f'data/top_hits_{species_of_interest}_grad_based.tsv'), sep='\t', index=False)




display_plots = False
if display_plots:
    # create a new dataset with the top hits
    top_genes = top_hits_df.sort_values(by='conv_value', ascending=False)['gene_id'].values[:n_top]
    top_hits_dataset = GeneMatrix(top_genes)
    # create a new data loader with the top hits
    top_hits_data_loader = DataLoader(top_hits_dataset, batch_size=n_top, shuffle=False)


    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec

    square_size = 50  # Size of the square to draw
    half_square = square_size // 2  # Half the size to center the square
    max_display_value = 8 # saturation for the big heatmap
    batch_i = -1 # batch index to display

    for batch in top_hits_data_loader: # ONLY IMPLEMMENTED ASSUMING BATCH SIZE = 1 = N_TOP_HITS
        for image, max_idx, gene_id in zip(batch, max_idx_hits_all_genes[top_hits_df.sort_values(by='conv_value', ascending=False).index], top_hits_data_loader.dataset.gene_ids):

            print(gene_id)

            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 0.05])  # Width ratios for main plot, zoomed plot and colorbar

            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            # Original Image with Square
            im1 = ax1.imshow(image.squeeze().cpu().numpy(), cmap="coolwarm",vmax=max_display_value)
            row, col = max_idx
            rect = patches.Rectangle((col - half_square, row - half_square), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)

            # Extract and Show Zoomed Region
            zoomed_region = image.squeeze()[max(0, row - half_square):row + half_square, max(0, col - half_square):col + half_square].cpu().numpy()
            im2 = ax2.imshow(zoomed_region, cmap="coolwarm")
            ax2.set_title("Zoomed Region")

            # Add colorbars
            cb_ax1 = plt.subplot(gs[2])
            plt.colorbar(im1, cax=cb_ax1)# PLOT 2 HAS a DIFFERENT COLORSCALE not represented here!!!
            
            plt.show()



