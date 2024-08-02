import pandas as pd 
import pyranges as pr
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import convolve2d
import math
import torch
from collections import OrderedDict
from plotnine import *
import argparse
import glob
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from collections import Counter
from Bio.Seq import Seq


def get_args():
    parser = argparse.ArgumentParser(description="This script reports for a certain species on pattern hits stats, location in the genome and plots them.")
    
    parser.add_argument('--species_of_interest', type=str)
    parser.add_argument('--figures_path', type=str, default='figures/')
    parser.add_argument('--top_hits_path', type=str)
    parser.add_argument('--stored_gene_id_ci_matrices', action='store_true', default=False)
    parser.add_argument('--stored_gene_id_matrices_path', type=str, default='data')

    
    args = parser.parse_args()

    
    return args


if __name__ == "__main__":
    args = get_args()
    species_of_interest = args.species_of_interest
    figures_path = args.figures_path
    top_hits_path = args.top_hits_path
    stored_gene_id_ci_matrices = args.stored_gene_id_ci_matrices
    stored_gene_id_matrices_path = args.stored_gene_id_matrices_path




def get_species_gtf_fasta_files(species_of_interest, file_type):

    species_folders = glob.glob(f'Sequences/ensembl_53/{file_type}/**/{species_of_interest}', recursive=True)
    
    assert len(species_folders) == 1
    species_folder = species_folders[0]

    pattern = os.path.join(species_folder, '**', 'dna', '*dna.toplevel.fa.gz') if file_type == 'fasta' else os.path.join(species_folder, '**', '*.gtf.gz')
    files = glob.glob(pattern, recursive=True)
    if len(files) > 1:
        if file_type == 'fasta':
            print(f'multiple fasta files found: {files}')
        else:
            for file in files:
                if 'chr' in file.split('.')[-3]:
                    #remove file from list
                    files.remove(file)
            if len(files) > 1:
                print(f'multiple gtf files found even after filtering for chr: {files}')

    print(files[0])
    return files[0]


print('Selected GTF and fasta files:')
fasta_path = get_species_gtf_fasta_files(species_of_interest, 'fasta')
gtf_path = get_species_gtf_fasta_files(species_of_interest, 'gtf')


if not stored_gene_id_ci_matrices:
    # load the dataframes
    print('Loading dataframes with snp preds...')
    snp_df = pd.read_parquet(f"/samples_upstream_kazachstania_k1_interaction/{species_of_interest}/snp.parquet").reset_index()
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

else:
    matrices_path_species = os.path.join(stored_gene_id_matrices_path, species_of_interest + '_k1_model')

    print(f'Loading stored gene id ci matrices from {matrices_path_species}...')

    #create a dict with the file names and the file paths
    genes_to_files = {f.split('.npy')[0]: os.path.join(matrices_path_species, f) for f in os.listdir(matrices_path_species)}
    all_genes = list(genes_to_files.keys())
    
    def get_ci_matrix(gene_id):
        if gene_id in genes_to_files:
            return torch.tensor(np.load(genes_to_files[gene_id])).to(torch.float32)
        else:
            return None


top_hits_df = pd.read_csv(top_hits_path, sep='\t')
top_hits_matrix_coords_df = top_hits_df.pivot(index=top_hits_df.columns.difference(['axis', 'coord']), columns='axis', values='coord').reset_index().copy()

#the indexes of the hits are not with respect to the start. The start position corresponds to the last index of the conv matrix 
top_hits_df['coord'] = top_hits_df['conv_output_length'] - top_hits_df['coord'] - 1 # -1 because the index is 0-based
top_hits_df = top_hits_df.pivot(index=top_hits_df.columns.difference(['axis', 'coord']), columns='axis', values='coord').reset_index()

gtf_pr = pr.read_gtf(gtf_path)
gtf_df = gtf_pr.df.copy()

def get_start_beginning(df): #get the start codon position that is at the beginning of the transcript. In rare cases the start codon is split in 2 with one intron in between
    df = df.sort_values('Start')
    if df.Strand.values[0] == '+':
        return df.iloc[[0]]
    else:
        return df.iloc[[-1]]

gtf_start = gtf_df[gtf_df.Feature=='start_codon'].groupby(['gene_id','transcript_id']).apply(get_start_beginning).reset_index(drop=True)
gtf_top_hits_df = top_hits_df.merge(gtf_start, on=['gene_id', 'transcript_id'])

assert gtf_top_hits_df.shape[0] == top_hits_df.shape[0] #assert that all transcripts are in the gtf


def get_genomic_hit_position(row, slack=2):
    if row.Strand == '+':
        x_start = row.Start - row.hit_x - slack
        x_end = row.Start - row.hit_x + slack + 1 # +1 because the end is exclusive
        y_start = row.Start - row.hit_y - slack
        y_end = row.Start - row.hit_y + slack + 1


        return (x_start, x_end, y_start, y_end, min(x_start, y_start), max(x_end, y_end))
    elif row.Strand == '-':
        x_start = row.End + row.hit_x - slack
        x_end = row.End + row.hit_x + slack + 1
        y_start = row.End + row.hit_y - slack
        y_end = row.End + row.hit_y + slack + 1
        return (x_start, x_end, y_start, y_end, min(x_start, y_start), max(x_end, y_end))
    else:
        raise ValueError('strand must be + or -')


hits_genomic_pos_df = gtf_top_hits_df.apply(get_genomic_hit_position, axis=1, result_type='expand')
hits_genomic_pos_df.columns = ['start_hit_x_genomic', 'end_hit_x_genomic', 'start_hit_y_genomic', 'end_hit_y_genomic', 'start_genomic', 'end_genomic']
gtf_top_hits_df = pd.concat([gtf_top_hits_df, hits_genomic_pos_df], axis=1)

hits_genomic_pos_x_df = gtf_top_hits_df.loc[:, ['Chromosome', 'Strand', 'start_hit_x_genomic', 'end_hit_x_genomic','conv_type','max_value','gene_id', 'hit_x']].copy().rename(columns={'start_hit_x_genomic': 'Start', 'end_hit_x_genomic': 'End', 'gene_id': 'downstream_gene_id'})
hits_genomic_pos_y_df = gtf_top_hits_df.loc[:, ['Chromosome', 'Strand','start_hit_y_genomic', 'end_hit_y_genomic','conv_type','max_value','gene_id', 'hit_y']].copy().rename(columns={'start_hit_y_genomic': 'Start', 'end_hit_y_genomic': 'End', 'gene_id': 'downstream_gene_id'})
hits_genomic_pos_merged_df = gtf_top_hits_df.loc[:, ['Chromosome', 'Strand','start_genomic', 'end_genomic','conv_type','max_value','gene_id']].copy().rename(columns={'start_genomic': 'Start', 'end_genomic': 'End', 'gene_id': 'downstream_gene_id'})

hits_genomic_pos_x_pr = pr.PyRanges(hits_genomic_pos_x_df.drop(columns=['Strand'])) # want to check overlaps regardless of strand
hits_genomic_pos_y_pr = pr.PyRanges(hits_genomic_pos_y_df.drop(columns=['Strand']))

# next we will extract the introns from the exon annotations
# get only the protein coding exons. The non coding ones will be treated differently later on
gtf_df = gtf_pr.df.copy()
gtf_exons_df = gtf_df[(gtf_df.Feature == "exon") & (gtf_df.gene_biotype == "protein_coding")].copy() 

#For every gene we get the start and end coordinates and get the gaps between the exons as coordinates for introns
def get_introns(gene_df):
    
    gene_pr = pr.PyRanges(gene_df)
    gene = gene_pr.gene_id.unique()[0]
    print(gene)
    
    #Get boundaries of the whole gene defined by the exon
    #Pyranges boundaries returns them as float (very annoying bug) where they should be int. 
    #Converting them back to df and then to pyranges solves it ;)
    gene_boundaries_pr = pr.PyRanges(gene_pr.boundaries(group_by='Feature').df)

    introns_df = gene_boundaries_pr.subtract(gene_pr).df #get introns
    introns_df['Feature'] = 'intron'
    
    return introns_df

print('Getting introns...')
gtf_introns_df = gtf_exons_df.groupby('gene_id').apply(get_introns)
gtf_introns_df = gtf_introns_df.droplevel(1).reset_index()

gtf_w_introns_df = pd.concat([gtf_df, gtf_introns_df], axis=0)

# now we get the CDS five_prime_utr and three_prime_utr of the coding genes plus the gene annoation of the non coding genes
gtf_processed_df = gtf_w_introns_df[(gtf_w_introns_df.Feature=='CDS') | (gtf_w_introns_df.Feature=='five_prime_utr') | (gtf_w_introns_df.Feature=='three_prime_utr') | 
                                    (gtf_w_introns_df.Feature=='intron') | ((gtf_w_introns_df.gene_biotype!='protein_coding') & (gtf_w_introns_df.Feature=='gene'))].copy()

gtf_processed_df['processed_feature'] = gtf_processed_df.apply(lambda row: row.Feature if 
                                                               (row.gene_biotype == 'protein_coding') or (row.Feature=='intron') else row.gene_biotype, axis=1)

#some genomic intervals may belong to more than one gene id annotation. Check for example the intron in SPAC823.02 and SPAC823.17 in S Pombe. 
#Or consider gene ids that are different but belong to the same gene (ex. ENSRNA049626437). we will keep one genomic interval + feature randomly if duplicates exist
gtf_processed_df.drop_duplicates(subset=['Chromosome', 'Start', 'End', 'Strand', 'processed_feature'], keep='first', inplace=True)
gtf_processed_pr = pr.PyRanges(gtf_processed_df)

#get overlapping regions
overlapping_pr = hits_genomic_pos_x_pr.join(gtf_processed_pr, suffix='_y', report_overlap=True, how='left')
overlapping_df = overlapping_pr.df.copy()


#unzip the fasta file in the current directory and read the file
import gzip
import shutil

temp_unzipped_fasta_path = f'{species_of_interest}_temp.fa'
with gzip.open(fasta_path, 'rb') as f_in:
    with open(temp_unzipped_fasta_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

n_extended_bases_either_side = 100
hits_genomic_pos_merged_pr = pr.PyRanges(hits_genomic_pos_merged_df).slack(n_extended_bases_either_side)
hits_genomic_pos_merged_pr.seq_around = pr.get_sequence(hits_genomic_pos_merged_pr, path=temp_unzipped_fasta_path)

hits_genomic_pos_x_pr = pr.PyRanges(hits_genomic_pos_x_df)
hits_genomic_pos_y_pr = pr.PyRanges(hits_genomic_pos_y_df)
hits_genomic_pos_x_pr.seq_x = pr.get_sequence(hits_genomic_pos_x_pr, path=temp_unzipped_fasta_path)
hits_genomic_pos_y_pr.seq_y = pr.get_sequence(hits_genomic_pos_y_pr, path=temp_unzipped_fasta_path)

#remove the unzipped fasta file
if os.path.exists(temp_unzipped_fasta_path):
    os.remove(temp_unzipped_fasta_path)

hits_x_y_seqs_df = hits_genomic_pos_x_pr.df.drop(columns=['Start','End']).drop_duplicates().merge(
    hits_genomic_pos_y_pr.df.drop(columns=['Start','End']).drop_duplicates(), on=['downstream_gene_id', 'Chromosome', 'max_value','Strand'])


def is_rev_comp(seq1, seq2, allow_wobble=True):
    for el in zip(seq1, seq2[::-1]):
        if el[0] == 'A' and el[1] == 'T':
            continue
        elif el[0] == 'T' and el[1] == 'A':
            continue
        elif el[0] == 'C' and el[1] == 'G':
            continue
        elif el[0] == 'G' and el[1] == 'C':
            continue
        elif allow_wobble and el[0] == 'G' and el[1] == 'T':
            continue
        elif allow_wobble and el[0] == 'T' and el[1] == 'G':
            continue
        else:
            return False
    
    return True


def count_rev_comp(seq1, seq2, allow_wobble=True):
    count = 0
    for el in zip(seq1, seq2[::-1]):
        if el[0] == 'A' and el[1] == 'T':
            count += 1
            continue
        elif el[0] == 'T' and el[1] == 'A':
            count += 1
            continue
        elif el[0] == 'C' and el[1] == 'G':
            count += 1
            continue
        elif el[0] == 'G' and el[1] == 'C':
            count += 1
            continue
        elif allow_wobble and el[0] == 'G' and el[1] == 'T':
            count += 1
            continue
        elif allow_wobble and el[0] == 'T' and el[1] == 'G':
            count += 1
            continue
    
    return count/len(seq1)

def count_equal(seq1, seq2):
    count = 0
    for el in zip(seq1, seq2):
        if el[0] == el[1]:
            count += 1
            continue
    
    return count/len(seq1)


hits_x_y_seqs_df['is_rev_complemment'] = hits_x_y_seqs_df.apply(lambda row: is_rev_comp(row.seq_x.upper(), row.seq_y.upper()), axis=1)
hits_x_y_seqs_df['frac_rev_complemment'] = hits_x_y_seqs_df.apply(lambda row: count_rev_comp(row.seq_x.upper(), row.seq_y.upper()), axis=1)
hits_x_y_seqs_df['is_equal'] = hits_x_y_seqs_df['seq_x'] == hits_x_y_seqs_df['seq_y']
hits_x_y_seqs_df['frac_equal'] = hits_x_y_seqs_df.apply(lambda row: count_equal(row.seq_x.upper(), row.seq_y.upper()), axis=1)

overlapping_df = overlapping_df.merge(hits_x_y_seqs_df.drop('Strand',axis=1), on=['downstream_gene_id', 'Chromosome', 'conv_type', 'max_value','hit_x'])

def rank_group(df):
    df_sorted = df.sort_values('max_value', ascending=False)
    df_sorted['rank'] = np.arange(len(df)) + 1
    return df_sorted

overlapping_df = overlapping_df.groupby('conv_type').apply(rank_group).reset_index(drop=True)
overlapping_df['processed_feature'] = overlapping_df['processed_feature'].str.replace('-1', 'intergenic')
overlapping_df.to_csv(os.path.join(project_path, f'data/{species_of_interest}_top_hits_with_gtf_overlaps.tsv'), sep='\t', index=False)

p = (ggplot(overlapping_df, aes('rank', 'max_value', color='processed_feature')) + geom_point() + facet_wrap('~conv_type', scales='free_y')  + theme_seaborn() + labs(x='Rank', y='Max convolution value', color='Gene biotype'))

if not os.path.exists(os.path.join(figures_path, species_of_interest)):
    os.makedirs(os.path.join(figures_path, species_of_interest))

p.save(os.path.join(figures_path, f'{species_of_interest}/rank_vs_max_value_{species_of_interest}_conv_filters.png'))


for conv_type in overlapping_df['conv_type'].unique():
    p = (ggplot(overlapping_df[overlapping_df['conv_type']==conv_type], aes('rank', 'max_value', color='processed_feature')) + geom_point() + 
        facet_wrap('~processed_feature', scales='free_y') + theme_seaborn() + theme(figure_size=(8, 6)) + labs(title=conv_type, x='Rank', y='Max convolution value', color='Gene biotype'))
    p.save(os.path.join(figures_path, f'{species_of_interest}/rank_vs_max_value_{species_of_interest}_{conv_type}.png'))


for conv_type in overlapping_df['conv_type'].unique():
    p = (ggplot(overlapping_df[overlapping_df['conv_type']==conv_type], aes('processed_feature', 'max_value')) + geom_boxplot() + 
        theme_seaborn() + theme(axis_text_x=element_text(rotation=45, hjust=1)) + labs(title=conv_type, x='Gene biotype', y='Max convolution value'))
    p.save(os.path.join(figures_path, f'{species_of_interest}/max_value_vs_processed_feature_boxplot_{species_of_interest}_{conv_type}.png'))


label_dict = {'anti_diag': 'Anti Diagonal', 'diag': 'Diagonal'}

p = (ggplot(overlapping_df[(overlapping_df.conv_type != 'hotspot_row') & (overlapping_df.conv_type != 'hotspot_col')], 
    aes('is_equal', 'max_value')) +  geom_violin() + geom_boxplot(width=.05) + facet_wrap('conv_type', scales='free_y',  labeller=label_dict) + theme_seaborn() +labs(x='Equal interacting sequences', y='Max convolution value'))
p.save(os.path.join(figures_path, f'{species_of_interest}/max_value_vs_equal_seqs_violin_boxplot_{species_of_interest}.png'))

p = (ggplot(overlapping_df[(overlapping_df.conv_type != 'hotspot_row') & (overlapping_df.conv_type != 'hotspot_col') & (overlapping_df.conv_type != 'anti_diag')],
    aes('factor(frac_equal)', 'max_value')) +  geom_violin() + geom_boxplot(width=.05) + theme_seaborn() + theme(figure_size=(8, 5))
    +labs(x='Fraction of equal nucleotides', y='Max convolution value'))
p.save(os.path.join(figures_path, f'{species_of_interest}/max_value_vs_frac_equal_violin_boxplot_{species_of_interest}.png'))


p = (ggplot(overlapping_df[(overlapping_df.conv_type != 'hotspot_row') & (overlapping_df.conv_type != 'hotspot_col')], 
    aes('is_rev_complemment', 'max_value')) +  geom_violin() + geom_boxplot(width=.05) + facet_wrap('conv_type', scales='free_y',  labeller=label_dict) + theme_seaborn() 
    +labs(x='Interacting sequences can base-pair', y='Max convolution value'))
p.save(os.path.join(figures_path, f'{species_of_interest}/max_value_vs_can_basepair_violin_boxplot_{species_of_interest}.png'))

p = (ggplot(overlapping_df[(overlapping_df.conv_type != 'hotspot_row') & (overlapping_df.conv_type != 'hotspot_col') & (overlapping_df.conv_type != 'diag')],
    aes('factor(frac_rev_complemment)', 'max_value')) +  geom_violin() + geom_boxplot(width=.05) + theme_seaborn() + theme(figure_size=(8, 5))
    +labs(x='Fraction of nucleotides that can base-pair', y='Max convolution value'))
p.save(os.path.join(figures_path, f'{species_of_interest}/max_value_vs_frac_basepair_violin_boxplot_{species_of_interest}.png'))



def find_repeats(sequence, min_length=5):
    """Find repeated sequences of a minimum length."""
    repeats = [sequence[i:i+min_length] for i in range(len(sequence) - min_length + 1)]
    repeat_counts = Counter(repeats)
    return [repeat for repeat, count in repeat_counts.items() if count > 1]

def color_print(sequence, min_length=5):
    """Print the sequence with repeats in different colors."""
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    repeats = find_repeats(sequence, min_length)
    colors = list(color_codes.keys())[:-2]  # exclude white and reset

    for repeat in repeats:
        color = colors[repeats.index(repeat) % len(colors)]
        colored_repeat = color_codes[color] + repeat + color_codes['reset']
        sequence = sequence.replace(repeat, colored_repeat)

    return sequence

def plot_matrix(gene_id, coords_df, conv_type, max_display_value=8, zoom_in_size=50, min_repeat_length=5, revcomp_seq=False, print_repeats=False):

    square_size = zoom_in_size  # Size of the square to draw
    half_square = square_size // 2  # Half the size to center the square
    max_display_value = 8 # saturation for the big heatmap

    ci_matrix = get_ci_matrix(gene_id)
  
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 0.05])  # Width ratios for main plot, zoomed plot and colorbar

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Original Image with Square
    im1 = ax1.imshow(ci_matrix, cmap="coolwarm",vmax=max_display_value)
    # add a title
    ax1.set_title(f"{gene_id}, {conv_type}")
    row, col = coords_df.loc[(coords_df.gene_id == gene_id) & (coords_df.conv_type==conv_type), ["hit_x", "hit_y"]].values[0]
    rect = patches.Rectangle((col - half_square, row - half_square), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)

    # Extract and Show Zoomed Region
    zoomed_region = ci_matrix[max(0, row - half_square):row + half_square, max(0, col - half_square):col + half_square].cpu().numpy()
    im2 = ax2.imshow(zoomed_region, cmap="coolwarm")
    ax2.set_title("Zoomed Region")

    # Add colorbars
    cb_ax1 = plt.subplot(gs[2])
    plt.colorbar(im1, cax=cb_ax1)# PLOT 2 HAS a DIFFERENT COLORSCALE not represented here!!!

    seq = hits_genomic_pos_merged_pr[hits_genomic_pos_merged_pr.downstream_gene_id==gene_id].seq_around.values[0]
    if revcomp_seq:
        seq = str(Seq(seq).reverse_complement())
        
    seq_text = 'sequence around the hit\n' + seq #('sequence around the hit\n' + color_print(seq, min_repeat_length)) if print_repeats else 'sequence around the hit\n' + seq

    # Add sequence text to the plot
    fig.text(0, 0.02, seq_text, ha='left', wrap=True, fontsize=8)

    return fig



print('Plotting top ci matrices for different filters...')

top_n_to_plot = 10
for conv_type in top_hits_df.conv_type.unique():
    print(conv_type)
    genes_to_plot = top_hits_df[top_hits_df.conv_type == conv_type].sort_values('max_value', ascending=False).iloc[:top_n_to_plot].gene_id.to_list()
    for i, gene_id in enumerate(genes_to_plot):
        print(gene_id)
        if conv_type == 'diag':
            plot_matrix(gene_id, coords_df = top_hits_matrix_coords_df, conv_type=conv_type, min_repeat_length=15, print_repeats=True)
        else:
            plot_matrix(gene_id, coords_df = top_hits_matrix_coords_df, conv_type=conv_type)

        if not os.path.exists(os.path.join(figures_path, species_of_interest, f'top_{conv_type}')):
            os.makedirs(os.path.join(figures_path, species_of_interest, f'top_{conv_type}'))

        plt.savefig(os.path.join(figures_path,  f'{species_of_interest}/top_{conv_type}/{i+1}_top_{species_of_interest}_{conv_type}_{gene_id}_ci_matrix.png'))

        #close the figure to avoid memory problems
        plt.close()


def select_row_or_keep(group):
    if (group['processed_feature'] == 'tRNA').any():
        # Select the row where gene_biotype is 'tRNA'
        return group[group['processed_feature'] == 'tRNA']
    elif (group['processed_feature'] == 'rRNA').any():
        # Select the row where gene_biotype is 'rRNA'
        return group[group['processed_feature'] == 'rRNA']
    else:
        # Keep all rows
        return group

# for a certain downstream gene_id, only keep the row where processed_feature is 'tRNA' or 'rRNA' if it exists. Otherwise, keep all rows. 
# that ensures that if an antidiagonal hit is found overlapping a gene or a tRNA/rRNA gene, the tRNA/rRNA gene is selected
result_df = overlapping_df[overlapping_df.conv_type == 'anti_diag'].groupby('downstream_gene_id').apply(select_row_or_keep).reset_index(drop=True) 


# plot genes with top anti_diag values but that are not tRNAs or rRNAs
top_n_to_plot = 300
genes_to_plot = result_df[(result_df.conv_type == 'anti_diag') & (result_df.processed_feature != 'tRNA') & (result_df.processed_feature != 'rRNA')].sort_values('max_value', ascending=False).iloc[:top_n_to_plot].downstream_gene_id.unique()

for i, gene_id in enumerate(genes_to_plot):

    print(gene_id)
    plot_matrix(gene_id, coords_df = top_hits_matrix_coords_df, conv_type='anti_diag')

    if not os.path.exists(os.path.join(figures_path, species_of_interest, 'top_non_tRNA_rRNA_anti_diagonal_hits')):
            os.makedirs(os.path.join(figures_path, species_of_interest, 'top_non_tRNA_rRNA_anti_diagonal_hits'))

    plt.savefig(os.path.join(figures_path,  f'{species_of_interest}/top_non_tRNA_rRNA_anti_diagonal_hits/{i+1}_top_{species_of_interest}_{gene_id}_ci_matrix.png'))


    plt.close()


if 'five_prime_utr' in result_df.processed_feature.unique():

    # plot genes with top anti_diag values but that are not tRNAs or rRNAs
    top_n_to_plot = 50
    genes_to_plot = result_df[(result_df.conv_type == 'anti_diag') & (result_df.processed_feature == 'five_prime_utr')].sort_values('max_value', ascending=False).iloc[:top_n_to_plot].downstream_gene_id.unique()

    for i, gene_id in enumerate(genes_to_plot):

        print(gene_id)
        plot_matrix(gene_id, coords_df = top_hits_matrix_coords_df, conv_type='anti_diag')

        if not os.path.exists(os.path.join(figures_path, species_of_interest, 'top_five_prime_utr_anti_diagonal_hits')):
                os.makedirs(os.path.join(figures_path, species_of_interest, 'top_five_prime_utr_anti_diagonal_hits'))

        plt.savefig(os.path.join(figures_path,  f'{species_of_interest}/top_five_prime_utr_anti_diagonal_hits/{i+1}_top_{species_of_interest}_{gene_id}_ci_matrix.png'))


        plt.close()

if 'three_prime_utr' in result_df.processed_feature.unique():

    # plot genes with top anti_diag values but that are not tRNAs or rRNAs
    top_n_to_plot = 50
    genes_to_plot = result_df[(result_df.conv_type == 'anti_diag') & (result_df.processed_feature == 'three_prime_utr')].sort_values('max_value', ascending=False).iloc[:top_n_to_plot].downstream_gene_id.unique()

    for i, gene_id in enumerate(genes_to_plot):

        print(gene_id)
        plot_matrix(gene_id, coords_df = top_hits_matrix_coords_df, conv_type='anti_diag')

        if not os.path.exists(os.path.join(figures_path, species_of_interest, 'top_three_prime_utr_anti_diagonal_hits')):
                os.makedirs(os.path.join(figures_path, species_of_interest, 'top_three_prime_utr_anti_diagonal_hits'))

        plt.savefig(os.path.join(figures_path,  f'{species_of_interest}/top_three_prime_utr_anti_diagonal_hits/{i+1}_top_{species_of_interest}_{gene_id}_ci_matrix.png'))


        plt.close()


