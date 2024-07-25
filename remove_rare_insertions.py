#!/usr/bin/env python

# Written by Evan Elko

from __future__ import division
import optparse, re, os, itertools
import fastatools as ft
#For subrpocesses
from subprocess import Popen, PIPE

#This script removes any positions in an alignment that contain "rare" insertions ('-' in more than X proportion of the sequences). 

def main():
    #To parse command line
    usage = '%prog [options] fasta1 [fasta2 ...]'
    p = optparse.OptionParser(usage)
    #Inputs and Outputs
    p.add_option('-f', '--fasta',  help='Aligned fasta with ambiguous characters. [None]')
    p.add_option('-t', '--thresh', default=0.02,  help='Threshold for proportion of seqs contain a - at a given position for insertion to be removed. [0.02]')

    opts, inputs = p.parse_args()
    print(opts)
    print(inputs)
    
    if not opts.fasta:
        p.error('No FASTA file(s) provided.')

    thresh = opts.thresh
    
    # Process each input file
    input_files = [opts.fasta] + inputs
    for fasta_file in input_files:
        print(f"Processing file: {fasta_file}")
        names, seqs = ft.read_fasta_lists(fasta_file)

        # Check that all sequences have the same length
        seq_lengths = set(len(seq) for seq in seqs)
        if len(seq_lengths) > 1:
            print(f"Error: Sequences in {fasta_file} have varying lengths: {seq_lengths}")
            continue

        new_seqs = replace_gaps(seqs, opts.thresh)
        output_file = f"nogaps_{os.path.basename(fasta_file)}"
        ft.write_fasta(names, new_seqs, output_file)
        print(f"Output written to: {output_file}")
        
#End of main------------------------------------------

    
def replace_gaps(seqs, thresh):
    """
    Replace gaps in sequences based on a given threshold.
    
    Parameters:
    seqs (list of str): List of sequences.
    thresh (float): Threshold for determining rare inserts.
    
    Returns:
    list of str: Sequences with gaps removed based on the threshold.
    """
    # Make a copy of the input sequences
    new_seqs = seqs[:]
    seq_count = len(seqs)
    gap_indices = []
    
    # Iterate through each base index in the sequences
    for base_index in range(len(seqs[0])):
        # Get the list of bases at the current index for all sequences
        base_list = get_base_list(base_index, seqs)
        
        # Check if the gap at the current index is rare based on the threshold
        if is_rare_gap(base_list, seq_count, thresh):
            gap_indices.append(base_index)
    
    # If there are any gap indices, remove the gaps from the sequences
    if gap_indices:
        for each in gap_indices[::-1]:  # Reverse order to avoid index shifting issues
            new_seqs = [seq[:each] + seq[each+1:] for seq in new_seqs]
    
    return new_seqs

def is_rare_gap(base_list, seq_count, thresh):
    """
    Check if the gap is rare based on the threshold.
    
    Parameters:
    base_list (list of str): List of bases at a specific index.
    seq_count (int): Total number of sequences.
    thresh (float): Threshold for determining rare inserts.
    
    Returns:
    bool: True if the gap is rare, False otherwise.
    """
    gap_count = base_list.count('-')
    nongap_count = seq_count - gap_count
    
    # Check if the proportion of non-gap characters is less than or equal to the threshold
    return nongap_count / seq_count <= float(thresh)

def get_base_list(index, seqs):
    """
    Get the list of bases at a specific index for all sequences.
    
    Parameters:
    index (int): Index of the base in the sequences.
    seqs (list of str): List of sequences.
    
    Returns:
    list of str: List of bases at the specified index.
    """
    return [seq[index].upper() for seq in seqs]


###-------------->>>

if __name__ == '__main__':
    main()    



