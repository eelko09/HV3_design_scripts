#!/usr/bin/env python3

import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import kmertools as kt	 # Available from https://github.com/jtladner/Modules
import fastatools as ft # Available from https://github.com/jtladner/Modules
import inout as io		 # Available from https://github.com/jtladner/Modules

import argparse, random, os
from collections import defaultdict

from scipy import sparse
import sknetwork as skn

from dynamicTreeCut import cutreeHybrid

def main():
	parser = argparse.ArgumentParser(description="Generates clusters of similar sequences for each protein", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	# Required arguments
	parser.add_argument("-i", "--input-files", nargs="+", default=[], required=True, help="Fasta files containing sequences to be clustered. One or more fasta files can be provided. Sequences in each will be separately clustered.")
	#parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save cluster files. This directory will be created, if it doesn't already exist.")

	# Optional arguments
	parser.add_argument("--method", type=str, required=False, default="Hierarchical", help="Method of generating clusters. Current options are: Hierarchical or Louvain")
	parser.add_argument('-g', '--genes', type=str, required=False, help='TSV metadata file with sequence name and gene annotation info. [None]')

	parser.add_argument("-d", "--distance-thresh", nargs="+", type=float, required=False, help="Required for Hierarchical Clustering. Distance thresholds to use for hierarchical clustering. Multiple values may be provided, all of which should be between 0 and 1.")
	parser.add_argument("-k", "--kmer-size", type=int, required=False, default=7, help="Size of kmers used to compare sequences.")
	#parser.add_argument("-m", "--meta-filepath", type=str, required=False, help="Optional tab-delimited file that can be used to link the input sequences to metadata. If provided, summary statistics about the generated clusters will be generated.")
	parser.add_argument("-p", "--min-propn", type=float, required=False, default=0, help="Proportion of the top 10%% of sequence sizes to be included in the initial round of clustering.")
	parser.add_argument("--clust-unassigned-short", required=False, default=False, action="store_true", help="If provided, unassigned, short sequences will be clustered using the specified method. Only supported now for hierarchical clustering.")
	parser.add_argument("--make-dist-boxplots", required=False, default=False, action="store_true", help="If provided, distance boxplots will be generated.")
	parser.add_argument("--boxplots-output-dir", type=str, default="boxplots", required=False, help="Directory to save boxplots if boxplots are made.")
	parser.add_argument("--min-seqs", type=int, default=2, required=False, help="Number of sequences a cluster needs to be included in the boxplot.")
	parser.add_argument("--make-sim-hists", required=False, default=False, action="store_true", help="If provided, distribution of k-mer similarities for each input file will be provided.")
	parser.add_argument("--hists-output-dir", type=str, required=False, default="histograms", help="Directory to save histograms if histograms are made.")
	parser.add_argument("--generate-vis", required=False, default=False, action="store_true", help="If provided, generates visualization of clusters for clustering method.")
	parser.add_argument("--vis-output-dir", type=str, required=False, default="visualizations", help="Directory to save visualizations if visualizations are made.")


	args=parser.parse_args()
	
	info = False
	id2info = None
	if args.genes:
		info = True
		id2info = extractGeneAnnotInfoTSV(args.genes)

	cluster(
		# meta_filepath = args.meta_filepath,
		input_files = args.input_files,
		method = args.method,
		distance_thresh = args.distance_thresh,
		kmer_size = args.kmer_size,
		min_propn = args.min_propn,
		make_dist_boxplots = args.make_dist_boxplots,
		boxplots_output_dir = args.boxplots_output_dir,
		min_seqs = args.min_seqs,
		make_sim_hists = args.make_sim_hists,
		hists_output_dir = args.hists_output_dir,
		gen_vis = args.generate_vis,
		vis_output_dir = args.vis_output_dir,
		#output_dir = args.output_dir,
		clust_unassigned_short = args.clust_unassigned_short,
		gene_info = info,
		id2info=id2info)

###---------------End of main()-----------------------------------


def cluster(
	input_files: list,
	method: str,
	distance_thresh: list,
	kmer_size: int,
	min_propn: float,
	make_dist_boxplots: bool,
	boxplots_output_dir: str,
	min_seqs: int,
	make_sim_hists: bool,
	hists_output_dir: str,
	gen_vis: bool,
	vis_output_dir: str,
	#output_dir: str,
	clust_unassigned_short: bool,
	gene_info: bool,
	id2info=None
) -> None:


	for i in input_files:
		# Format output directory name and create output directories
		outName = i.replace('.fasta', '_clusters')
		
		create_directory(outName, f"The directory \"{outName}\" already exists. Files may be overwritten.")
		if make_dist_boxplots or make_sim_hists or gen_vis:
			create_directory(f"{outName}/{vis_output_dir}", f"The directory \"{vis_output_dir}\" already exists. Visualizations may be overwritten.")
			
		outD = {}
		fD = ft.read_fasta_dict(i)	# Read in sequences from fasta file

		if len(fD) == 1:  # Special case for single sequence
			for dt in distance_thresh:
				outD[dt] = [0]
				output_data(outD, i, output_dir, list(fD.keys()))
			continue

		largeD, smallD = sortSeqsBySize(fD, min_propn)	# Sort sequences by size
		kmer_dict = kt.kmerDictSet(largeD, kmer_size, ["X"])  # Generate kmers for large sequences
		dists, seqNames, similarities, seq_sim_list = calcDistances(kmer_dict)	# Calculate distances for large sequences
		kmer_dict_small = kt.kmerDictSet(smallD, kmer_size, ["X"])	# Generate kmers for small sequences
		shortSeqNames = list(smallD.keys())

		# Hierarchical clustering
		if method == "Hierarchical":
			for dt in distance_thresh:
				clusters, hm = clusterSeqs(dists, dt, seqNames)	 # Cluster large sequences

				if gen_vis:	 # Generate dendrogram visualization
					fig, ax = plt.subplots(figsize=(30, 40), facecolor='w')
					ax = sch.dendrogram(hm, color_threshold=dt)
					plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
					plt.axhline(y=dt, c='k')
					plt.savefig(f"./{outName}/{vis_output_dir}/dendrogram_{os.path.basename(i)}_{dt}.png", dpi=300, bbox_inches='tight')
					mpl.pyplot.close()

				seq2clust = assignShortSequences(kmer_dict_small, kmer_dict, clusters, dt) if kmer_dict_small else {}
				for clustNum, seqL in clusters.items():
					for sn in seqL:
						seq2clust[sn] = clustNum

				if clust_unassigned_short:
					kmer_dict_small_unass = {k: v for k, v in kmer_dict_small.items() if seq2clust[k] == ""}
					if kmer_dict_small_unass:
						if len(kmer_dict_small_unass) == 1:
							clusters_su = {"0": list(kmer_dict_small_unass.keys())}
						else:
							dists_su, seqNames_su, similarities_su, seq_sim_list_su = calcDistances(kmer_dict_small_unass)
							clusters_su, hm_su = clusterSeqs(dists_su, dt, seqNames_su)
						baseClustNum = max([int(s) for s in seq2clust.values() if s != ""]) + 1
						for clustNum, seqL in clusters_su.items():
							for sn in seqL:
								seq2clust[sn] = str(int(clustNum) + baseClustNum)

				outD[dt] = [seq2clust[sn] for sn in seqNames + shortSeqNames]
				output_data(outD, i, outName, seqNames + shortSeqNames)
				if gene_info:
					write_cluster_fastas(outName, clustNum, seq2clust, fD, gene_info, id2info)
				else:
					write_cluster_fastas(outName, clustNum, seq2clust, fD, gene_info)
					
						 
						 

		# Louvain clustering
		elif method == "Louvain":
			# no distance threshold for louvain clustering
			dt = 0

			# Extract unique nodes from sequence similarity list
			nodes = list(set([x[0] for x in seq_sim_list] + [x[1] for x in seq_sim_list]))
			node_indices = {node: idx for idx, node in enumerate(nodes)}

			# Create edge list with node indices and similarities
			edges = [(node_indices[a], node_indices[b], sim) for a, b, sim in seq_sim_list]

			# Create adjacency matrix from edge list
			adjacency = skn.data.from_edge_list(edges)

			# Perform Louvain clustering
			louvain = skn.clustering.Louvain()
			labels = louvain.fit_predict(adjacency)

			# Generate visualization if required
			if gen_vis:
				image = skn.visualization.visualize_graph(adjacency, labels=labels, filename=f"{vis_output_dir}/visualization_{os.path.basename(i)}")

			# Group sequences by cluster labels
			clusters = defaultdict(list)
			for idx, seqName in enumerate(nodes):
				clusters[ labels[idx] ].append( seqName )

			# Assign short sequences to clusters if provided
			if len(kmer_dict_small) > 0:
				# Start to assigning the short sequences to the clusters built using long sequences
				seq2clust = assignShortSequences(kmer_dict_small, kmer_dict, clusters, dt)
			else:
				# Make a dictionary linking a sequence name to its assigned cluster
				seq2clust = {}

			# Map sequence names to their assigned clusters
			for clustNum, seqL in clusters.items():
				for sn in seqL:
					seq2clust[sn] = clustNum

			# Add cluster numbers to output dictionary in order of seqNames
			outD[dt] = [seq2clust[sn] for sn in seqNames + shortSeqNames]

			# Output data to specified directory
			output_data( outD, i, output_dir, seqNames + shortSeqNames )
			
			# Write cluster FASTA files
			for clustNum in clusters:
				outName = f"{output_dir}/cluster_{clustNum}.fasta"
				if gene_info:
					write_cluster_fastas(outName, clustNum, seq2clust, fD, gene_info, id2info)
				else:
					write_cluster_fastas(outName, clustNum, seq2clust, fD, gene_info)
	

		else:
			raise Exception("Invalid method name provided.")

		if make_dist_boxplots:
			make_boxplots(f"./{outName}/{vis_output_dir}/boxplots_{os.path.basename(i)}", dt, clusters, kmer_dict, outD, boxplots_output_dir, min_seqs)

		if make_sim_hists:
			make_hist(f"./{outName}/{vis_output_dir}/hist_{os.path.basename(i)}", similarities, hists_output_dir)



			# This is an example of calculating some summary statistics to help us better understand the clusters
			# We will likely want to expand this functionality in the future
# Want to make this section optional 
#			numClusts = len(clusters)
#			multi, initialSpecies, multiClustSpecies = clustersBySpecies(clusters, speciesD)
#			print(f"{dt}\t{numClusts}\t{multi}\t{multi/numClusts:.4f}\t{len(multiClustSpecies)}\t{len(multiClustSpecies)/initialSpecies:.4f}")

# For each short sequence, calculate the smallest distance between this sequence and the clusters of long sequences
	# Based on these distances, assign the short sequences to clusters
def assignShortSequences(shortKD, longKD, clusters, distThresh):
	clustAssign = {}
	unassigned = []
	assigned = []
	
	# Initial comparison of short sequences to clusters of long sequences
	for ssN, ssK in shortKD.items():
		distD = {}
		for clustNum, seqL in clusters.items():
			theseDists = []
			for lsN in seqL:
				lsK = longKD[lsN]
				ovlp = ssK.intersection(lsK)
				theseDists.append( 1-( len(ovlp)/(min([len(ssK), len(lsK)])) ) )
			distD[clustNum] = min(theseDists)
		
		bestScore = min(distD.values())
		if bestScore <= distThresh:
			bestClusts = [k for k,v in distD.items() if v==bestScore]
			clustAssign[ssN] = random.choice(bestClusts)
			assigned.append(ssN)
		else:
			unassigned.append(ssN)
	
	# Check remaining short sequences for matches to the short sequences already assigned to clusters
	
	while len(unassigned)>0 and len(assigned)>0:
		assigned=[]
		for ssN in unassigned:
			best = [1,""]
			ssK = shortKD[ssN]
			for otherN, clustNum in clustAssign.items():
				otherK = shortKD[otherN]
				ovlp = ssK.intersection(otherK)
				thisDist = 1-( len(ovlp)/(min([len(ssK), len(otherK)])) )
				if thisDist<best[0]:
					best=[thisDist,clustNum]
			if best[0] <= distThresh:
				clustAssign[ssN] = best[1]
				assigned.append(ssN)
		
		# Remove any names from unassigned that have now been assigned
		for name in assigned:
			unassigned.remove(name)

	#Assign unassigned sequences to their own clusters
	for name in unassigned:
		clustAssign[name] = ""
	
	return clustAssign
		
def sortSeqsBySize(fastaDict, min_propn, topPerc=0.1):
	# Sort the sequence lengths in ascending order
	lenSortedL = sorted([len(seq) for seq in fastaDict.values()])
	# Determine the number of sequences needed to contain 10% of the total
	numForTopPerc = round(len(fastaDict)*topPerc)
	# Calculate the average size for the top specified percentage of sequences
	topAvg = np.mean(lenSortedL[-numForTopPerc:])
	# Calculate threshold for inclusion in the initial round of clustering
	thresh = topAvg*min_propn
	
	# Generate a dictionary to contain the longer sequences to be used in initial round of clustering
	largeD={}
	# Generate a dictionary to contain the shorter sequences that will not be used in initial round of clustering
	smallD={}
	for k,v in fastaDict.items():
		if len(v)>=thresh:
			largeD[k] = v
		else:
			smallD[k] = v
	
	return largeD, smallD

# within the same kmer dictionary
def calcDistances(kD):
	seqNames = list(kD.keys())

	# Currently, one distance measure is implemented
	# But, in the future, it would be nice to support multiple distance options
	dists = []
	similarities = []
	seq_sim_list = []
	for i, ni in enumerate(seqNames):
		ki = kD[seqNames[i]]
		for j, nj in enumerate(seqNames):
			if j>i:
				kj = kD[seqNames[j]]
				ovlp = ki.intersection(kj)
				similarity = len(ovlp)/(min([len(ki), len(kj)]))
				similarities.append(similarity)
				dists.append(1-similarity)

				seq_sim_list.append((ni, nj, similarity))

	return dists, seqNames, similarities, seq_sim_list

#TODO: account for similarities
def calcDistancesBetweenClusters(kmer_dict1, kmer_dict2):
	seqNames1 = list(kmer_dict1.keys())
	seqNames2 = list(kmer_dict2.keys())

	dists = []
	for i, ni in enumerate(seqNames1):
		ki = kmer_dict1[seqNames1[i]]
		for j, nj in enumerate(seqNames2):
			kj = kmer_dict2[seqNames2[j]]
			ovlp = ki.intersection(kj)
			dists.append(1-(len(ovlp)/(min([len(ki), len(kj)]))))
	return dists


def clusterSeqs(dists, distThresh, seqNames, linkage_meth='average'):
	hm = sch.linkage(np.array(dists), method=linkage_meth)
	groups = sch.cut_tree(hm,height=distThresh)
	#clusters = cutreeHybrid(hm, np.array(dists), cutHeight=distThresh, minClusterSize = 1)
	#groups = [[x] for x in clusters["labels"]]

	gD=defaultdict(list)
	for i,g in enumerate(groups):
		gD[g[0]].append(seqNames[i])
	return gD, hm

def clustersBySpecies(clusters, speciesD):
	clusterCountD = defaultdict(int)
	multi=0

	for grpNum, memList in clusters.items():
		spL = [speciesD[m] for m in memList]
		spS = set(spL)
		if len(spS)>1:
			multi+=1
#			print(spS)
		for each in spS:
			clusterCountD[each]+=1
	
	return multi, len(clusterCountD),{k:v for k,v in clusterCountD.items() if v>1}

# Generate some summary statistics about the generated clusters
def clusterStats(outD, inName, output_dir):
	with open(f"{output_dir}/{inName}_summaryStats.tsv", "w") as fstats:
		fstats.write("Distance_Threshold\tNumberOfClusters\tAvgSeqsPerCluster\tSeqsUnassigned\n")
		for dt, clustL in outD.items():
			uniq = set(clustL).difference({""})
			numClust = len(uniq)
			numUnassigned = clustL.count("")
			perClust=np.mean([clustL.count(x) for x in uniq])
			fstats.write(f"{dt:.3f}\t{numClust}\t{perClust:.1f}\t{numUnassigned}\n")

def make_boxplots(input_filename, dist_thresh, clusters, kmer_dict, outD, boxplots_output_dir, min_seqs):
	clustDistWithin = list()
	clustDistBetween = list()

	for clustNum1, clustSeqNames1 in clusters.items():
		if len(clustSeqNames1) >= min_seqs:
			kmerSubDict1 = {seq:kmer_dict[seq] for seq in clustSeqNames1}

			clustDistsWithin = calcDistances(kmerSubDict1)[0]

			for dist in clustDistsWithin:
				clustDistWithin.append((dist, clustNum1, "Within"))

			for clustNum2, clustSeqNames2 in clusters.items():
				if clustNum1 != clustNum2:
					kmerSubDict2 = {seq:kmer_dict[seq] for seq in clustSeqNames2}

					clustDistsBetween = calcDistancesBetweenClusters(kmerSubDict1, kmerSubDict2)

					for dist in clustDistsBetween:
						clustDistBetween.append((dist, clustNum1, "Between"))

	withinDf = pd.DataFrame(clustDistWithin, columns = ["Distance", "Cluster", "Type"])
	betweenDf=pd.DataFrame(clustDistBetween, columns = ["Distance", "Cluster", "Type"])

	clustDistDf = pd.concat([withinDf, betweenDf])

	width = max(outD[dist_thresh])
	height = 20
	fontsize = width * 5/4
	fig, ax = plt.subplots(figsize=(width, height), facecolor='w')
	sns.boxplot(x="Cluster", y="Distance", hue="Type", data=clustDistDf, ax=ax)
	ax.set_xlabel("Cluster", fontsize=fontsize)
	ax.set_ylabel("Distance", fontsize=fontsize)
	plt.grid()
	plt.savefig(f"{input_filename}_{dist_thresh}_boxplot.png", dpi=300, bbox_inches='tight')
	mpl.pyplot.close()

def make_hist(input_filename, similarities, output_dir):
	fig, ax = plt.subplots(figsize=(11,8.5), facecolor='w')
	ax.hist(similarities, bins=100)
	plt.xticks(np.arange(0,1, step=0.1))
	plt.xlabel("similarity")
	plt.ylabel("count")
	plt.savefig(f"{input_filename}_histogram.png")
	mpl.pyplot.close()
	
def extractGeneAnnotInfoTSV(file):
	id2info = io.fileDictHeader(file,'SequenceName','Gene')
	return(id2info)


def output_data( outD, i, output_dir, allSeqNames ):
	# Write out results for this input file
	outDF = pd.DataFrame(outD, index=allSeqNames)
	outDF.to_csv(f"{output_dir}/clusters_{os.path.basename(i)}.tsv", sep="\t", index_label="Sequence")
	
	# Summary statistics
	clusterStats(outD, os.path.basename(i), output_dir)
	
def create_directory(path, warning_message):
	"""Creates a directory if it doesn't exist, and prints a warning if it does."""
	if not os.path.exists(path):
		os.mkdir(path)
	else:
		print(f"Warning: {warning_message}")


def write_cluster_fastas(outName, clustNum, seq2clust, fD, gene_info, id2info=None):
	"""
	Writes out cluster FASTA files and cluster information.

	Parameters:
	outName (str): The base name for the output directory.
	clustNum (int): The number of clusters.
	seq2clust (dict): Dictionary mapping sequence names to cluster numbers.
	fD (dict): Dictionary of sequence data.
	gene_info (bool): Flag to include gene information.
	id2info (dict, optional): Dictionary mapping sequence IDs to gene information.
	"""
	#os.mkdir(outName)

	with open(f"{outName}/{outName}_clusterinfo.txt", "w") as fout:
		fout.write("Cluster#\tGeneName\tSeqCount\n")
		newN_list = []
		for i in range(clustNum + 1):
			newN = [key for key, value in seq2clust.items() if value == i]
			newN_list.append(newN)
			newC = [value for key, value in fD.items() if key in newN]
			ft.write_fasta(newN, newC, f"{outName}/{outName}_cluster{i:03d}.fasta")
		for i, each in enumerate(newN_list):
			if gene_info and id2info:
				gs = [id2info[x] for x in each if "PV2" not in x]
				for gene in set(gs):
					fout.write(f"{i:03d}\t{gene}\t{gs.count(gene)}\n")
			else:
				fout.write(f"{i:03d}\t\t{len(each)}\n")


if __name__ == "__main__":
	main()