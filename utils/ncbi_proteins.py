import os
import sys
sys.path.append("../mutation_effect_analysis")

import time
import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from urllib.error import HTTPError


Entrez.email = "akabir0101@gmail.com"
Entrez.api_key = "328570309ccd040632796143ec88b51bcf08"
retmax = 500

def download_a_protein_seq(prot_acc_version, idx=0, home_dir=""):
    out_filepath = home_dir+f"data/np_proteins/fastas/{prot_acc_version}.fasta"
    if os.path.exists(out_filepath): 
        print(idx, prot_acc_version, "Already existis")
        return
    else:
        print(idx, prot_acc_version, "Downloading ...")
        
    attempt = 0
    while (attempt < 3):
        attempt += 1
        try:
            fetch_handle = Entrez.efetch(db="protein", id=prot_acc_version, rettype="fasta", retmode="text")
            # print(handle.read())
        except HTTPError as err:
            if 500 <= err.code <= 599:
                print("Received error from server %s" % err)
                print("Attempt %i of 3" % attempt)
                time.sleep(10)
            else:
                raise
    if fetch_handle:
        f_out_handle = open(out_filepath, 'w')
        seq_record = SeqIO.read(fetch_handle, "fasta")
        # print(record)
        # print(record.id)
        # print(str(record.seq))
        SeqIO.write(seq_record, f_out_handle, "fasta")
        
        fetch_handle.close()
        f_out_handle.close()

# download_protein_seq_and_save(0, prot_acc="NP_003673.3")


def download_protein_list(protein_acc_list, start_i=0, home_dir=""):
    # print(protein_acc_list)
    for i, prot_acc in enumerate(protein_acc_list):
        if i<start_i: continue
        download_a_protein_seq(prot_acc, i, home_dir)
        # if i==20: break    




def download_protein_list_mpi(protein_acc_list:list, n_split=1):
    from mpi4py.futures import MPIPoolExecutor
    
    # n_split should be close to length of protein_acc_list
    # but if length of protein_acc_list is very large, it should be divided into sublists
    list_of_protein_acc_list = np.array_split(protein_acc_list, n_split)
    
    executor = MPIPoolExecutor()
    # executor.map(download_prot_list, [protein_acc_list])
    executor.map(download_protein_list, list_of_protein_acc_list)
    # executor.map(download_protein_seq_and_save, protein_acc_list)
    
    executor.shutdown()
    
    
def create_combined_fasta(protein_acc_list, out_filepath, home_dir=""):
    import fileinput
    # the main input of this model is a fasta formatted sequences file. So the input file must only contain the relevant sequences.
    # out_filepath = home_dir+"models/aa_common/datasets_population_freq/SNVs_with_popu_freq.fasta"
    
    download_protein_list(protein_acc_list, start_i=0, home_dir=home_dir) # sequential downloading
    # download_protein_list_mpi(protein_acc_list, len(protein_acc_list)) # download proteins
    print("#-unique NCBI protein sequences downloaded: ", len(protein_acc_list))

    print("Creating merged fasta document.")
    file_list = [home_dir+f"data/np_proteins/fastas/{prot}.fasta" for prot in protein_acc_list]
    with open(out_filepath, 'w') as file:
        input_lines = fileinput.input(file_list)
        file.writelines(input_lines)

# variants_df = pd.read_csv("models/aa_common/datasets_population_freq/proteomic_SNVs.txt", sep="\t")                
# protein_acc_list = list(variants_df["prot_acc_version"].unique())
# create_fasta(protein_acc_list)   



# the following exact line of commands works
# salloc --partition=normal --mem=2G --ntasks=51
# module load gnu10 #gnu10/10.3.0-ya
# module load openmpi # openmpi/4.1.2-4a    
# source /projects/ashehu/akabir4/venvs/hopper_variant_effect_analysis_mine/bin/activate
# mpirun -np x python -m mpi4py.futures mpi/download_proteins.py
# x must be <=41