import sys
sys.path.append("../mutation_effect_analysis")

import pandas as pd
from Bio import SeqIO
# use only very common libraries here


def get_protein_sequences(fasta_filepath, return_type=None):
    """
    seq_return_type: protid_seq_dict, seq_record_list
    """
    fasta_iterator = SeqIO.parse(fasta_filepath, format="fasta")

    if return_type == "seq_record_list":
        data = [seq_record for seq_record in fasta_iterator]
    else: # protid_seq_dict
        data = {seq_record.id: str(seq_record.seq) for seq_record in fasta_iterator}

    return data
# get_protein_sequences()



# ------------------------------the next 3 functions are for loading base 3 datasets-----------------------------
def get_popu_freq_SNVs(home_dir=""):
    print("\nLog: Loading data ...")
    variants_df = pd.read_csv(home_dir+"data/temp_popufreq_dataset/cls_labeled_variants_short.tsv", sep="\t")
    print(variants_df.columns)
    print("total: ", variants_df.shape)
    print("#-unique_rsids:", variants_df["snp_id"].unique().shape[0])
    print("#-unique_genes:",variants_df["gene_name"].unique().shape[0])
    print("#-unique_refseq_prots:",variants_df["mane_refseq_prot"].unique().shape[0])
    print(variants_df["class"].value_counts())

    return variants_df
# get_population_freq_SNVs()#force=True)

def get_patho_and_likelypatho_SNVs(home_dir=""):
    filepath = home_dir+f"models/aa_common/datasets_patho/patho_and_likelypatho.tsv"

    variants_df = pd.read_csv(filepath, sep="\t")
    variants_df.drop_duplicates(keep="first", inplace=True, ignore_index=True)
    print(variants_df.columns)
    print(variants_df["class"].value_counts())
    print("total: ", variants_df.shape)
    
    return variants_df
# get_patho_and_likelypatho_SNVs()


def get_pmd_dataset(home_dir=""):
    print("\nLog: Loading Protein Mutation Dataset (PMD) ...")
    pmd_df = pd.read_csv(home_dir+"models/aa_common/datasets_pmd/pmd.tsv", sep="\t") # PMD: protein mutation dataset
    pmd_df.drop_duplicates(keep="first", inplace=True, ignore_index=True)
    
    print(pmd_df.columns)
    print(pmd_df["functional_effect"].value_counts())
    print(pmd_df.shape)
    
    return pmd_df
# get_pmd_dataset()


# ----------------------------------------the following 3 data-loader we are going to use for model running------------------------
def get_pmd_dbnsfp_dataset(home_dir="", seq_return_type=None):
    filepath = home_dir+f"models/aa_common/datasets_pmd/pmd_dbnsfp"
    df = pd.read_csv(filepath+".tsv", sep="\t")
    seq_data = get_protein_sequences(fasta_filepath=filepath+".fasta", return_type=seq_return_type)

    print(df.columns)
    print(df.shape)
    print(df["class"].value_counts())
    print("#-unique prots: ", len(seq_data))
    return df, seq_data
# get_pmd_dbnsfp_dataset()

def get_patho_likelypatho_neutral_dbnsfp_dataset(home_dir="", seq_return_type=None):
    filepath = home_dir+f"models/aa_common/datasets_patho/patho_likelypatho_neutral_dbnsfp"
    df = pd.read_csv(filepath+".tsv", sep="\t")
    seq_data = get_protein_sequences(fasta_filepath=filepath+".fasta", return_type=seq_return_type)

    print(df.columns)
    print(df.shape)
    print(df["class"].value_counts())
    print("#-unique prots: ", len(seq_data))
    return df, seq_data
# get_patho_likelypatho_neutral_dbnsfp_dataset()

def get_popu_freq_dbnsfp_dataset(home_dir="", seq_return_type=None):
    filepath = home_dir+f"data/datasets_popu_freq/popu_freq_with_dbnsfp_sampled"
    df = pd.read_csv(filepath+".tsv", sep="\t")
    seq_data = get_protein_sequences(fasta_filepath=filepath+".fasta", return_type=seq_return_type)

    print(df.columns)
    print(df.shape)
    print("#-rsids:", df["snp_id"].unique().shape[0])
    print("#-genes", df["gene_name"].unique().shape[0])
    print("#-refseq-prots:", df["mane_refseq_prot"].unique().shape[0])
    print("#-seqs:", len(seq_data))

    print(df["class"].value_counts())
    return df, seq_data
# get_popu_freq_dbnsfp_dataset()

# ---------------------------- loading merged and result analysis things------------------------------
def get_merged_scores_df(task, home_dir=""):
    result_df = pd.read_csv(home_dir+f"models/aa_common/merged_predictions/{task}.tsv", sep="\t")
    print(result_df.columns)
    print(result_df.shape)
    print(result_df["class"].value_counts())
    return result_df