import sys
sys.path.append("../mutation_effect_analysis")
home_dir = ""

import os
import numpy as np
import pandas as pd
from models.aa_common.data_loader import get_pmd_dbnsfp_dataset, get_popu_freq_dbnsfp_dataset, get_patho_likelypatho_neutral_dbnsfp_dataset
import models.jax_unirep.model_utils as model_utils
import utils.pickle_utils as pickle_utils
from jax_unirep import get_reps

task = "popu_freq" # pmd, popu_freq, patho
# variants_df, protid_seq_dict = get_pmd_dbnsfp_dataset(home_dir)
variants_df, protid_seq_dict = get_popu_freq_dbnsfp_dataset(home_dir)
# variants_df, protid_seq_dict = get_patho_likelypatho_neutral_dbnsfp_dataset(home_dir)

model_name = "unirep"
model_task_out_dir, model_logits_out_dir = model_utils.create_output_directories(model_name, task, home_dir)


def get_embedding(seq, filename):
    filepath = f"{model_logits_out_dir}{filename}.pkl"

    if os.path.exists(filepath):
        # print(f"Model logits already exists: {filename}")
        embedding = pickle_utils.load_pickle(filepath) 
    else: 
        # print(f"Computing model logits: {filename}")
        h_avg, h_final, c_final = get_reps(seq)
        embedding = h_avg[0] # np-array of shape 1900
        pickle_utils.save_as_pickle(embedding, filepath)
    return embedding


def compute_variant_effect_score(protid, seq, one_indexed_mut_pos, wt_aa, mt_aa):
    wt_seq = seq
    mt_seq = list(wt_seq)
    mt_seq[one_indexed_mut_pos-1] = mt_aa
    mt_seq = "".join(mt_seq)
    # print(wt_seq)
    # print(mt_seq)
    # print(wt_seq==mt_seq)

    wt_filename = f"{protid}"
    mt_filename = f"{protid}_{str(one_indexed_mut_pos)}_{mt_aa}"
    
    wt_embedding = get_embedding(wt_seq, wt_filename)
    mt_embedding = get_embedding(mt_seq, mt_filename)
    print(protid, len(seq), one_indexed_mut_pos, wt_seq[one_indexed_mut_pos-1], wt_aa,  mt_seq[one_indexed_mut_pos-1], mt_aa, wt_embedding.shape, mt_embedding.shape)

    # effect_score = abs(mt_embedding - wt_embedding).sum() / 1900 # embedding_dim = 1900
    effect_score = np.linalg.norm(mt_embedding - wt_embedding)
    # print(effect_score)
    return effect_score


def execute(row):
    index = row[0]
    data = row[1]
    
    protid, seq, one_indexed_mut_pos, wt_aa, mt_aa = data["prot_acc_version"], protid_seq_dict[data["prot_acc_version"]], data["1indexed_prot_mt_pos"], data["wt_aa_1letter"], data["mt_aa_1letter"]
    effect_score = compute_variant_effect_score(protid, seq, one_indexed_mut_pos, wt_aa, mt_aa)
    print(index, protid, one_indexed_mut_pos, wt_aa, mt_aa, effect_score)

    row = variants_df.loc[index]
    row = dict(row)
    row["pred"] = effect_score
    
    return row

def compute_wt_embedding(i, protid_seq_tuple):
    protid, seq = protid_seq_tuple
    get_embedding(seq, f"{protid}")
    print(f"Computing wild-type seq embedding: {i}|{len(protid_seq_dict)}: {protid}")

if __name__ == "__main__":

    # variants_df = variants_df.head(5)
    preds = [] 
    
    # computing the wt-seq embedding
    from mpi4py.futures import MPIPoolExecutor
    executor = MPIPoolExecutor()
    executor.map(compute_wt_embedding, list(range(len(protid_seq_dict))), list(protid_seq_dict.items()), unordered=False)
    executor.shutdown()

    # computing the mt-seq embedding and variant score
    executor = MPIPoolExecutor()
    for i, row in enumerate(executor.map(execute, variants_df.iterrows(), unordered=False)):
        preds.append(row)
    executor.shutdown()

    # for row in variants_df.iterrows():
    #     preds.append(execute(row))

    result_df = pd.DataFrame(preds)   
    print(result_df.shape)
    print(result_df.head())
    result_df.to_csv(f"{model_task_out_dir}/preds_{model_name}_masked.tsv", sep="\t", index=False, header=True)