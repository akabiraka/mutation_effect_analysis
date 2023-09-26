import sys

sys.path.append("../mutation_effect_analysis")
home_dir = ""

import time
import pandas as pd


import models.bioembeddings_dallago.model_utils as model_utils

dms_filename = "PTEN_HUMAN_Matreyek_2021"
protid = dms_filename

variants_df = pd.read_csv(f"data/dms/ProteinGym_substitutions/{protid}.csv", sep=",")
print(variants_df.shape)
variants_df["1indexed_prot_mt_pos"] = variants_df["mutant"].apply(
    lambda mut: int(mut[1:-1])
)
variants_df["wt_aa_1letter"] = variants_df["mutant"].apply(lambda mut: mut[0])
variants_df["mt_aa_1letter"] = variants_df["mutant"].apply(lambda mut: mut[-1])
variants_df["prot_acc_version"] = protid
# print(df.tail())

# seq extraction
first_row = variants_df.loc[0]
mutant = first_row["mutant"]
from_aa, one_indexed_mut_pos, to_aa = mutant[0], int(mutant[1:-1]), mutant[-1]
zero_indexed_mut_pos = one_indexed_mut_pos - 1
first_mt_seq = first_row["mutated_sequence"]
wt_seq = (
    first_mt_seq[:zero_indexed_mut_pos]
    + from_aa
    + first_mt_seq[zero_indexed_mut_pos + 1 :]
)
protid_seq_dict = {protid: wt_seq}
# print(len(seq), seq)

# loading model and tokenizer
task = "dms"
model_name = "plus_rnn"  # tested for: plus_rnn
model, tokenizer, model_name = model_utils.get_model_tokenizer(model_name)
model_task_out_dir, model_logits_out_dir = model_utils.create_output_directories(
    model_name, task
)

import os
import torch
import numpy as np
import utils.pickle_utils as pickle_utils


def get_embedding(seq, prot_acc_version):
    filepath = f"{model_logits_out_dir}{prot_acc_version}.pkl"

    if os.path.exists(filepath):
        print(f"Model embedding already exists: {prot_acc_version}")
        embedding = pickle_utils.load_pickle(filepath)
    else:
        print(f"Computing model embedding: {prot_acc_version}")
        with torch.no_grad():
            embedding = model.embed(seq)
            pickle_utils.save_as_pickle(embedding, filepath)
    print(embedding.shape)
    return embedding


def compute_wt_embedding(i, protid_seq_tuple):
    protid, seq = protid_seq_tuple
    get_embedding(seq, f"{protid}")
    print(f"Computing wild-type seq embedding: {i}|{len(protid_seq_dict)}: {protid}")


def compute_variant_effect_score(
    protid, mutation, wt_seq, mt_seq, zero_indexed_mut_pos
):
    wt_filename = f"{protid}"
    mt_filename = f"{protid}_{mutation}"

    wt_embedding = get_embedding(wt_seq, wt_filename)
    mt_embedding = get_embedding(mt_seq, mt_filename)
    print(
        protid,
        len(wt_seq),
        mutation,
        wt_embedding.shape,  # seq_len, embed_dim
        mt_embedding.shape,
    )
    wt_embedding = wt_embedding[zero_indexed_mut_pos, :]
    mt_embedding = mt_embedding[zero_indexed_mut_pos, :]
    print(wt_embedding.shape, mt_embedding.shape)  # seq_len, embed_dim

    effect_score = np.linalg.norm(mt_embedding - wt_embedding)
    raise
    # print(effect_score)
    return effect_score


def execute(row):
    index = row[0]
    data = row[1]

    mutation, mt_seq, one_indexed_mut_pos = (
        data["mutant"],
        data["mutated_sequence"],
        data["1indexed_prot_mt_pos"],
    )
    zero_indexed_mut_pos = one_indexed_mut_pos - 1
    effect_score = compute_variant_effect_score(
        protid, mutation, wt_seq, mt_seq, zero_indexed_mut_pos
    )
    print(index, protid, one_indexed_mut_pos, mutation, effect_score)

    row = variants_df.loc[index]
    row = dict(row)
    row["pred"] = effect_score

    return row


if __name__ == "__main__":
    start = time.time()

    variants_df = variants_df.head(5)
    preds = []

    # computing the wt-seq embedding
    from mpi4py.futures import MPIPoolExecutor

    # computing wt-seqs embeddings
    executor = MPIPoolExecutor()
    executor.map(
        compute_wt_embedding,
        list(range(len(protid_seq_dict))),
        list(protid_seq_dict.items()),
        unordered=False,
    )
    executor.shutdown()

    # computing the mt-seq embedding and variant score
    executor = MPIPoolExecutor()
    for i, row in enumerate(
        executor.map(execute, variants_df.iterrows(), unordered=False)
    ):
        preds.append(row)
    executor.shutdown()

    result_df = pd.concat(preds)
    result_df.drop(
        columns=[
            "1indexed_prot_mt_pos",
            "wt_aa_1letter",
            "mt_aa_1letter",
            "prot_acc_version",
        ],
        inplace=True,
    )
    print("Saving predictions ...")
    result_df.to_csv(
        f"{model_task_out_dir}/preds_{protid}.tsv",
        sep="\t",
        index=False,
        header=True,
    )
    print(variants_df.shape, result_df.shape)
    print(result_df.head())

    print(f"total time: {(time.time() - start)/60} minutes")
