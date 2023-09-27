import sys

sys.path.append("../mutation_effect_analysis")
home_dir = ""

import os
import time
import numpy as np
import pandas as pd
import models.jax_unirep.model_utils as model_utils
import utils.pickle_utils as pickle_utils
from jax_unirep import get_reps

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
mutated_seq = first_row["mutated_sequence"]
seq = (
    mutated_seq[:zero_indexed_mut_pos]
    + from_aa
    + mutated_seq[zero_indexed_mut_pos + 1 :]
)
# print(len(seq), seq)
protid_seq_dict = {protid: seq}

# loading model and tokenizer
task = "dms"
model_name = "unirep"
model_task_out_dir, model_logits_out_dir = model_utils.create_output_directories(
    model_name, task, home_dir
)


def get_embedding(seq, filename):
    filepath = f"{model_logits_out_dir}{filename}.pkl"

    if os.path.exists(filepath):
        # print(f"Model logits already exists: {filename}")
        embedding = pickle_utils.load_pickle(filepath)
    else:
        # print(f"Computing model logits: {filename}")
        h_avg, h_final, c_final = get_reps(seq)
        embedding = h_avg[0]  # np-array of shape 1900
        pickle_utils.save_as_pickle(embedding, filepath)
    return embedding


def compute_wt_embedding(i, protid_seq_tuple):
    protid, seq = protid_seq_tuple
    get_embedding(seq, f"{protid}")
    print(f"Computing wild-type seq embedding: {i}|{len(protid_seq_dict)}: {protid}")


def compute_variant_effect_score(protid, wt_seq, mt_seq, mutation):
    wt_filename = f"{protid}"
    mt_filename = f"{protid}_{mutation}"

    wt_embedding = get_embedding(wt_seq, wt_filename)
    mt_embedding = get_embedding(mt_seq, mt_filename)
    print(
        protid,
        len(wt_seq),
        mutation,
        wt_embedding.shape,
        mt_embedding.shape,
    )

    # effect_score = abs(mt_embedding - wt_embedding).sum() / 1900 # embedding_dim = 1900
    effect_score = np.linalg.norm(mt_embedding - wt_embedding)
    # print(effect_score)
    return effect_score


def execute(row):
    index = row[0]
    data = row[1]

    mutation, mt_seq = data["mutant"], data["mutated_sequence"]
    effect_score = compute_variant_effect_score(protid, seq, mt_seq, mutation)
    print(index, protid, mutation, effect_score)

    row = variants_df.loc[index]
    row = dict(row)
    row["pred"] = effect_score

    return row


if __name__ == "__main__":
    start = time.time()
    # variants_df = variants_df.head(5)
    preds = []

    # computing the wt-seq embedding
    from mpi4py.futures import MPIPoolExecutor

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

    # for row in variants_df.iterrows():
    #     preds.append(execute(row))

    result_df = pd.DataFrame(preds)
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
