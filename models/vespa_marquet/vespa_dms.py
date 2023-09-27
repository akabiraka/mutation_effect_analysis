import sys

sys.path.append("../mutation_effect_analysis")
home_dir = ""

import time
import numpy as np
import pandas as pd
from models.vespa_marquet.vespa_model_utils import (
    create_output_directories,
    get_predictions,
)

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

variants_df["mut_real"] = variants_df["mutant"]
protids_list = [protid]

# loading model and tokenizer
task = "dms"
model_name = "vespa"
model_task_out_dir, model_preds_out_dir = create_output_directories(model_name, task)


def execute(protids_list):
    preds = []
    for protid in protids_list:
        mutations = (
            variants_df[variants_df["prot_acc_version"] == protid]["mut_real"]
            .unique()
            .tolist()
        )
        # mutations = ['T164I', 'V297M', 'I201V', 'I201A']
        preds_related_to_aprot = get_predictions(
            protid, seq, mutations, variants_df, model_preds_out_dir, task
        )
        preds += preds_related_to_aprot
    preds_df = pd.DataFrame(preds)
    return preds_df


if __name__ == "__main__":
    start = time.time()
    data = protids_list

    chunk_size = 1  # 32 if torch.cuda.is_available() else 1
    data_chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    # data_chunks = data_chunks[:20]
    print(f"#-of chunks: {len(data_chunks)}, 1st chunk size: {len(data_chunks[0])}")

    pred_dfs = []
    # sequential run and debugging
    for i, data_chunk in enumerate(data_chunks):
        pred_df = execute(data_chunk)
        print(f"Finished {i}/{len(data_chunks)}th chunk: {pred_df.shape}")
        pred_dfs.append(pred_df)

    # mpi run
    # from mpi4py.futures import MPIPoolExecutor

    # executor = MPIPoolExecutor()
    # for i, pred_df in enumerate(executor.map(execute, data_chunks, unordered=True)):
    #     print(f"Finished {i}/{len(data_chunks)}th chunk: {pred_df.shape}")
    #     pred_dfs.append(pred_df)
    # executor.shutdown()

    result_df = pd.concat(pred_dfs)
    result_df.drop(
        columns=[
            "1indexed_prot_mt_pos",
            "wt_aa_1letter",
            "mt_aa_1letter",
            "prot_acc_version",
            "mut_real",
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
