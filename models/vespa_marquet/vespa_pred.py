import sys
sys.path.append("../mutation_effect_analysis")
home_dir = ""

import  pandas as pd

from models.aa_common.data_loader import get_pmd_dbnsfp_dataset, get_popu_freq_dbnsfp_dataset, get_patho_likelypatho_neutral_dbnsfp_dataset
from models.vespa_marquet.vespa_model_utils import create_output_directories, get_predictions

model_name = "vespa"
task = "patho" # pmd, popu_freq, patho
# variants_df, protid_seq_dict = get_pmd_dbnsfp_dataset(home_dir)
# variants_df, protid_seq_dict = get_popu_freq_dbnsfp_dataset(home_dir)
variants_df, protid_seq_dict = get_patho_likelypatho_neutral_dbnsfp_dataset(home_dir)

def compute_mutation(row):
    return row["wt_aa_1letter"]+str(row["1indexed_prot_mt_pos"])+row["mt_aa_1letter"]
if "mut_real" not in variants_df.columns.to_list():
    variants_df["mut_real"] = variants_df.apply(compute_mutation, axis=1)
# print(variants_df[["wt", "prot_pos", "mut", "mut_real"]])

model_task_out_dir, model_preds_out_dir = create_output_directories(model_name, task)

def execute(protids_list):
    preds = []
    for protid in protids_list:
        seq = protid_seq_dict[protid]
        mutations = variants_df[variants_df["prot_acc_version"]==protid]["mut_real"].unique().tolist()
        # mutations = ['T164I', 'V297M', 'I201V', 'I201A']
        preds_related_to_aprot = get_predictions(protid, seq, mutations, variants_df, model_preds_out_dir)
        preds += preds_related_to_aprot
    preds_df = pd.DataFrame(preds)   
    return preds_df

if __name__ == "__main__":

    data = variants_df["prot_acc_version"].unique().tolist() # protids_list

    chunk_size = 1 # 32 if torch.cuda.is_available() else 1
    data_chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]
    # data_chunks = data_chunks[:10] 
    print(f"#-of chunks: {len(data_chunks)}, 1st chunk size: {len(data_chunks[0])}")

    pred_dfs = []
    # sequential run and debugging
    # for i, data_chunk in enumerate(data_chunks):
    #     pred_df = execute(data_chunk)
    #     print(f"Finished {i}/{len(data_chunks)}th chunk: {pred_df.shape}")
    #     pred_dfs.append(pred_df)

    # mpi run    
    from mpi4py.futures import MPIPoolExecutor
    executor = MPIPoolExecutor()
    for i, pred_df in enumerate(executor.map(execute, data_chunks, unordered=True)):
        print(f"Finished {i}/{len(data_chunks)}th chunk: {pred_df.shape}")
        pred_dfs.append(pred_df)
    executor.shutdown()

    result_df = pd.concat(pred_dfs)  
    print("Saving predictions ...")
    result_df.to_csv(f"{model_task_out_dir}/preds_{model_name}_masked.tsv", sep="\t", index=False, header=True)
    print(result_df.shape)
    print(result_df.head())