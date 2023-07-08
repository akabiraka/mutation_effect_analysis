import sys
sys.path.append("../variant_effect_analysis")
home_dir = ""

import time
import pandas as pd

from models.aa_common.data_loader import get_patho_and_likelypatho_SNVs, get_protein_sequences
import models.sequnet_dunham.model_utils as model_utils

task = "patho_and_likelypatho"
patho_and_likelypatho_variants_df = get_patho_and_likelypatho_SNVs(home_dir)
seq_record_list = get_protein_sequences(home_dir=home_dir, max_seq_len=1022, return_type="seq_record_list", data_type=task)


model_name = "sequnet"
model = model_utils.get_model()
model_task_out_dir, model_logits_out_dir = model_utils.create_output_directories(model_name=model_name, task=task)

def execute(seq_record_list):
    variants_df = patho_and_likelypatho_variants_df.copy(deep=True)
    preds = []   
    for i, seq_record in enumerate(seq_record_list):
        if "U" in str(seq_record.seq) or "B" in str(seq_record.seq): # seqnet skips the protein if it contains unknown amino acid "U"
            return pd.DataFrame(preds)
        pssm_df = model_utils.compute_model_prediction(model, seq_record, model_logits_out_dir)
        preds_related_to_aprot = model_utils.compute_variant_effect_scores(variants_df, seq_record, pssm_df)
        preds += preds_related_to_aprot
    preds_df = pd.DataFrame(preds)   
    return preds_df


if __name__=="__main__":
    start = time.time()

    data = seq_record_list

    chunk_size = 1
    data_chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]
    # data_chunks = data_chunks[:10] 
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
        #     # print(f"Finished {i}/{len(data_chunks)}th chunk: {pred_df.shape}")
        #     pred_dfs.append(pred_df)
        # executor.shutdown()
        
    result_df = pd.concat(pred_dfs)  
    print("Saving predictions ...")
    result_df.to_csv(f"{model_task_out_dir}/preds_{model_name}.tsv", sep="\t", index=False, header=True)
    print(result_df.shape)
    # print(result_df.head())
        
    print(f"Time taken: {time.time()-start} seconds")


