import sys
sys.path.append("../mutation_effect_analysis")
home_dir = ""

import pandas as pd

from models.aa_common.data_loader import get_pmd_dbnsfp_dataset, get_popu_freq_dbnsfp_dataset, get_patho_likelypatho_neutral_dbnsfp_dataset
import models.rostlab_huggingface.model_utils as model_utils



task = "pmd" # pmd, popu_freq, patho
variants_df, protid_seq_dict = get_pmd_dbnsfp_dataset(home_dir)
# variants_df, protid_seq_dict = get_popu_freq_dbnsfp_dataset(home_dir)
# variants_df, protid_seq_dict = get_patho_likelypatho_neutral_dbnsfp_dataset(home_dir)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "prottrans_t5_xl_u50" 
model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50") # from transformers import T5Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", use_fast=False)#, force_download=True)
model = model.to("cpu")
model_aa_prefix="▁"

model_task_out_dir, model_logits_out_dir = model_utils.create_output_directories(model_name, task)

def execute(protid_mutpos_tuple_list):
    preds = []        
    for i, (protid, mut_pos) in enumerate(protid_mutpos_tuple_list):
        # mut_pos is 1 indexed here
        seq = protid_seq_dict[protid]
        output_logits = model_utils.compute_model_logits_from_masked_sequences(model, tokenizer, protid, seq, mut_pos, model_logits_out_dir)
        preds_related_to_aprot = model_utils.compute_variant_effect_scores_from_masked_logits(variants_df, tokenizer, protid, mut_pos, output_logits, model_aa_prefix)
        preds += preds_related_to_aprot
    preds_df = pd.DataFrame(preds)   
    return preds_df

if __name__ == "__main__":

    protid_mutpos_df = variants_df[["prot_acc_version", "1indexed_prot_mt_pos"]].drop_duplicates(keep="first") 
    data = list(protid_mutpos_df.itertuples(index=False, name=None))

    chunk_size = 1 # 32 if torch.cuda.is_available() else 1
    data_chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]
    # data_chunks = data_chunks[:5] 
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