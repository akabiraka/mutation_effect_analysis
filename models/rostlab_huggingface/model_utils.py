import sys
sys.path.append("../mutation_effect_analysis")

import re
import os
import torch
import numpy as np

import utils.pickle_utils as pickle_utils


def create_output_directories(model_name=None, task=None, home_dir=""):
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/rostlab_huggingface/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


def compute_model_logits_from_masked_sequences(model, tokenizer, protid, seq, mut_pos, model_logits_out_dir):
    zero_indexed_mutpos = mut_pos-1

    filepath = f"{model_logits_out_dir}{protid}_{str(mut_pos)}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {protid}_{str(mut_pos)}")
        logits = pickle_utils.load_pickle(filepath) 
    else: 
        print(f"Computing model logits: {protid}_{str(mut_pos)}")
        seq_len = len(seq)
        seq = re.sub(r"[UZOB]", "X", seq) # replacing unknown amino acid with unknown token
        seq = list(seq)

        seq[zero_indexed_mutpos] = "<extra_id_0>" #"[MASK]"# tokenizer.mask_token #'<extra_id_0>' # mut_pos must be 0-indexed. replace AA by special mask token used by the model

        seq = " ".join(list(seq)) # space separated amino acids
        # print(seq)

        # <eos> token at the end
        # starts from 0-index
        # input_ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest")
        # tokenized_sequences = torch.tensor(input_ids["input_ids"]).to("cpu")
        # attention_mask = torch.tensor(input_ids["attention_mask"]).to("cpu")
        input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cpu")

        with torch.no_grad():
            # logits = model(input_ids=tokenized_sequences, attention_mask=attention_mask, decoder_input_ids=tokenized_sequences).logits
            logits = model(input_ids, labels=input_ids).logits

        logits = logits.squeeze().detach().numpy()
        logits = logits[:seq_len]

        pickle_utils.save_as_pickle(logits, filepath)

        # print(seq_len, logits.shape)
    return logits # (seq_len, vocab_size=128)



def compute_variant_effect_scores_from_masked_logits(variants_df, tokenizer, prot_acc_version, mut_pos, output_logits, model_aa_prefix): # "▁"
    preds = []
    indices = variants_df[(variants_df["prot_acc_version"]==prot_acc_version) & (variants_df["1indexed_prot_mt_pos"]==mut_pos)].index 
    for idx in indices:
        tuple = variants_df.loc[idx]
        
        wt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.wt_aa_1letter)
        mt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.mt_aa_1letter)
        # print(wt_tok_idx, mt_tok_idx)
        pos = mut_pos-1 #ncbi prot variants are 1 indexed, outputs are 0-indexed, so -1. PMD mut_real col is 1-indexed
        
        wt_logit = output_logits[pos][wt_tok_idx]
        mt_logit = output_logits[pos][mt_tok_idx]
        var_effect_score = mt_logit - wt_logit
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds