import sys
sys.path.append("../mutation_effect_analysis")
home_dir=""

import os
import numpy as np

import utils.pickle_utils as pickle_utils


def create_output_directories(model_name=None, task=None, home_dir=""):
    print("\nLog: Creating output directories ...") 
    model_out_dir = home_dir+f"models/proteinbert_brandes/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


from proteinbert import load_pretrained_model
from proteinbert.tokenization import token_to_index, index_to_token

def get_model_tokenizer(model_name="proteinbert"):
    pretrained_model_generator, tokenizer = load_pretrained_model(local_model_dump_dir=home_dir+"models/proteinbert_brandes/cache/", 
                                                                      local_model_dump_file_name="epoch_92400_sample_23500000.pkl")
    model = pretrained_model_generator.create_model(1024)
    return model, tokenizer

def compute_model_logits(model, tokenizer, prot_acc_version, seq, logits_output_path)->np.array:
    filepath = f"{logits_output_path}{prot_acc_version}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {prot_acc_version}")
        logits = pickle_utils.load_pickle(filepath)
    else: 
        print(f"Computing model logits: {prot_acc_version}")
        x = tokenizer.encode_X([seq], 1024)
        logits, annotations = model(x) # shape=(n_seq=1, seq-len=1024, vocab_size=26), shape=(1, 8943). this returns a softmax scores
        logits = logits[0].numpy()
        pickle_utils.save_as_pickle(logits, filepath)
    # print(logits.shape)
    return logits

def compute_variant_effect_scores(variants_df, tokenizer, prot_acc_version, output_logits):
    preds = []
    indices = variants_df[variants_df["prot_acc_version"]==prot_acc_version].index 
    for idx in indices:
        tuple = variants_df.loc[idx]
        
        wt_tok_idx = token_to_index[tuple.wt_aa_1letter]
        mt_tok_idx = token_to_index[tuple.mt_aa_1letter]
        pos = tuple["1indexed_prot_mt_pos"] #ncbi prot variants are 1 indexed, outputs are 1-indexed, so no change
        
        wt_logit = np.log(output_logits[pos][wt_tok_idx])
        mt_logit = np.log(output_logits[pos][mt_tok_idx])
        var_effect_score = mt_logit - wt_logit
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds