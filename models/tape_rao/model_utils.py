import sys
sys.path.append("../mutation_effect_analysis")

import os
import torch
import numpy as np

from tape import ProteinBertForMaskedLM, UniRepForLM, TAPETokenizer

import utils.pickle_utils as pickle_utils

def create_output_directories(model_name=None, task=None, home_dir=""):
    """model_name: protbert, unirep
       task: pathogenic, likely_pathogenic
    """
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/tape_rao/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


def get_model_tokenizer(model_name):
    """model_name: protbert, unirep
    """
    print("\nLog: Model loading ...")
    if model_name=="protbert":
        tokenizer = TAPETokenizer(vocab='iupac')
        model = ProteinBertForMaskedLM.from_pretrained('bert-base', cache_dir="models/tape_rao/cache/protbert")#, force_download=True)
    elif model_name=="unirep":
        tokenizer = TAPETokenizer(vocab='unirep')
        model = UniRepForLM.from_pretrained('babbler-1900', cache_dir="models/tape_rao/cache/unirep")
    model = model.to("cpu")
    model.eval()
    return model, tokenizer

# model, tokenizer = get_model_tokenizer("protbert")
# print(model)

# def compute_model_logits(model, tokenizer, prot_acc_version, seq, logits_output_path)->np.array:
#     filepath = f"{logits_output_path}{prot_acc_version}.pkl"
#     if os.path.exists(filepath):
#         print(f"Model logits already exists: {prot_acc_version}")
#         logits = pickle_utils.load_pickle(filepath) 
#     else: 
#         print(f"Computing model logits: {prot_acc_version}")
#         with torch.no_grad():
#             token_ids = torch.tensor(np.array([tokenizer.encode(seq)]))
#             logits = model(token_ids)[0][0].detach().numpy() 
#             pickle_utils.save_as_pickle(logits, filepath)
#     # print(logits.shape)
#     return logits
# unirep logits shape: l x vocab_size=25
# protbert logits shape: l x vocab_size=30

def compute_model_logits_from_masked_sequences(model, tokenizer, protid, seq, mut_pos, model_logits_out_dir):
    mut_pos_zero_idxd = mut_pos-1

    filepath = f"{model_logits_out_dir}{protid}_{str(mut_pos)}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {protid}_{str(mut_pos)}")
        logits = pickle_utils.load_pickle(filepath) 
    else: 
        print(f"Computing model logits: {protid}_{str(mut_pos)}")
        with torch.no_grad():
            seq = list(seq)
            seq[mut_pos_zero_idxd] = "<mask>"
            token_ids = torch.tensor(np.array([tokenizer.encode(seq)]))
            logits = model(token_ids)[0][0].detach().numpy() 
            pickle_utils.save_as_pickle(logits, filepath)
    # print(logits.shape)
    return logits


# def compute_variant_effect_scores(variants_df, tokenizer, prot_acc_version, output_logits):
#     preds = []
#     indices = variants_df[variants_df["prot_acc_version"]==prot_acc_version].index
#     # print(len(indices))
#     for idx in indices:
#         tuple = variants_df.loc[idx]
        
#         wt_tok_idx = tokenizer.vocab[tuple.wt]
#         mt_tok_idx = tokenizer.vocab[tuple.mut]
#         pos = tuple.prot_pos #ncbi prot variants are 1 indexed, so <cls> at 0-position does not matter
        
#         wt_logit = output_logits[pos][wt_tok_idx]
#         mt_logit = output_logits[pos][mt_tok_idx]
#         var_effect_score = mt_logit - wt_logit
#         tuple = dict(tuple)
#         tuple["pred"] = var_effect_score
#         preds.append(tuple)
#         # print(preds)
#         # break
#     return preds

def compute_variant_effect_scores_from_masked_logits(variants_df, tokenizer, protid, mut_pos, output_logits):
    preds = []
    indices = variants_df[(variants_df["prot_acc_version"]==protid) & (variants_df["1indexed_prot_mt_pos"]==mut_pos)].index 
    # print(len(indices))
    for idx in indices:
        tuple = variants_df.loc[idx]
        
        wt_tok_idx = tokenizer.vocab[tuple.wt_aa_1letter]
        mt_tok_idx = tokenizer.vocab[tuple.mt_aa_1letter]
        pos = mut_pos #ncbi prot variants are 1 indexed, so <cls> at 0-position does not matter
        
        wt_logit = output_logits[pos][wt_tok_idx]
        mt_logit = output_logits[pos][mt_tok_idx]
        var_effect_score = mt_logit - wt_logit
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds