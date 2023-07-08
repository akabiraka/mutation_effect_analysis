import sys
sys.path.append("../mutation_effect_analysis")

import os
import torch

import esm

import utils.pickle_utils as pickle_utils


def create_output_directories(model_name=None, task=None, home_dir=""):
    """model_name: esm1_t6_43M_UR50S, esm1b_t33_650M_UR50S, esm1v_t33_650M_UR90S, esm2_t33_650M_UR50D
       task: pathogenic, likely_pathogenic
    """
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/esm_rives/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


def get_model_tokenizer(model_name="esm1b_t33_650M_UR50S"):
    """model_name: esm1_t6_43M_UR50S, esm1b_t33_650M_UR50S, esm1v_t33_650M_UR90S, esm2_t33_650M_UR50D
    """
    print("\nLog: Model loading ...")
    if model_name == "esm1_t6_43M_UR50S":
        model, alphabet = esm.pretrained.esm1_t6_43M_UR50S() 
    elif model_name == "esm1b_t33_650M_UR50S": # l+2 x vocab_size=33
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_name == "esm1v_t33_650M_UR90S": # l+2 x vocab_size=33
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
    elif model_name == "esm2_t33_650M_UR50D": # l+2 x vocab_size=33
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, batch_converter


def compute_model_logits(model, batch_converter, prot_acc_version, seq, logits_output_path):
    filepath = f"{logits_output_path}{prot_acc_version}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {prot_acc_version}")
        logits = pickle_utils.load_pickle(filepath) # numpy array of l+2 x vocab_size=33
    else: 
        print(f"Computing model logits: {prot_acc_version}")
        with torch.no_grad():
            _, _, batch_tokens = batch_converter([(prot_acc_version, seq)])
            logits = model(batch_tokens)["logits"][0].detach().numpy() # l+2 x vocab_size=33
            pickle_utils.save_as_pickle(logits, filepath)
    # print(logits.shape)
    return logits

def compute_model_logits_from_masked_sequences(model, batch_converter, protid, seq, mut_pos, model_logits_out_dir):
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
            seq = " ".join(seq)
            _, _, batch_tokens = batch_converter([(protid, seq)])
            logits = model(batch_tokens)["logits"][0].detach().numpy() # l+2 x vocab_size=33
            pickle_utils.save_as_pickle(logits, filepath)
    return logits


def compute_variant_effect_scores(variants_df, alphabet, prot_acc_version, output_logits):
    preds = []
    indices = variants_df[variants_df["prot_acc_version"]==prot_acc_version].index
    for idx in indices:
        tuple = variants_df.loc[idx]

        wt_tok_idx = alphabet.get_idx(tuple.wt_aa_1letter)
        mt_tok_idx = alphabet.get_idx(tuple.mt_aa_1letter)
        pos = tuple["1indexed_prot_mt_pos"] #ncbi prot variants are 1 indexed, so <cls> at 0-position does not matter

        wt_logit = output_logits[pos][wt_tok_idx]
        mt_logit = output_logits[pos][mt_tok_idx]
        var_effect_score = mt_logit - wt_logit

        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds


def compute_variant_effect_scores_from_masked_logits(variants_df, alphabet, protid, mut_pos, output_logits):
    preds = []
    indices = variants_df[(variants_df["prot_acc_version"]==protid) & (variants_df["1indexed_prot_mt_pos"]==mut_pos)].index 
    for idx in indices:
        tuple = variants_df.loc[idx]

        wt_tok_idx = alphabet.get_idx(tuple.wt_aa_1letter)
        mt_tok_idx = alphabet.get_idx(tuple.mt_aa_1letter)
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