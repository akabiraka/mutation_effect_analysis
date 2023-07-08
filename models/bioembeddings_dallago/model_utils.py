import sys
sys.path.append("../mutation_effect_analysis")

import os
import torch
import numpy as np

import utils.pickle_utils as pickle_utils


def create_output_directories(model_name=None, task=None, home_dir=""):
    """model_name: plus_rnn, prottrans_bert_bfd, prottrans_albert_bfd, prottrans_xlnet_uniref100, prottrans_t5_bfd, prottrans_t5_uniref50, prottrans_t5_xl_u50
       task: pathogenic, likely_pathogenic
    """
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/bioembeddings_dallago/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


def get_model_tokenizer(model_name=None):
    """model_name: plus_rnn, prottrans_bert_bfd, prottrans_albert_bfd, prottrans_xlnet_uniref100, prottrans_t5_bfd, prottrans_t5_uniref50, prottrans_t5_xl_u50
    """
    print("\nLog: Model loading ...") #-------------------------------------------
    model = load_prottrans_lm_model(model_name) # plus_rnn, prottrans_bert_bfd, prottrans_albert_bfd, prottrans_xlnet_uniref100, prottrans_t5_bfd, prottrans_t5_uniref50, prottrans_t5_xl_u50
    return model, model._tokenizer, model.name


def compute_model_logits(model, prot_acc_version, seq, logits_output_path)->np.array:
    filepath = f"{logits_output_path}{prot_acc_version}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {prot_acc_version}")
        logits = pickle_utils.load_pickle(filepath)
    else: 
        print(f"Computing model logits: {prot_acc_version}")
        with torch.no_grad():
            logits = model.embed(seq) 
            pickle_utils.save_as_pickle(logits, filepath)
    # print(logits.shape)
    return logits

def compute_model_logits_from_masked_sequences(model, tokenizer, protid, seq, mut_pos, model_logits_out_dir):
    zero_indexed_mutpos = mut_pos-1

    filepath = f"{model_logits_out_dir}{protid}_{str(mut_pos)}.pkl"
    if os.path.exists(filepath):
        print(f"Model logits already exists: {protid}_{str(mut_pos)}")
        logits = pickle_utils.load_pickle(filepath) 
    else: 
        print(f"Computing model logits: {protid}_{str(mut_pos)}")
        logits = model.get_masked_logits(seq, zero_indexed_mutpos) 
        pickle_utils.save_as_pickle(logits, filepath)
    return logits # (seq_len, vocab_size=128)


def compute_variant_effect_scores(variants_df, tokenizer, prot_acc_version, output_logits, model_aa_prefix):
    preds = []
    indices = variants_df[variants_df["prot_acc_version"]==prot_acc_version].index 
    # print(prot_acc_version, len(indices)) # indices can be of different shape for different runs, b/c we sample the singletons when computing variants_df
    for idx in indices:
        tuple = variants_df.loc[idx]
        
        wt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.wt_aa_1letter)
        mt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.mt_aa_1letter)
        pos = tuple["1indexed_prot_mt_pos"]-1 #ncbi prot variants are 1 indexed, outputs are 0-indexed, so -1

        # print(output_logits.shape, pos)
        # if pos>=output_logits.shape[0]: continue
        
        wt_logit = output_logits[pos][wt_tok_idx]
        mt_logit = output_logits[pos][mt_tok_idx]
        var_effect_score = mt_logit - wt_logit
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds

def compute_variant_effect_scores_from_masked_logits(variants_df, tokenizer, protid, mut_pos, output_logits, model_aa_prefix):
    preds = []
    indices = variants_df[(variants_df["prot_acc_version"]==protid) & (variants_df["1indexed_prot_mt_pos"]==mut_pos)].index  # mut_pos is 1-indexed here
    for idx in indices:
        tuple = variants_df.loc[idx]
        
        wt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.wt_aa_1letter)
        mt_tok_idx = tokenizer.convert_tokens_to_ids(model_aa_prefix+tuple.mt_aa_1letter)
        pos = mut_pos-1 #ncbi prot variants are 1 indexed, outputs are 0-indexed, so -1. PMD mut_real col is 1-indexed

        # print(output_logits.shape, pos)
        # if pos>=output_logits.shape[0]: continue
        
        wt_logit = output_logits[pos][wt_tok_idx]
        mt_logit = output_logits[pos][mt_tok_idx]
        var_effect_score = mt_logit - wt_logit
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(preds)
        # break
    return preds


def load_prottrans_lm_model(model_name:str):
    """options: plus_rnn, prottrans_bert_bfd, prottrans_albert_bfd, prottrans_xlnet_uniref100, prottrans_t5_bfd, prottrans_t5_uniref50, prottrans_t5_xl_u50
    """
    if model_name == "prottrans_bert_bfd":
        from models.bioembeddings_dallago.lm_heads.prottrans_bert_bfd_lm import ProtTransBertBFDLM
        model = ProtTransBertBFDLM() # logits shape: (seq_len, vocab_size=30)
    
    elif model_name == "prottrans_albert_bfd":
        from models.bioembeddings_dallago.lm_heads.prottrans_albert_bfd_lm import ProtTransAlbertBFDLM
        model = ProtTransAlbertBFDLM() # logits shape: (seq_len, vocab_size=34)
    
    elif model_name == "prottrans_xlnet_uniref100":
        from models.bioembeddings_dallago.lm_heads.prottrans_xlnet_uniref100_lm import ProtTransXLNetUniRef100LM
        model = ProtTransXLNetUniRef100LM() # logits shape: (seq_len, vocab_size=37)
    
    elif model_name == "prottrans_t5_bfd":
        from models.bioembeddings_dallago.lm_heads.prottrans_t5_lm import ProtTransT5BFDLM 
        model = ProtTransT5BFDLM() # logits shape: (seq_len, vocab_size=128)
        
    elif model_name == "prottrans_t5_uniref50":
        from models.bioembeddings_dallago.lm_heads.prottrans_t5_lm import ProtTransT5UniRef50LM
        model = ProtTransT5UniRef50LM() # logits shape: (seq_len, vocab_size=128)
        
    elif model_name == "prottrans_t5_xl_u50":
        from models.bioembeddings_dallago.lm_heads.prottrans_t5_lm import ProtTransT5XLU50LM
        model = ProtTransT5XLU50LM() # logits shape: (seq_len, vocab_size=128)
        
    elif model_name == "plus_rnn":
        from models.bioembeddings_dallago.lm_heads.plus_rnn_lm import PLUSRNNLM
        model = PLUSRNNLM() # logits shape: (seq_len, vocab_size=21)
        
    else:
        raise NotImplementedError()
        
    return model

