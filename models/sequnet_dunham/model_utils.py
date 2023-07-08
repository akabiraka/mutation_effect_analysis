import sys
sys.path.append("../mutation_effect_analysis")

import os
import pandas as pd

from sequence_unet.models import load_trained_model
from sequence_unet.predict import predict_sequence

import utils.pickle_utils as pickle_utils


def create_output_directories(model_name="seqnet", task=None, home_dir=""):
    """model_name: seqnet
       task: pathogenic, likely_pathogenic
    """
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/sequnet_dunham/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir


def get_model():
    print("\nLog: Model loading ...")
    # model = load_trained_model(model="freq_classifier", download=True, root=os.path.abspath("models/sequnet_dunham"))
    model = load_trained_model(model="pssm_predictor", root=os.path.abspath("models/sequnet_dunham/cache")) # download=True, 
    return model


def compute_model_prediction(model, seq_record, pssm_output_path):
    filepath = f"{pssm_output_path}{seq_record.id}.pkl"
    if os.path.exists(filepath):
        print(f"Model pssm already exists: {seq_record.id}")
        pssm_df = pickle_utils.load_pickle(filepath) 
    else: 
        print(f"Computing model pssm: {seq_record.id}")
        pssm_df = pd.concat([p for p in predict_sequence(model, sequences=[seq_record], wide=True)])
        pickle_utils.save_as_pickle(pssm_df, filepath)
    return pssm_df


def compute_variant_effect_scores(variants_df, seq_record, pssm_df):
    preds = []
    indices = variants_df[variants_df["prot_acc_version"]==seq_record.id].index
    # print(len(indices))
    for idx in indices:
        tuple = variants_df.loc[idx]
        # pssm_df["position"] and tuple["1indexed_prot_mt_pos"] both are 1-indexed
        # print(tuple["1indexed_prot_mt_pos"], tuple.mut)
        if pssm_df[pssm_df["position"]==tuple["1indexed_prot_mt_pos"]].shape[0]==0: continue
        mt_score = pssm_df[pssm_df["position"]==tuple["1indexed_prot_mt_pos"]][tuple.mt_aa_1letter].values[0]
        wt_score = pssm_df[pssm_df["position"]==tuple["1indexed_prot_mt_pos"]][tuple.wt_aa_1letter].values[0]
        var_effect_score = mt_score - wt_score
        tuple = dict(tuple)
        tuple["pred"] = var_effect_score
        preds.append(tuple)
        # print(var_effect_score)
    return preds