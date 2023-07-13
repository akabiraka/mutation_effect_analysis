import sys
sys.path.append("../mutation_effect_analysis")

import os

def create_output_directories(model_name=None, task=None, home_dir=""):
    print("\nLog: Creating output directories ...") #-------------------------------------------
    model_out_dir = home_dir+f"models/jax_unirep/outputs/{model_name}/"
    model_logits_out_dir = f"{model_out_dir}lm_outputs/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir