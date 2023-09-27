import sys

sys.path.append("../mutation_effect_analysis")
home_dir = ""

import os
import torch

from vespa.predict.config import DEVICE, MODEL_PATH_DICT, EMBEDDING_HALF_PREC

# print(EMBEDDING_HALF_PREC)
# print(DEVICE)
is_vespa = True  # this param ensure both vespa and vespal


# step 1: embedding generation
from vespa.predict.embedding import T5_Embed

t5_emb = T5_Embed(cache_dir=home_dir + "models/vespa_marquet/cache")
# EMBEDDING_HALF_PREC must be set False for t5_emb model to run
model, tokenizer = t5_emb.prott5.get_model(0)  # EMBED=0


def compute_embedding(seq):
    seq_len = len(seq)
    seq = " ".join(list(seq))
    token_encoding = tokenizer(
        [seq], add_special_tokens=True, padding="longest", return_tensors="pt"
    )
    input_ids = token_encoding["input_ids"].to(DEVICE)
    attention_mask = token_encoding["attention_mask"].to(DEVICE)
    # print(input_ids, attention_mask)

    with torch.no_grad():
        embedding_repr = model(
            input_ids, attention_mask=attention_mask
        )  # 1 x seq_len x embedding_dim
        emb = embedding_repr.last_hidden_state[0, :seq_len]
        emb = emb.detach().cpu().numpy().squeeze()  # seq_len, 1024
        # print(emb.shape)
    return emb


# step 2: conservation prediction
from vespa.predict.conspred import ProtT5Cons
from pathlib import Path

checkpoint_path = Path(MODEL_PATH_DICT["CONSCNN"])
conspred = ProtT5Cons(checkpoint_path)
# print(checkpoint_path)
# conspred.predictor


def compute_conservation(embedding):
    with torch.no_grad():
        Yhat = conspred.predictor(torch.tensor(embedding).unsqueeze(0))
        prob = conspred.model.extract_probabilities(Yhat)
        # cls = conspred.model.extract_conservation_score(Yhat)

    # Yhat = Yhat.squeeze(0).detach().cpu().numpy()
    prob = prob.squeeze(0).detach().cpu().numpy()
    # cls = cls.squeeze(0).detach().cpu().numpy()
    # print(Yhat.shape, prob.shape, cls.shape) # shapes: (9, seq_len) (9, seq_len) (seq_len,)
    return prob


# step 3: computing log-odds
from vespa.predict.logodds import T5_condProbas

t5_condProbas = T5_condProbas(cache_dir=home_dir + "models/vespa_marquet/cache")


def compute_log_odds(seq_dict, mutation_generator):
    proba_dict = t5_condProbas.get_proba_dict(seq_dict, mutation_generator)
    dmiss_data = t5_condProbas.get_log_odds(
        proba_dict
    )  # seq_len, 20. DMISS (Deep mutational in-silico scanning.)
    return dmiss_data


# step 4: running vespal or (vespa, vespal)
from vespa.predict.vespa import VespaPred

vespa_predictor = VespaPred(vespa=is_vespa, vespal=True)

from vespa.predict.utils import MutationGenerator
import utils.pickle_utils as pickle_utils


def make_one_based_mutation(mutation):
    one_based_pos = str(int(mutation[1:-1]) + 1)
    return mutation[0] + one_based_pos + mutation[-1]


def compute_predictions(protid, seq, mutations):
    mutations_dict = {protid: mutations}  # mutations: ["M1C", "E2S"]

    temp_mutations_filepath = (
        home_dir + f"models/vespa_marquet/cache/temp_mutations_popu_freq/{protid}.txt"
    )
    temp_protid = "protid"
    with open(temp_mutations_filepath, "w") as f:
        for mutation in mutations_dict[protid]:
            f.write(f"{temp_protid}_{mutation}\n")
    temp_seq_dict = {temp_protid: seq}

    mutations_file_path = Path(temp_mutations_filepath)
    mutation_generator = MutationGenerator(
        temp_seq_dict, file_path=mutations_file_path, one_based_file=True
    )

    embedding = compute_embedding(seq)  # shape: seq_len, 1024
    conservation = compute_conservation(embedding)
    conservation_dict = {temp_protid: conservation}

    if is_vespa:
        log_odds = compute_log_odds(temp_seq_dict, mutation_generator)
        predictions = vespa_predictor.generate_predictions(
            mutation_generator, conservation_dict, log_odds
        )
    else:
        predictions = vespa_predictor.generate_predictions(
            mutation_generator, conservation_dict
        )

    # predictions[protid] = predictions.pop(temp_protid)
    predictions = predictions[temp_protid]
    predictions_dict = {
        make_one_based_mutation(mutation): vespa_preds
        for mutation, vespa_preds in predictions
    }
    return predictions_dict


def get_predictions(
    prot_acc_version, seq, mutations, variants_df, model_preds_out_dir, task
):
    new_out_path = home_dir + f"{model_preds_out_dir}{prot_acc_version}.pkl"
    prev_out_path = (
        home_dir
        + f"models/vespa_marquet/outputs/vespa/{task}/raw_predictions_prev/{prot_acc_version}.pkl"
    )

    if os.path.exists(new_out_path):
        print(
            f"..................loading vespa preds for {prot_acc_version}...................."
        )
        predictions_dict = pickle_utils.load_pickle(new_out_path)

    elif os.path.exists(prev_out_path):
        print(
            f"..................merging vespa preds for {prot_acc_version}......................"
        )
        prev_predictions = pickle_utils.load_pickle(prev_out_path)

        # print(prev_predictions)
        prev_predictions = prev_predictions[prot_acc_version]

        prev_predictions_dict = {
            make_one_based_mutation(mutation): vespa_preds
            for mutation, vespa_preds in prev_predictions
        }

        predictions_dict = {}
        remaining_mutations = []
        for m in mutations:
            if m in prev_predictions_dict:
                predictions_dict[m] = prev_predictions_dict[m]
            else:
                remaining_mutations.append(m)

        if len(remaining_mutations) > 0:
            new_predictions_dict = compute_predictions(
                prot_acc_version, seq, remaining_mutations
            )
            for m in remaining_mutations:
                predictions_dict[m] = new_predictions_dict[m]

        pickle_utils.save_as_pickle(predictions_dict, new_out_path)
    else:
        print(
            f"..................computing vespa preds for {prot_acc_version}..................."
        )
        predictions_dict = compute_predictions(prot_acc_version, seq, mutations)
        pickle_utils.save_as_pickle(predictions_dict, new_out_path)

    # raise
    print(predictions_dict)
    preds = []
    for mutation, vespa_preds in predictions_dict.items():
        indices = variants_df[
            (variants_df["prot_acc_version"] == prot_acc_version)
            & (variants_df["mut_real"] == mutation)
        ].index  # this must give 1 index at a time.
        print(prot_acc_version, mutation, indices)
        tuple = variants_df.loc[indices[0]]
        tuple = dict(tuple)

        if is_vespa:
            tuple["vespa_pred"] = vespa_preds["VESPA"]
        tuple["vespal_pred"] = vespa_preds["VESPAl"]
        preds.append(tuple)

    return preds


def create_output_directories(model_name=None, task=None, home_dir=""):
    print(
        "\nLog: Creating output directories ..."
    )  # -------------------------------------------
    model_out_dir = home_dir + f"models/vespa_marquet/outputs/{model_name}/"
    model_task_out_dir = f"{model_out_dir}{task}/"
    model_logits_out_dir = f"{model_task_out_dir}raw_predictions/"
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(model_logits_out_dir, exist_ok=True)
    os.makedirs(model_task_out_dir, exist_ok=True)
    return model_task_out_dir, model_logits_out_dir
