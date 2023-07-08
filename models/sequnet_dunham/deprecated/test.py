import sys
sys.path.append("../../variant_effect_analysis")

import os
import pandas as pd
from Bio import SeqIO

from sequence_unet.models import load_trained_model, download_trained_model
from sequence_unet.predict import predict_sequence

# load a model
# download_trained_model(model="pssm_predictor", root=os.path.abspath("models/sequnet_dunham"), model_format="tf")
model = load_trained_model(model="pssm_predictor", download=True, root=os.path.abspath("models/sequnet_dunham"))

# Predict from a fasta file
fasta_iterator = SeqIO.parse("models/sequnet_dunham/sample_seqs.fasta", format="fasta")
for seq_record in fasta_iterator:
    pssm_df = pd.concat([p for p in predict_sequence(model, sequences=[seq_record], wide=True)])
    print(pssm_df.shape)
    print(pssm_df)
    one_indexed_mut_pos = 62
    print(pssm_df[(pssm_df["position"]==one_indexed_mut_pos)])
    print(pssm_df[(pssm_df["position"]==one_indexed_mut_pos)]["S"].values[0], pssm_df[(pssm_df["position"]==one_indexed_mut_pos)]["N"].values[0])
    print(pssm_df[(pssm_df["position"]==one_indexed_mut_pos)]["S"].values[0] - pssm_df[(pssm_df["position"]==one_indexed_mut_pos)]["N"].values[0])
