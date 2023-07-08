from pathlib import Path

from transformers import AlbertTokenizer, AlbertForMaskedLM

from bio_embeddings.embed.prottrans_base_embedder import ProtTransBertBaseEmbedder

import re
import torch

class ProtTransAlbertBFDLM(ProtTransBertBaseEmbedder):
    """ProtTrans-Albert-BFD Embedder (ProtAlbert-BFD)

    Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's
    Code Through Self-Supervised Deep Learning and High Performance Computing."
    arXiv preprint arXiv:2007.06225 (2020). https://arxiv.org/abs/2007.06225
    """

    _model: AlbertForMaskedLM
    name = "prottrans_albert_bfd"
    embedding_dimension = 4096
    number_of_layers = 1
    aa_prefix = "▁"

    def __init__(self, **kwargs):
        """Initialize Albert embedder.

        :param model_directory:
        :param half_precision_model:
        """
        super().__init__(**kwargs)

        self._model_directory = self._options["model_directory"]
        self._half_precision_model = self._options.get("half_precision_model", False)
        
        # make model
        self._model = AlbertForMaskedLM.from_pretrained(self._model_directory)
        # Compute in half precision, which is a lot faster and saves us half the memory
        if self._half_precision_model:
            self._model = self._model.half()
        self._model = self._model.eval().to(self._device)
        self._model_fallback = None
        self._tokenizer = AlbertTokenizer(
            str(Path(self._model_directory) / "albert_vocab_model.model"),
            do_lower_case=False,
        )

    def _get_fallback_model(self) -> AlbertForMaskedLM:
        """ Returns the CPU model """
        if not self._model_fallback:
            self._model_fallback = AlbertForMaskedLM.from_pretrained(
                self._model_directory
            ).eval()
        return self._model_fallback
    
    def get_masked_logits(self, seq, zero_indexed_mutpos):
        seq_len = len(seq)
        seq = re.sub(r"[UZOB]", "X", seq) # replacing unknown amino acid with unknown token
        seq = list(seq)
        seq[zero_indexed_mutpos] = self._tokenizer.mask_token # mut_pos must be 0-indexed. replace AA by special mask token used by the model
        seq = " ".join(list(seq)) # space separated amino acids

        ids = self._tokenizer.batch_encode_plus(
            [seq], add_special_tokens=True, padding="longest"
        )
        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)

        with torch.no_grad():
            logits = self._model(input_ids=tokenized_sequences, attention_mask=attention_mask)
        
        logits = logits[0].squeeze().cpu().numpy()
        logits = logits[1:seq_len+1]
        # print(logits.shape) # seq_len, vocab_size=30
        return logits
