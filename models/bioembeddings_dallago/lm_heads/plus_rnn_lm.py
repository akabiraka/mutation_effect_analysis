from typing import Union, List, Generator

import torch
import torch.nn.functional as F
from numpy import ndarray
from plus.config import ModelConfig, RunConfig
from plus.data.alphabets import Protein
from plus.data.dataset import Embedding_dataset, collate_sequences_for_embedding
from plus.model.plus_rnn import PLUS_RNN, get_embedding
from plus.train import Trainer
from plus.utils import set_seeds
from torch.utils.data import DataLoader

from bio_embeddings.embed import EmbedderInterface


class PLUSRNNTokenizer(Protein):
    def __init__(self):
        super(PLUSRNNTokenizer, self).__init__()

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self.encode(tokens.encode().upper())[0]

        ids = []
        for token in tokens:
            ids.append(self.encode(token.encode().upper()))
        return ids


class PLUSRNNLM(EmbedderInterface):
    """PLUS RNN LM

    Pre-Training of Deep Bidirectional Protein Sequence Representations with Structural Information
    Seonwoo Min, Seunghyun Park, Siwon Kim, Hyun-Soo Choi, Sungroh Yoon
    https://arxiv.org/abs/1912.05625"""

    name = "plus_rnn"
    number_of_layers = 1
    embedding_dimension = 1024
    aa_prefix = ""

    necessary_files = ["model_file"]

    _tokenizer: PLUSRNNTokenizer  # Protein
    _model: PLUS_RNN
    _model_cfg: ModelConfig
    _run_cfg: RunConfig

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)
        # This seed is copied from PLUS
        set_seeds(2020)

        # We inlined the config json files since they aren't shipped with the package
        self._tokenizer = PLUSRNNTokenizer()  # Protein()
        self._model_cfg = ModelConfig(input_dim=len(self._tokenizer))
        self._model_cfg.model_type = "RNN"
        self._model_cfg.rnn_type = "B"
        self._model_cfg.num_layers = 3
        self._model_cfg.hidden_dim = 512
        self._model_cfg.embedding_dim = 100
        self._run_cfg = RunConfig(sanity_check=True)
        self._run_cfg.batch_size_eval = 512
        self._model = PLUS_RNN(self._model_cfg)
        self._model_directory = self._options["model_file"]
        self._model.load_weights(self._options["model_file"])
        self._model = self._model.eval().to(self._device)

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        sequences = [
            self._tokenizer.encode(sequence.encode().upper()) for sequence in batch
        ]
        # print(sequences) # tokenization of the input sequences
        test_dataset = [torch.from_numpy(sequence).long() for sequence in sequences]
        test_dataset = Embedding_dataset(
            test_dataset, self._tokenizer, self._run_cfg, True
        )

        iterator_test = DataLoader(
            test_dataset,
            self._run_cfg.batch_size_eval,
            collate_fn=collate_sequences_for_embedding,
        )

        model_list = [self._model, "", True, False, False]
        # model_list_lm = [self._model, "lm", True, False, False]
        tasks_list = [["", [], []]]  # list of lists [idx, metrics_train, metrics_eval]
        trainer = Trainer([model_list], get_embedding, self._run_cfg, tasks_list)
        for tokens, lengths in iterator_test:
            # https://github.com/pytorch/pytorch/issues/43227
            batch = (tokens.to(self._device), lengths)
            trainer.embed(batch, {"data_parallel": False})

        embeddings = trainer.tasks_dict["results_eval"][0]["embeddings"]
        # 1 is d_h with 1024 dimensions
        for i in range(len(embeddings[0])):
            yield embeddings[1][i]  # .numpy()

        trainer.reset()

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        embedding = embedding.unsqueeze(0)
        b, n, d = embedding.size()  # batch_size, seq_len, dimension (1, seq_len, 1024)
        embedding = self._model.decoder(embedding.contiguous().view(-1, d)).view(
            b, n, -1
        )
        logits = F.log_softmax(embedding, dim=2)
        return logits.squeeze_().detach().numpy()  # seq_len, 21
        # return embedding

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return embedding.mean(axis=0)
