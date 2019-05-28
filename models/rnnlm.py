from .base import Lm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class RnnLm(Lm):
    def __init__(
        self,
        V = None,
        emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        tieweights = True,
    ):
        super(RnnLm, self).__init__()

        if tieweights:
            assert(emb_sz == rnn_sz)

        self._N = 0

        self.V = V
        self.emb_sz = emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout

        self.lut = nn.Embedding(
            num_embeddings = len(V),
            embedding_dim = emb_sz,
            padding_idx = V.stoi[self.PAD],
        )
        self.rnn = nn.LSTM(
            input_size = emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = 0.3,
            bidirectional = False,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(
            in_features = rnn_sz,
            out_features = len(V),
            bias = False,
        )

        # Tie weights
        if tieweights:
            self.proj.weight = self.lut.weight


    def forward(self, x, s, lens, r=None, lenr=None):
        emb = self.lut(x)
        p_emb = pack(emb, lens)
        x, s = self.rnn(p_emb, s)
        return self.proj(self.drop(unpack(x)[0])), s


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
            )
        return self._state
