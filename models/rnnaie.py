import math

from .iebase import Ie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch

from .fns import attn

class RnnAie(Ie):
    def __init__(
        self,
        Ve = None,
        Vt = None,
        Vv = None,
        Vx = None,
        x_emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        attn = "emb",
        joint = False,
    ):
        super(RnnAie, self).__init__()
        assert(attn in ["emb", "rnn"])

        self.attn = attn
        self.joint = joint

        self._N = 0
        self.steps = 0

        self.Ve = Ve
        self.Vt = Vt
        self.Vv = Vv
        self.Vx = Vx
        self.x_emb_sz = x_emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout

        self.lutx = ntorch.nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        ).spec("time", "x")
        self.rnn = ntorch.nn.LSTM(
            input_size = x_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = dropout,
            bidirectional = True,
        ).spec("x", "time", "rnns")
        self.drop = ntorch.nn.Dropout(dropout)

        self.We = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = 2*rnn_sz,
            bias = False,
        )
        self.Wt = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = 2*rnn_sz,
            bias = False,
        )
        self.We1 = ntorch.nn.Linear(
            in_features = 2*rnn_sz + x_emb_sz
                if self.attn == "emb"
                else 4*rnn_sz,
            out_features = 2*rnn_sz,
            bias = False,
        )
        self.Weo = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = 2*rnn_sz,
            bias = False,
        )

        self.eproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Ve),
            bias = False,
        ).spec("rnns", "e")
        self.tproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vt),
            bias = False,
        ).spec("rnns", "t")


    def forward(self, x, x_info, ds=None, e=None, t=None, v=None, learn=False):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        rnn_o, _ = self.rnn(emb, None, x_info.lengths)

        eA = (self.We(self.drop(rnn_o))
            .rename("time", "self")
            .dot("rnns", rnn_o))
        eA.masked_fill_(1-x_info.mask.rename("time", "self"), float("-inf"))
        eA = eA.softmax("self")
        ec = (eA.dot("self", emb.rename("time", "self")).rename("x", "rnns")
            if self.attn == "emb"
            else eA.dot("self", rnn_o.rename("time", "self")))
        eo = self.Weo(self.We1(ntorch.cat([ec, rnn_o], "rnns")).tanh())

        log_pe = self.eproj(eo).log_softmax("e")
        log_pt = self.tproj(rnn_o).log_softmax("t")

        nlpe = 0
        nlpt = 0
        total = 0
        for batch, d in enumerate(ds):
            for i, (e, t) in d.items():
                i = i + 1
                T = e.shape["e"]
                lpe = (log_pe
                    .get("batch", batch)
                    .get("time", i)
                    .gather("e", e, "e")
                    .logsumexp("e")
                    - math.log(T)
                )
                lpt = (log_pt
                    .get("batch", batch)
                    .get("time", i)
                    .gather("t", t, "t")
                    .logsumexp("t")
                    - math.log(T)
                )
                nlpe = nlpe - lpe
                nlpt = nlpt - lpt

                total += 1
        nll = nlpe + nlpt
        if learn:
            nll.div(total).backward()
            self.steps += 1
        return nll, log_pe, log_pt


    def init_state(self, N):
        # what's this for?
        if self._N != N:
            self._N = N
            self._state = (
                ntorch.zeros(
                    self.nlayers, N, self.rnn_sz,
                    names=("layers", "batch", "rnns"),
                ).to(self.lutx.weight.device),
                ntorch.zeros(
                    self.nlayers, N, self.rnn_sz,
                    names=("layers", "batch", "rnns"),
                ).to(self.lutx.weight.device),
            )
        return self._state
