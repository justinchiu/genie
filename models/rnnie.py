import math

from .iebase import Ie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch

from .fns import attn

class RnnIe(Ie):
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
        joint = False,
    ):
        super(RnnIe, self).__init__()


        self._N = 0
        self.steps = 0

        self.joint = joint

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
        self.vproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vv),
            bias = False,
        ).spec("rnns", "v")


    def forward(self, x, x_info, etvs=None, e=None, t=None, v=None, learn=False):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        rnn_o, _ = self.rnn(emb, None, x_info.lengths)

        log_pe = self.eproj(self.drop(rnn_o)).log_softmax("e").cpu()
        log_pt = self.tproj(self.drop(rnn_o)).log_softmax("t").cpu()
        log_pv = self.vproj(self.drop(rnn_o)).log_softmax("v").cpu()

        nll = 0
        total = 0
        for batch, etv in enumerate(etvs):
            for i, (e, t, v, vt) in enumerate(etv):
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
                lpv = (log_pv
                    .get("batch", batch)
                    .get("time", i)
                    .gather("v", v, "v")
                    .logsumexp("v")
                    - math.log(T)
                )
                nll = nll - lpe - lpt - lpv
                total += 1
        if learn:
            nll.div(total).backward()
            self.steps += 1
        # return unnormalized vocab
        return nll, log_pe, log_pt, log_pv


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
