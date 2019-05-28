import math

from .iebase import Ie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch

from .fns import attn

class RnnVie(Ie):
    def __init__(
        self,
        Ve = None,
        Vt = None,
        Vv = None,
        Vx = None,
        x_emb_sz = 256,
        r_emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        attn = "emb",
        joint = False,
    ):
        super(RnnVie, self).__init__()
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
        self.r_emb_sz = r_emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout

        self.lutx = ntorch.nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        ).spec("time", "x")
        self.lute = ntorch.nn.Embedding(
            num_embeddings = len(Ve),
            embedding_dim = r_emb_sz,
            padding_idx = Ve.stoi[self.PAD],
        ).spec("els", "e")
        self.lutt = ntorch.nn.Embedding(
            num_embeddings = len(Vt),
            embedding_dim = r_emb_sz,
            padding_idx = Vt.stoi[self.PAD],
        ).spec("els", "t")
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
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("e", "rnns")
        self.Wt = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("t", "rnns")
        self.Wet = ntorch.nn.Linear(
            in_features = 2*r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("et", "rnns")
        self.Wet1 = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("rnns", "rnns")

        # use text embeddings later if this isn't too slow
        self.vproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vv),
            bias = False,
        ).spec("rnns", "v")


    def forward(self, x, x_info, e, t, v=None, r_info=None, ds=None, learn=False):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        rnn_o, _ = self.rnn(emb, None, x_info.lengths)
        query = self.Wet(ntorch.cat(
            [
                self.lute(e).rename("e", "et"),
                self.lutt(t).rename("t", "et"),
            ],
            "et",
        ))
        uattn = query.dot("rnns", self.drop(rnn_o))
        uattn.masked_fill_(1-x_info.mask, float("-inf"))
        attn = uattn.softmax("time")
        ctxt = rnn_o.dot("time", attn)
        log_pv = self.vproj(ctxt).log_softmax("v")

        nll = None
        if v is not None:
            ll = log_pv.gather("v", v.repeat("i", 1), "i").get("i", 0)
            nll = -ll[r_info.mask].sum()
            total = r_info.mask.sum()

            if learn:
                nll.div(total).backward()
                self.steps += 1
        return nll, log_pv, attn


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
