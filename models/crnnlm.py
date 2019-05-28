from .base import Lm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch, NamedTensor

from .fns import attn

class CrnnLm(Lm):
    def __init__(
        self,
        Ve = None,
        Vt = None,
        Vv = None,
        Vx = None,
        r_emb_sz = 256,
        x_emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        tieweights = True,
        inputfeed = True,
        noattn = False,
    ):
        super(CrnnLm, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self.numexamples = 0

        self.Ve = Ve
        self.Vt = Vt
        self.Vv = Vv
        self.Vx = Vx
        self.r_emb_sz = r_emb_sz
        self.x_emb_sz = x_emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout
        self.inputfeed = inputfeed
        self.noattn = noattn

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
        self.lutv = ntorch.nn.Embedding(
            num_embeddings = len(Vv),
            embedding_dim = r_emb_sz,
            padding_idx = Vv.stoi[self.PAD],
        ).spec("els", "v")
        self.lutx = ntorch.nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        ).spec("time", "x")
        self.rnn = ntorch.nn.LSTM(
            input_size = x_emb_sz
                if not self.inputfeed
                else x_emb_sz + r_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = dropout,
            bidirectional = False,
        ).spec("x", "time", "rnns")
        self.drop = ntorch.nn.Dropout(dropout)
        self.proj = ntorch.nn.Linear(
            in_features = rnn_sz,
            out_features = len(Vx),
            bias = False,
        ).spec("ctxt", "vocab")

        # attn projection
        self.Wa = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("r", "rnns")

        # context projection
        self.Wc = ntorch.nn.Linear(
            in_features = r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")

        # Tie weights
        if tieweights:
            self.proj.weight = self.lutx.weight


    def forward(self, x, s, x_info, r, r_info, ue, ue_info, ut, ut_info, v2d):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        e = self.lute(r[0]).rename("e", "r")
        t = self.lutt(r[1]).rename("t", "r")
        v = self.lutv(r[2]).rename("v", "r")
        # r: R x N x Er, Wa r: R x N x H
        r = self.Wa(ntorch.cat([e, t, v], dim="r").tanh())

        if not self.inputfeed:
            # rnn_o: T x N x H
            rnn_o, s = self.rnn(emb, s, x_info.lengths)
            # ea: T x N x R
            _, ea, ec = attn(rnn_o, r, r_info.mask)
            if self.noattn:
                ec = r.mean("els").repeat("time", ec.shape["time"])
            self.ea = ea
            out = self.Wc(ntorch.cat([rnn_o, ec], "rnns")).tanh()
        else:
            out = []
            ect = NamedTensor(
                torch.zeros(N, self.r_emb_sz).to(emb.values.device),
                names=("batch", "rnns"),
            )
            for t in range(T):
                inp = ntorch.cat([emb.get("time", t), ect.rename("rnns", "x")], "x").repeat("time", 1)
                rnn_o, s = self.rnn(inp, s)
                rnn_o = rnn_o.get("time", 0)
                _, eat, ect = attn(rnn_o, r, r_info.mask)
                out.append(ntorch.cat([rnn_o, ect], "rnns"))
            out = self.Wc(ntorch.stack(out, "time")).tanh()

        # return unnormalized vocab
        return self.proj(self.drop(out)), s


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
