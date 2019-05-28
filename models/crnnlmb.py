from .base import Lm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch

from .fns import attn

class CrnnLmB(Lm):
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
        noattnvalues = False,
        sigbias = 0,
    ):
        super(CrnnLmB, self).__init__()

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
        self.noattnvalues = noattnvalues

        self.sigbias = sigbias

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
        self.Wae = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
        ).spec("e", "rnns")
        self.Wat = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
        ).spec("t", "rnns")
        self.War = ntorch.nn.Linear(
            in_features = 3*r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("r", "rnns")

        # context projection
        self.Wc = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz + rnn_sz,
            #in_features = 2 * r_emb_sz + rnn_sz,
            #in_features = r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")
        self.Wc_nov = ntorch.nn.Linear(
            in_features = 2 * r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")
        self.Wr = ntorch.nn.Linear(
            in_features = 2 * rnn_sz,
            #in_features = 2 * r_emb_sz + rnn_sz,
            #in_features = r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")

        # Tie weights
        if tieweights:
            self.proj.weight = self.lutx.weight


    def forward(self, x, s, x_info, r, r_info, vt, ue, ue_info, ut, ut_info, v2d):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        e = self.lute(r[0]).rename("e", "r")
        t = self.lutt(r[1]).rename("t", "r")
        v = self.lutv(r[2]).rename("v", "r")
        # r: R x N x Er, Wa r: R x N x H
        r = self.War(ntorch.cat([e, t, v], dim="r"))

        uee = self.lute(ue)
        ute = self.lutt(ut)
        eA = self.Wae(uee).chop("rnns", ("a", "rnns"), a=2)
        tA = self.Wat(ute).chop("rnns", ("a", "rnns"), a=2)
        v2dx = self.lutx(
            v2d.stack(("t", "e"), "time")
        ).chop("time", ("t", "e"), t=v2d.shape["t"]).rename("x", "rnns")

        if not self.inputfeed:
            # rnn_o: T x N x H
            rnn_o, s = self.rnn(emb, s, x_info.lengths)

            pv0 = (rnn_o.dot("rnns", r) - self.sigbias).sigmoid()
            vc = pv0.dot("els", r)

            out = self.Wr(ntorch.cat([rnn_o, vc], "rnns")).tanh()
            """

            # independent attn
            # masking doesn't work yet
            log_pe = rnn_o.dot("rnns", eA).log_softmax("a")
            #log_pe.masked_fill_(1-ue_info.mask, float("-inf"))
            log_pt = rnn_o.dot("rnns", tA).log_softmax("a")
            #log_pt.masked_fill_(1-ut_info.mask, float("-inf"))

            ec = log_pe.get("a", 1).exp().dot("els", uee).rename("e", "rnns")
            tc = log_pt.get("a", 1).exp().dot("els", ute).rename("t", "rnns")

            le = log_pe.rename("els", "e")
            lt = log_pt.rename("els", "t")
            lv = le + lt
            vc = lv.exp().dot(("t", "e"), v2dx).get("a", 1)

            self.le = le
            self.lt = lt
            self.lv = lv

            # Need bias so context doesn't dominate language model in beginning
            pe0 = (rnn_o.dot("rnns", eA).get("a", 1) - self.sigbias).sigmoid()
            #log_pe.masked_fill_(1-ue_info.mask, float("-inf"))
            pt0 = (rnn_o.dot("rnns", tA).get("a", 1) - self.sigbias).sigmoid()
            #log_pt.masked_fill_(1-ut_info.mask, float("-inf"))

            ec0 = pe0.dot("els", uee).rename("e", "rnns")
            tc0 = pt0.dot("els", ute).rename("t", "rnns")

            pv0 = pe0.rename("els", "e") * pt0.rename("els", "t")
            vc0 = pv0.dot(("t", "e"), v2dx)

            self.pe = pe0
            self.pt = pt0
            self.pv = pv0

            # no ent or typ
            #out = self.Wc(ntorch.cat([rnn_o, ec], "rnns")).tanh()
            # cat ent and type, this seems fine
            out = (self.Wc_nov(ntorch.cat([rnn_o, ec, tc], "rnns"))
                if self.noattnvalues else
                #self.Wc(ntorch.cat([rnn_o, vc, ec, tc], "rnns"))
                self.Wc(ntorch.cat([rnn_o, vc0, ec0, tc0], "rnns"))
            ).tanh()
            """
            # add ent and typ
            #out = self.Wc(ntorch.cat([rnn_o, ec + ec_ET], "rnns")).tanh()
        else:
            raise NotImplimentedError
            out = []
            self.ea = []
            self.ta = []
            self.a = []
            ec_ETt = ntorch.zeros(
                N, self.r_emb_sz, names=("batch", "rnns")
            ).to(emb.values.device)
            for t in range(T):
                ec_ETt = ec_ETt.rename("rnns", "x")
                inp = ntorch.cat([emb.get("time", t), ec_ETt], "x").repeat("time", 1)
                rnn_o, s = self.rnn(inp, s)
                rnn_o = rnn_o.get("time", 0)
                log_e, ea_Et, ec_Et = attn(rnn_o, eA, ue_info.mask)
                log_t, ea_Tt, ec_Tt = attn(rnn_o, tA, ut_info.mask)
                ec_ETt = ec_Et + ec_Tt
                le = log_e.rename("els", "e")
                lt = log_t.rename("els", "t")
                aw = (le + lt).exp()
                ect = aw.dot(("t", "e"), v2dx)
                out.append(ntorch.cat([rnn_o, ect, ec_Et, ec_Tt], "rnns"))
                self.ea.append(ea_Et.detach())
                self.ta.append(ea_Tt.detach())
                self.a.append(aw.detach())
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


