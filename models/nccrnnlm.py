from .lvmnc import LvmNc, RvInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from .rnnvie import RnnVie
from .crnnlm import CrnnLm


from namedtensor import ntorch, NamedTensor

from .fns import attn, kl_qz

class CrnnLmV(CrnnLm):
    def forward(self, x, s, x_info, r, r_info, ue, ue_info, ut, ut_info, v2d):
        emb = self.lutx(x)
        N = emb.shape["batch"]
        T = emb.shape["time"]

        e,t,v = r
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
            out = self.Wc(ntorch.cat([rnn_o.repeat("k", ec.shape["k"]), ec], "rnns")).tanh()
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


class NcCrnnLm(LvmNc):
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
        super(NcCrnnLm, self).__init__()
        self.rnnvie = RnnVie(
            Ve = Ve,
            Vt = Vt,
            Vv = Vv,
            Vx = Vx,
            x_emb_sz = x_emb_sz,
            r_emb_sz = r_emb_sz,
            rnn_sz = rnn_sz,
            nlayers = nlayers,
            dropout = dropout,
            attn    = "emb",
            joint   = False,
        )
        self.crnnlm = CrnnLmV(
            Ve = Ve,
            Vt = Vt,
            Vv = Vv,
            Vx = Vx,
            r_emb_sz = r_emb_sz,
            x_emb_sz = x_emb_sz,
            rnn_sz = rnn_sz,
            nlayers = nlayers,
            dropout = dropout,
            tieweights = tieweights,
            inputfeed = inputfeed,
            noattn = False,
        )

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

        self.vproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vv),
            bias = False,
        ).spec("rnns", "v")

        self.queryproj = ntorch.nn.Linear(
            in_features = 2*r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("et", "rnns")


    def sample_v(self, probs, logits, K, dim, sampledim):
        v_s = NamedTensor(torch.multinomial(
            probs.stack(("batch", "els"), "samplebatch").transpose("samplebatch", dim).values,
            K,
            True,
        ), names=("samplebatch", sampledim)).chop("samplebatch", ("batch", "els"), batch=logits.shape["batch"])
        v_s_log_p = logits.gather(dim, v_s, sampledim)
        return v_s, v_s_log_p

    def log_pv_x(self, e, t):
        query = self.queryproj(ntorch.cat(
            [
                self.lute(e).rename("e", "et"),
                self.lutt(t).rename("t", "et"),
            ],
            "et",
        ))
        log_pv = self.vproj(query).log_softmax("v")
        return log_pv

    def forward(
        self,
        text, text_info,
        x, states, x_info,
        r, r_info, vt,
        ue, ue_info,
        ut, ut_info,
        v2d,
        y, y_info,
        T = None, E = None, R = None,
        learn = False,
    ):
        e,t,v = r

        # posterior
        nll, log_pv_y, attn = self.rnnvie(text, text_info, e, t)
        pv_y = log_pv_y.exp()

        log_pv = self.log_pv_x(e, t)

        pv_y = log_pv_y.exp()
        soft_v = pv_y.dot("v", NamedTensor(self.crnnlm.lutv.weight, names=("v", "r")))

        e = self.crnnlm.lute(e).rename("e", "r")
        t = self.crnnlm.lutt(t).rename("t", "r")
        #v = self.crnnlm.lutv(v).rename("v", "r")

        v_total = r_info.mask.sum()

        # sample from log_pr_x
        v_s, v_s_log_q = self.sample_v(pv_y, log_pv_y, self.K, "v", "k")
        hard_v = self.crnnlm.lutv(v_s).rename("v", "r")

        soft_r = [e.repeat("k", 1),      t.repeat("k", 1),      soft_v.repeat("k", 1)]
        hard_r = [e.repeat("k", self.K), t.repeat("k", self.K), hard_v]

        log_py_Ev, s = self.crnnlm(x, None, x_info, soft_r, r_info, ue, ue_info, ut, ut_info, v2d)
        log_py_v,  s = self.crnnlm(x, None, x_info, hard_r, r_info, ue, ue_info, ut, ut_info, v2d)

        log_py_Ev = log_py_Ev.log_softmax("vocab")
        log_py_v = log_py_v.log_softmax("vocab")

        kl = pv_y * (log_pv_y - log_pv)
        kl_sum = kl[r_info.mask].sum()

        y_mask = y.ne(1)
        nwt = y_mask.sum()

        ll_soft = log_py_Ev.gather("vocab", y.repeat("y", 1), "y").get("y", 0)
        ll_hard = log_py_v.gather("vocab", y.repeat("y", 1), "y").get("y", 0)
        nll_sum = -ll_hard.mean("k")[y_mask].sum()

        reward = (ll_hard.detach() - ll_soft.detach()) * v_s_log_q
        reward_sum = -reward.mean("k")[y_mask].sum()
        #import pdb; pdb.set_trace()

        if learn:
            (nll_sum + kl_sum + reward_sum).div(nwt).backward()

        if kl_sum.item() < 0:
            import pdb; pdb.set_trace()

        rvinfo = RvInfo(
            log_py_v = log_py_v,
            log_py_Ev = log_py_Ev,
            log_pv_y = log_pv_y,
            log_pv = log_pv,
        )
        return (
            rvinfo, s, nll_sum.detach(),
            kl_sum.detach(), kl_sum.clone().fill_(0),
            v_total, nwt,
        )

    def forward_sup(
        self,
        text, text_info,
        x, states, x_info,
        r, r_info, vt,
        ue, ue_info,
        ut, ut_info,
        v2d,
        y, y_info,
        T = None, E = None, R = None,
        learn = False,
    ):
        e,t,v = r

        N = x.shape["batch"]

        # posterior
        nll, log_pv_y, attn = self.rnnvie(text, text_info, e, t, v=v, r_info = r_info, learn=learn)
        pv_y = log_pv_y.exp()

        log_pv = self.log_pv_x(e, t)

        pv_y = log_pv_y.exp()
        soft_v = pv_y.dot("v", NamedTensor(self.crnnlm.lutv.weight, names=("v", "r")))

        nll_pv = -log_pv.gather("v", v.repeat("i", 1), "i").get("i", 0)[r_info.mask].sum()
        nll_qv_y = -log_pv_y.gather("v", v.repeat("i", 1), "i").get("i", 0)[r_info.mask].sum()
        v_total = r_info.mask.sum()

        e = self.crnnlm.lute(e).rename("e", "r")
        t = self.crnnlm.lutt(t).rename("t", "r")
        v = self.crnnlm.lutv(v).rename("v", "r")

        sup_r = [e.repeat("k", 1), t.repeat("k", 1), v.repeat("k", 1)]

        log_py_v,  s = self.crnnlm(x, None, x_info, sup_r, r_info, ue, ue_info, ut, ut_info, v2d)
        log_py_v = log_py_v.log_softmax("vocab")

        y_mask = y.ne(1)
        nwt = y_mask.sum()

        nll_py_v = -log_py_v.gather("vocab", y.repeat("y", 1), "y").get("y", 0)[y_mask].sum()

        if learn:
            # nll_qv_y / v_total is in self.rnnvie
            #((nll_pv + nll_qv_y) / v_total + nll_py_v / nwt).backward()
            (nll_pv / v_total + nll_py_v / nwt).backward()

        rvinfo = RvInfo(
            log_py_v = log_py_v,
            log_pv_y = log_pv_y,
            log_pv = log_pv,
        )
        return rvinfo, s, nll_pv, nll_py_v, nll_qv_y, v_total, nwt

    def init_state(self, N):
        # what's this for?
        if self._N != N:
            self._N = N
            self._state = (
                ntorch.zeros(
                    self.nlayers, N, self.rnn_sz,
                    names=("layers", "batch", "rnns"),
                ).to(self.crnnlm.lutx.weight.device),
                ntorch.zeros(
                    self.nlayers, N, self.rnn_sz,
                    names=("layers", "batch", "rnns"),
                ).to(self.crnnlm.lutx.weight.device),
            )
        return self._state
