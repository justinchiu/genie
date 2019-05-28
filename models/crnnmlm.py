from .lvm import Lvm, RvInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from namedtensor import ntorch, NamedTensor
from namedtensor.distributions.distributions import NamedDistribution

from .fns import attn

def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")

class MixInfo(RvInfo):
    pass

class CrnnMlm(Lvm):
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
    ):
        super(CrnnMlm, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self.K = 1

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
            dropout = 0.3,
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

        # Inference network
        self.brnn = ntorch.nn.LSTM(
            input_size = x_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = 0.3,
            bidirectional = True,
        ).spec("x", "time", "rnns")
        self.Wi = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("r", "rnns")


    def query(self):
        pass

    """
    emb_x: T x N x Hx
    r: R x N x Hr
    ctxt: T x N x Hc = f(emb_x, r)
    prior
    a: T x N x R ~ Cat(r^T W ctxt)
    unnormalized likelihood
    y: T x N x V = g(ctxt, attn(a, r))
    """

    # likelihood
    def log_py_a(self, past, ctxt, x_info, r_info, y=None):
        # past: T x N x H?
        # index in with z?
        # r: R x N x H
        out = self.Wc(ntorch.cat(
            [
                past.repeat("k", 2),
                ctxt,
            ],
            "rnns",
        )).tanh()
        log_py_a = self.proj(self.drop(out)).log_softmax("vocab")
        return (
            log_py_a
            if y is None else
            log_py_a.gather(
                "vocab",
                y.chop("batch", ("lol", "batch"), lol=1),
                "lol",
            )
        ).get("lol", 0)

    def pa0(self, emb_x, s, rW, x_info, r_info):
        T = emb_x.shape["time"]
        N = emb_x.shape["batch"]
        R = rW.shape["els"]
        ea, ec, output = None, None, None
        if not self.inputfeed:
            rnn_o, s = self.rnn(emb_x, s, lengths=x_info.lengths)
            # ea: T x N x R
            # ec: T x N x H
            log_ea, ea, ec = attn(rnn_o, rW, r_info.mask)
            output = rnn_o
        else:
            raise NotImplementedError
            log_ea = []
            ea = []
            ec = []
            output = []
            ect = torch.zeros(N, self.r_emb_sz).to(emb_x.device)
            for t in range(T):
                inp = torch.cat([emb_x[t], ect], dim=-1)
                rnn_o, s = self.rnn(inp.unsqueeze(0), s)
                rnn_o = rnn_o.squeeze(0)
                log_eat, eat, ect = attn(rnn_o, r, lenr)
                log_ea.append(log_eat)
                ea.append(eat)
                ec.append(ect)
                output.append(rnn_o)
            log_ea = torch.stack(log_ea, 0)
            ea = torch.stack(ea, 0)
            ec = torch.stack(ec, 0)
            output = torch.stack(output, 0)
        return log_ea, ea, ec, output, s

    # posterior
    def pay(self, emb_y, r, y_info, r_info):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        log_ea, ea, ec = attn(rnn_o, self.Wi(r), r_info.mask)
        return log_ea, ea, ec

    def forward(self, x, s, x_info, r, r_info, vt, y=None, y_info=None):
        e = self.lute(r[0])
        t = self.lutt(r[1])
        v = self.lutv(r[2])

        # r: R x N x Er
        # Wa r: R x N x H
        e = self.lute(r[0]).rename("e", "r")
        t = self.lutt(r[1]).rename("t", "r")
        v = self.lutv(r[2]).rename("v", "r")
        #r = self.Wa(ntorch.cat([e, t, v], "r").tanh())
        r = ntorch.cat([e, t, v], "r")
        rW = self.Wa(r)

        emb_x = self.lutx(x)

        log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, rW, x_info, r_info)

        # what should we do with pa, ec, and rnn_o?
        # ec is only used for a baseline
        R, N, H = r.shape
        T = x.shape["time"]
        K = self.K

        if y is not None:
            emb_y = self.lutx(y)
            log_pa_y, pa_y, eyc = self.pay(emb_y, r, x_info, r_info)
        else:
            log_pa_y, pa_y = None, None

        dist = (
            ntorch.distributions.Categorical(
                logits=log_pa,
                batch_names = ("batch", "time"),
                param_names = ("els",),
            ) if y is None
            else ntorch.distributions.Categorical(
                logits=log_pa_y,
                batch_names = ("batch", "time"),
                param_names = ("els",),
            )
        )
        # First dimension should be number of samples
        # K x T x N
        a_s = dist.sample((K,), ("k",))
        a_s_log_p = (
            (log_pa if y is None else log_pa_y)
            .gather("els", a_s, "k")
        )

        ctxt = rW.gather("els", a_s, "k")
        # add baseline
        ctxt = ntorch.cat([ctxt, ec.chop("batch", ("k", "batch"), k=1)], "k")

        log_py_as = self.log_py_a(rnn_o, ctxt, x_info, r_info, y)
        log_py_Ea = log_py_as.get("k", -1)
        log_py_a = log_py_as.narrow("k", 0, self.K)

        rvinfo = RvInfo(
            log_py = None,
            log_py_a = log_py_a,
            log_pa = log_pa,
            log_pa_y = log_pa_y,
            a_s = a_s,
            a_s_log_p = a_s_log_p,
            log_py_Ea  = log_py_Ea,
        )
        return rvinfo, s



    def init_state(self, N):
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


    def marginal_nll(self, x, s, x_info, r, r_info, vt, y, y_info, size=32):
        e = self.lute(r[0])
        t = self.lutt(r[1])
        v = self.lutv(r[2])

        # r: R x N x Er
        # Wa r: R x N x H
        e = self.lute(r[0]).rename("e", "r")
        t = self.lutt(r[1]).rename("t", "r")
        v = self.lutv(r[2]).rename("v", "r")
        #r = self.Wa(ntorch.cat([e, t, v], "r").tanh())
        r = ntorch.cat([e, t, v], "r")
        rW = self.Wa(r)

        emb_x = self.lutx(x)

        log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, rW, x_info, r_info)

        # what should we do with pa, ec, and rnn_o?
        # ec is only used for a baseline
        R, N, H = r.shape
        T = x.shape["time"]
        K = self.K

        if y is not None:
            emb_y = self.lutx(y)
            log_pay, pay, eyc = self.pay(emb_y, r, x_info, r_info)
        else:
            log_pay, pay = None, None

        ctxt = rW
        lyas = []
        Tsplit = 128
        for rnn_t, lpa_t, y_t in zip(rnn_o.split(Tsplit, "time"), log_pa.split(Tsplit, "time"), y.split(Tsplit, "time")):
            lyas_t = []
            for ctxt_e, lpa_e in zip(ctxt.split(size, "els"), lpa_t.split(size, "els")):
                out = self.Wc(ntorch.cat(
                    [
                        rnn_t.expand("els", ctxt_e.shape["els"]),
                        ctxt_e.expand("time", rnn_t.shape["time"])
                    ],
                    "rnns",
                )).tanh()
                # log p(y|a)
                log_py_a = (self.proj(out)
                    .log_softmax("vocab")
                    .gather("vocab", y_t.chop("batch", ("lol", "batch"), lol=1), "lol")
                ).get("lol", 0)
                # log p(y,a)
                # would be nicer with keepdim
                lya = (log_py_a + lpa_e).logsumexp("els")
                lyas_t.append(lya)
            lyas.append(ntorch.stack(lyas_t, "els").logsumexp("els"))
            #import pdb; pdb.set_trace()
        # log p(y)
        log_py = ntorch.cat(lyas, "time")
        # need E log p(y|a)??
        rvinfo = RvInfo(
            log_py = log_py,
        )       # need E log p(y|a)??
        return rvinfo, s
