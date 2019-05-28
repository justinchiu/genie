from .lvm import Lvm, RvInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import backward 

from namedtensor import ntorch, NamedTensor
from namedtensor.distributions.distributions import NamedDistribution

from .fns import attn, logaddexp


def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")

class CrnnMlmCem(Lvm):
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
        super(CrnnMlmCem, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self._mode = "marginal"
        self._q = "pay"
        self.K = 0

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

        self.dtype = torch.float

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

        # copy projection
        self.Wcopy = ntorch.nn.Linear(
            in_features = rnn_sz,
            out_features = 2,
            bias = False,
        ).spec("rnns", "copy")

        # Tie weights
        if tieweights:
            self.proj.weight = self.lutx.weight

        # Inference network
        self.brnn = ntorch.nn.LSTM(
            input_size = x_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = dropout,
            bidirectional = True,
        ).spec("x", "time", "rnns")
        self.Wi = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("r", "rnns")

        self.type(self.dtype)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # maybe iwae later, but i don't really care about it
        if mode == "elbo":
            # elbo can be exact or approximated through sampling
            assert(self.K >= 0)
        elif mode == "marginal":
            # only exact for marginal
            assert(self.K == 0)
        else:
            raise IndexError("self.mode must be elbo or marginal")
        self._mode = mode

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        assert(q in ["pa", "qay", "pay"])
        self._q = q


    """
    emb_x: T x N x Hx
    r: R x N x Hr
    ctxt: T x N x Hc = f(emb_x, r)
    prior
    a: T x N x R ~ Cat(r^T W ctxt)
    unnormalized likelihood
    y: T x N x V = g(ctxt, attn(a, r))
    """

    def log_py_ac0(self, out, y=None):
        # past: T x N x H?
        # index in with z?
        # r: R x N x H
        log_py_ac0 = self.proj(self.drop(out)).log_softmax("vocab")
        return (
            log_py_ac0
            if y is None else
            log_py_ac0.gather(
                "vocab",
                y.chop("batch", ("lol", "batch"), lol=1),
                "lol",
            )
        ).get("lol", 0)

    def log_py_ac1(self, vt, a_s, y=None):
        # the word values in the cells
        yc = vt.gather("els", a_s, "k")
        py_ac = yc == y
        py_ac = py_ac._new(py_ac.type(self.dtype))
        return py_ac.log()

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


    def log_pc_a(self, rnn_o, rW_a):
        # rnn_o = f(y<t)
        # rW_a = g(r, a) = r[a]
        return self.Wcopy(rnn_o + rW_a).log_softmax("copy")

    # posterior
    def log_qa_y(self, emb_y, r, y_info, r_info):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        log_ea, ea, ec = attn(rnn_o, self.Wi(r), r_info.mask)
        return log_ea, ea, ec

    # not needed?
    def log_py_a(self):
        pass

    def log_py(self, rnn_o, ctxt, log_pa, vt, y):
        log_pc_a = self.log_pc_a(rnn_o, ctxt)
        past = self.Wc(ntorch.cat(
            [
                rnn_o.repeat("els", ctxt.shape["els"]),
                ctxt.repeat("time", rnn_o.shape["time"]),
            ],
            "rnns",
        )).tanh()
        log_py_ac0 = self.log_py_ac0(past, y)
        py_ac1 = vt == y
        py_ac1 = py_ac1._new(py_ac1.type(self.dtype))
        log_py_ac1 = py_ac1.log()

        # p(y|a) = p(y|a,c=0)p(c=0|a) + p(y|a,c=1)p(c=1|a)
        log_py_a = logaddexp(
            log_py_ac0 + log_pc_a.get("copy", 0),
            log_py_ac1 + log_pc_a.get("copy", 1),
        )
        log_pya = log_py_a + log_pa
        return log_pya.logsumexp("els")


    def sample_a(self, probs, logits, K):
        a_s = NamedTensor(torch.multinomial(
            probs.stack(("batch", "time"), "samplebatch").transpose("samplebatch", "els").values,
            K,
            True,
        ), names=("samplebatch", "k")).chop("samplebatch", ("batch", "time"), batch=logits.shape["batch"])
        a_s_log_p = logits.gather("els", a_s, "k")
        return a_s, a_s_log_p

    def forward(self, x, s, x_info, r, r_info, vt, y=None, y_info=None, T=None, E=None, learn=False):
        # shared encoding
        # r: R x N x Er
        # Wa r: R x N x H
        e = self.lute(r[0]).rename("e", "r")
        t = self.lutt(r[1]).rename("t", "r")
        v = self.lutv(r[2]).rename("v", "r")
        # would be nice to have this separately, but i'm afraid it'll take too much mem
        #k = ntorch.cat([e, t], "r")
        r = ntorch.cat([e, t, v], "r")
        rW = self.Wa(r)

        emb_x = self.lutx(x)

        # shared attention
        # This may need to be broken up over time of not enough memory...
        # length stuff might be a bit annoying
        log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, rW, x_info, r_info)

        K = self.K

        # Baseline if sample
        with torch.no_grad():
            past = self.Wc(ntorch.cat([rnn_o, ec], "rnns")).tanh().chop(
                "batch", ("k", "batch"), k=1)
            log_py_Ea = self.log_py_ac0(past, y) if K > 0 else None
            log_py_Eas = log_py_Ea.split(T, "time") if K > 0 else None

        nll = 0
        kl = 0

        log_py = []
        log_py_a = []
        log_pa_y = []
        log_pc_a = []
        log_py_ac0 = []
        log_py_ac1 = []

        # sample only
        a_s = [] if K > 0 else None
        a_s_log_p = [] if K > 0 else None

        rnn_grads = []
        log_pa_grads = []
        rW_grads = []

        nwt = y.ne(1).sum()
        # break up over time
        for t, (rnn_t, log_pa_t, y_t) in enumerate(zip(
            rnn_o.split(T, "time"),
            log_pa.split(T, "time"),
            y.split(T, "time"),
        )):
            # accumulate gradients by hand so we don't need to retain graph
            rnn_t = rnn_t._new(rnn_t.detach().values.requires_grad_(True))
            log_pa_t = log_pa_t._new(log_pa_t.detach().values.requires_grad_(True))
            rW_t = rW._new(rW.detach().values.requires_grad_(True))

            y_maskt = y_t.ne(1)

            # Exact
            # accumulate for queries
            log_pc_a_t = []
            log_py_ac0_t = []
            log_py_ac1_t = []
            log_py_a_t = []
            log_pa_y_t = []

            ctxt = rW_t

            with torch.no_grad():
                log_py_t = NamedTensor(log_pa.values.new([float("-inf")]), names=("els",))
                for e, (ctxt_e, log_pa_e, vt_e) in enumerate(zip(
                    ctxt.split(E, "els"),
                    log_pa_t.split(E, "els"),
                    vt.split(E, "els"),
                )):
                    log_py_e = self.log_py(rnn_t, ctxt_e, log_pa_e, vt_e, y_t)
                    log_py_t = logaddexp(log_py_e, log_py_t)
                log_py.append(log_py_t)

            # break up over els
            rnn_t_grads = []
            ctxt_grads = []
            log_pa_t_grads = []
            for e, (ctxt_e, log_pa_e, vt_e) in enumerate(zip(
                ctxt.split(E, "els"),
                log_pa_t.split(E, "els"),
                vt.split(E, "els"),
            )):
                # accumulate gradients by hand so we don't need to retain graph
                # ctxt_e g> cat e sum t
                # lpa_e g> cat e cat t
                # vt_e g> cat e sum t
                # what about KL?
                rnn_e = rnn_t._new(rnn_t.detach().values.requires_grad_(True))
                ctxt_e = ctxt_e._new(ctxt_e.detach().values.requires_grad_(True))
                log_pa_e = log_pa_e._new(log_pa_e.detach().values.requires_grad_(True))

                log_pc_a_te = self.log_pc_a(rnn_e, ctxt_e)

                past = self.Wc(ntorch.cat(
                    [
                        rnn_e.repeat("els", ctxt_e.shape["els"]),
                        ctxt_e.repeat("time", rnn_t.shape["time"]),
                    ],
                    "rnns",
                )).tanh()
                log_py_ac0_te = self.log_py_ac0(past, y_t)
                py_ac1 = vt_e == y_t
                py_ac1 = py_ac1._new(py_ac1.type(self.dtype))
                log_py_ac1_te = py_ac1.log()

                # p(y|a) = p(y|a,c=0)p(c=0|a) + p(y|a,c=1)p(c=1|a)
                log_py_a_te = logaddexp(
                    log_py_ac0_te + log_pc_a_te.get("copy", 0),
                    log_py_ac1_te + log_pc_a_te.get("copy", 1),
                )
                log_pya_te = log_py_a_te + log_pa_e

                log_pa_y_te = (log_pya_te.detach() - log_py_t)
                pa_y_te = log_pa_y_te.exp()

                nll_e = -pa_y_te * log_pya_te
                nll_e.masked_fill_(log_pya_te == float("-inf"), 0)
                nll_e = nll_e[y_maskt].sum()
                nll += nll_e.detach()
                if learn:
                    nll_e.div(nwt).backward()
                    # accumulate grads
                    # sum over rnn_e
                    rnn_t_grads.append(rnn_e._new(rnn_e.values.grad))
                    # cat over everything else
                    ctxt_grads.append(ctxt_e._new(ctxt_e.values.grad))
                    log_pa_t_grads.append(log_pa_e._new(log_pa_e.values.grad))
                log_pc_a_t.append(log_pc_a_te.detach())
                log_py_ac0_t.append(log_py_ac0_te.detach())
                log_py_ac1_t.append(log_py_ac1_te.detach())
                log_py_a_t.append(log_py_a_te.detach())
                log_pa_y_t.append(log_pa_y_te.detach())
                if self.numexamples == 1:
                    self.numexamples = 0
            # end E-loop if K == 0
            if learn:
                rnn_grads.append(sum(rnn_t_grads))
                log_pa_grads.append(ntorch.cat(log_pa_t_grads, "els"))
                rW_grads.append(ntorch.cat(ctxt_grads, "els"))
            with torch.no_grad():
                log_pc_a.append(ntorch.cat(log_pc_a_t, "els"))
                log_py_ac0.append(ntorch.cat(log_py_ac0_t, "els"))
                log_py_ac1.append(ntorch.cat(log_py_ac1_t, "els"))
                log_py_a.append(ntorch.cat(log_py_a_t, "els"))
                log_pa_y.append(ntorch.cat(log_pa_y_t, "els"))
        # end T-loop

        if learn:
            self.numexamples += 1
            rnn_grads = ntorch.cat(rnn_grads, "time")
            log_pa_grads = ntorch.cat(log_pa_grads, "time")
            rW_grads = sum(rW_grads)
            backward(
                [rnn_o.values, log_pa.values, rW.values],
                [rnn_grads.values, log_pa_grads.values, rW_grads.values]
            )

        if K > 0:
            a_s = ntorch.cat(a_s, "time")
            a_s_log_p = ntorch.cat(a_s_log_p, "time")
        log_pc_a = ntorch.cat(log_pc_a, "time")
        log_py_ac0 = ntorch.cat(log_py_ac0, "time")
        log_py_ac1 = ntorch.cat(log_py_ac1, "time")
        log_py_a = ntorch.cat(log_py_a, "time")
        log_py = ntorch.cat(log_py, "time")
        log_pa_y = ntorch.cat(log_pa_y, "time")

        rvinfo = RvInfo(
            log_py     = log_py,
            log_py_a   = log_py_a,
            log_py_ac0 = log_py_ac0,
            log_py_ac1 = log_py_ac1,
            log_pa     = log_pa,
            log_pa_y   = log_pa_y,
            log_pc_a   = log_pc_a,
            log_pc_y   = None,
            a_s        = a_s,
            a_s_log_p  = a_s_log_p,
            log_py_Ea  = log_py_Ea,
        )
        kl = nll.clone().fill_(0)
        return rvinfo, s, nll, kl


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                NamedTensor(
                    torch.zeros(self.nlayers, N, self.rnn_sz)
                        .type(self.dtype),
                    names=("layers", "batch", "rnns"),
                ).to(self.lutx.weight.device),
                NamedTensor(
                    torch.zeros(self.nlayers, N, self.rnn_sz)
                        .type(self.dtype),
                    names=("layers", "batch", "rnns"),
                ).to(self.lutx.weight.device),
            )
        return self._state


    # too lazy to linearize this
    def marginal_nll(self, x, s, x_info, r, r_info, vt, y, y_info, E=32, T=129, learn=False):
        assert(learn == False)
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
        K = self.K

        """
        if y is not None:
            emb_y = self.lutx(y)
            log_pa_y, pay, eyc = self.pa_y(emb_y, r, x_info, r_info)
        else:
            log_pa_y, pay = None, None
        """

        ctxt = rW
        lyas = []
        # sum_Ai sum_Cj [ sum_{a in Ai} p(a) sum_{c in Ci} p(y|a,c)p(c|a) ]
        for rnn_t, lpa_t, y_t in zip(
            rnn_o.split(T, "time"),
            log_pa.split(T, "time"),
            y.split(T, "time"),
        ):
            lyas_t = []
            for ctxt_e, lpa_e, vt_e in zip(
                ctxt.split(E, "els"),
                lpa_t.split(E, "els"),
                vt.split(E, "els"),
            ):
                out = self.Wc(ntorch.cat(
                    [
                        rnn_t.expand("els", ctxt_e.shape["els"]),
                        ctxt_e.expand("time", rnn_t.shape["time"])
                    ],
                    "rnns",
                )).tanh()
                # is this right
                #import pdb; pdb.set_trace()
                log_pc_a = self.log_pc_a(rnn_t, ctxt_e)
                log_pc0_a = log_pc_a.get("copy", 0)
                log_pc1_a = log_pc_a.get("copy", 1)
                # TODO: Need to generalize conditional dists.
                # log p(y|a)
                log_py_ac0 = (self.proj(out)
                    .log_softmax("vocab")
                    .gather("vocab", y_t.chop("batch", ("lol", "batch"), lol=1), "lol")
                ).get("lol", 0)
                #log_py_ac1 = (vt_e == y_t).float().log()
                py_ac1 = vt_e == y_t
                py_ac1 = py_ac1._new(py_ac1.type(self.dtype))
                log_py_ac1 = py_ac1.log()

                log_py_a = logaddexp(
                    log_py_ac0 + log_pc0_a,
                    log_py_ac1 + log_pc1_a,
                )
                # log p(y|a,c=0)
                log_pya = (log_py_a + lpa_e)
                lyas_t.append(log_pya.logsumexp("els"))
            lyas.append(ntorch.stack(lyas_t, "els").logsumexp("els"))
        # log p(y)
        log_py = ntorch.cat(lyas, "time")
        rvinfo = RvInfo(
            log_py = log_py,
        )       # need E log p(y|a)??
        return rvinfo, s


    # move to lvm.py? maybe later
    def kl(self, qa, log_qa, log_pa, lens):
        kl = []
        for i, l in enumerate(lens.tolist()):
            qa0 = qa.get("batch", i).narrow("els", 0, l)
            log_qa0 = log_qa.get("batch", i).narrow("els", 0, l)
            log_pa0 = log_pa.get("batch", i).narrow("els", 0, l)
            kl0 =  qa0 * (log_qa0 - log_pa0)
            infmask = log_qa0 != float("-inf")
            # workaround for namedtensor bug that puts empty tensors on different devices
            kl0 = kl0._new(kl0.values.where(infmask.values, torch.zeros_like(kl0.values))).sum("els")
            kl.append(kl0)
        return ntorch.stack(kl, "batch")

