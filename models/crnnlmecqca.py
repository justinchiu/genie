import math

from .lvma import LvmA, RvInfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import backward 

from namedtensor import ntorch, NamedTensor
from namedtensor.distributions.distributions import NamedDistribution

from .fns import attn, logaddexp, attn2, concrete_attn


def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")

class CrnnLmEcqca(LvmA):
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
        noattnvalues = False,
        nuisance = False,
        jointcopy = False,
        qc = True,
        qcrnn = True,
        temp = 1,
        qconly = False,
        v2d = False,
        initvy = False,
        glove = False,
        initu = False,
        tanh = False,
        vctxt = False,
        wcv = False,
        etvctxt = False,
        bil = False,
        mlp = False,
        untie = False,
    ):
        super(CrnnLmEcqca, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self._mode = "elbo"
        self._q = "qay"
        self.K = 0
        self.Ke = 0
        self.Kl = 0

        # should be:
        # self.nokl = not qconly
        self.nokl = False
        self.v2d = v2d
        self.initvy = initvy
        self.glove = glove
        self.initu = initu
        self.tanh = tanh
        self.vctxt = vctxt
        self.wcv = wcv
        self.etvctxt = etvctxt
        self.bil = bil
        self.mlp = mlp
        self.untie = untie

        self.noattn = False
        self.noattnvalues = noattnvalues
        self.nuisance = nuisance
        self.jointcopy = jointcopy
        self.qc = qc
        self.qcrnn = qcrnn
        self.qconly = qconly

        self.temp = temp

        self.numexamples = 0

        self.kl_anneal_steps = 0
        self.c_anneal_steps = 0
        self.c_warmup_steps = 0
        self.steps = 0
        self.weightannealsteps = None
        self._annealed = None

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
        if self.initu:
            self.lute.weight.data.uniform_(-0.1, 0.1)
        self.lute.weight.data[Ve.stoi[self.PAD]].fill_(0)
        self.lutt = ntorch.nn.Embedding(
            num_embeddings = len(Vt),
            embedding_dim = r_emb_sz,
            padding_idx = Vt.stoi[self.PAD],
        ).spec("els", "t")
        if self.initu:
            self.lutt.weight.data.uniform_(-0.1, 0.1)
        self.lutt.weight.data[Vt.stoi[self.PAD]].fill_(0)
        self.lutv = ntorch.nn.Embedding(
            num_embeddings = len(Vv),
            embedding_dim = r_emb_sz,
            padding_idx = Vv.stoi[self.PAD],
        ).spec("els", "v")
        if self.initu:
            self.lutv.weight.data.uniform_(-0.1, 0.1)
        self.lutv.weight.data[Vv.stoi[self.PAD]].fill_(0)
        self.lutx = ntorch.nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        ).spec("time", "x")
        if self.initu:
            self.lutx.weight.data.uniform_(-0.1, 0.1)
        self.lutx.weight.data[Vx.stoi[self.PAD]].fill_(0)
        if self.glove or self.untie:
            self.lutgx = ntorch.nn.Embedding(
                num_embeddings = len(Vx),
                embedding_dim = x_emb_sz,
                padding_idx = Vx.stoi[self.PAD],
            ).spec("time", "x")
            if self.initu:
                self.lutgx.weight.data.uniform_(-0.1, 0.1)
            if self.Vx.vectors is not None and self.glove:
                self.lutgx.weight.data.masked_scatter_(
                    self.Vx.vectors.ne(0),
                    self.Vx.vectors,
                )
                self.lutgx.weight.requires_grad = False
            self.lutgx.weight.data[Vx.stoi[self.PAD]].fill_(0)
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
        if self.initu:
            for name, param in self.rnn.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param.data.uniform_(-0.1, 0.1)
        self.drop = ntorch.nn.Dropout(dropout)
        self.proj = ntorch.nn.Linear(
            in_features = rnn_sz,
            out_features = len(Vx),
            bias = False,
        ).spec("ctxt", "vocab")
        if self.glove or self.untie:
            self.gproj = ntorch.nn.Linear(
                in_features = rnn_sz,
                out_features = len(Vx),
                bias = False,
            ).spec("ctxt", "vocab")
        self.vproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vv),
            bias = False,
        ).spec("rnns", "v")

        # attn projection
        self.Wae = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("e", "rnns")
        self.Wat = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("t", "rnns")
        self.War = ntorch.nn.Linear(
            in_features = 3*r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("r", "rnns")

        # context projection
        self.Wc = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")
        self.Wc_nov = ntorch.nn.Linear(
            in_features = 2*r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")
        self.Wcv = ntorch.nn.Linear(
            in_features = r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "ctxt")

        if self.bil:
            self.Wvy = ntorch.nn.Linear(
                in_features = rnn_sz,
                out_features = rnn_sz,
                bias = False
            ).spec("ctxt", "ctxt")
            self.Wvy.weight.data.copy_(torch.eye(rnn_sz))
        if self.mlp:
            self.Wvy0 = ntorch.nn.Linear(
                in_features = rnn_sz,
                out_features = rnn_sz,
                bias = False
            ).spec("ctxt", "ctxt")
            self.Wvy1 = ntorch.nn.Linear(
                in_features = rnn_sz,
                out_features = rnn_sz,
                bias = False
            ).spec("ctxt", "ctxt")

        # copy projection
        self.Wcopy = ntorch.nn.Linear(
            in_features = rnn_sz,
            out_features = 2,
            bias = False,
        ).spec("rnns", "copy")

        # inputfeeding
        self.Wif = ntorch.nn.Linear(
            in_features = 3 * r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        ).spec("rnns", "rnns")

        # Tie weights
        if tieweights:
            self.proj.weight = self.lutx.weight
            if self.glove:
                self.gproj.weight = self.lutgx.weight
        # initialize lutv to lutx
        if self.initvy:
            self.lutv.weight.data.copy_(
                self.lutx.weight.data.index_select(
                    0,
                    torch.LongTensor([self.Vx.stoi[word] for word in self.Vv.itos]),
                )
            )

        # Inference network
        self.brnn = ntorch.nn.LSTM(
            input_size = x_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = dropout,
            bidirectional = True,
        ).spec("x", "time", "rnns")
        if self.qcrnn:
            self.brnnc = ntorch.nn.LSTM(
                input_size = x_emb_sz,
                hidden_size = rnn_sz,
                num_layers = nlayers,
                bias = False,
                dropout = dropout,
                bidirectional = True,
            ).spec("x", "time", "rnns")
        self.Wicopy = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = 2,
            bias = False,
        ).spec("rnns", "copy")
        self.Wie = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("e", "rnns")
        self.Wit = ntorch.nn.Linear(
            in_features = r_emb_sz,
            out_features = 2*rnn_sz,
            bias = False,
        ).spec("t", "rnns")

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

    @property
    def annealed(self):
        return self._annealed

    @annealed.setter
    def annealed(self, val):
        if not self._annealed:
            self._annealed = val
            self.annealstep = self.steps
            print("SETTING MODEL AS ANNEALED")
        #else:
            #raise NotImplementedError


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
                y.repeat("lol", 1),
                "lol",
            ).get("lol", 0)
        )

    def log_py_ac1(self, vt, a_s, y=None):
        raise NotImplementedError
        # the word values in the cells
        yc = vt.gather("els", a_s, "k")
        py_ac = yc == y
        py_ac = py_ac._new(py_ac.type(self.dtype))
        return py_ac.log()

    # maybe change pa0 to not use v2dx
    # so the model *must* copy to look at values
    # but only after doing some analysis
    def pa0(self, emb_x, s, x_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        T = emb_x.shape["time"]
        N = emb_x.shape["batch"]

        log_ea, ea, ec = None, None, None
        log_ta, ta, tc = None, None, None
        log_a, a, c = None, None, None
        output = None

        if not self.inputfeed:
            # rnn_o: T x N x H
            rnn_o, s = self.rnn(emb_x, s, x_info.lengths)
            # ea: T x N x R
            log_ea, ea, ec = attn(rnn_o, emb_e, ue_info.mask)
            #log_t, ea_T, ec_T = attn(rnn_o + ec_E, tA, ut_info.mask)
            log_ta, ta, tc = attn(rnn_o, emb_t, ut_info.mask)
            if self.noattn:
                ec = r.mean("els").repeat("time", ec.shape["time"])
            log_ea = log_ea.rename("els", "e")
            log_ta = log_ta.rename("els", "t")
            log_va = log_ea + log_ta
            vc = log_va.exp().dot(("t", "e"), v2dx)
            va = log_va.exp()

            ea = ea.rename("els", "e")
            ta = ta.rename("els", "t")

            output = rnn_o
        else:
            log_ea, ea, ec = [], [], []
            log_ta, ta, tc = [], [], []
            log_va, va, vc = [], [], []
            out = []
            etc_t = ntorch.zeros(
                N, self.r_emb_sz, names=("batch", "rnns")
            ).to(emb_x.values.device)
            for t in range(T):
                etc_t = etc_t.rename("rnns", "x")
                inp = ntorch.cat([emb_x.get("time", t), etc_t], "x").repeat("time", 1)
                rnn_o, s = self.rnn(inp, s)
                rnn_o = rnn_o.get("time", 0)
                log_ea_t, ea_t, ec_t = attn(rnn_o, emb_e, ue_info.mask)
                log_ta_t, ta_t, tc_t = attn(rnn_o, emb_t, ut_info.mask)
                log_ea_t = log_ea_t.rename("els", "e")
                log_ta_t = log_ta_t.rename("els", "t")
                log_va_t = log_ea_t + log_ta_t
                va_t = log_va_t.exp()
                vc_t = va_t.dot(("t", "e"), v2dx)
                out.append(
                    self.Wif(ntorch.cat([rnn_o, vc_t, ec_t, tc_t], "rnns"))
                )

                log_ea.append(log_ea_t)
                ea.append(ea_t)
                ec.append(ec_t)
                log_ta.append(log_ta_t)
                ta.append(ta_t)
                tc.append(tc_t)
                log_va.append(log_va_t)
                va.append(va_t)
                vc.append(vc_t)

            output = ntorch.stack(out, "time")

            log_ea = ntorch.stack(log_ea, "time")
            ea = ntorch.stack(ea, "time")
            ec = ntorch.stack(ec, "time")
            log_ta = ntorch.stack(log_ta, "time")
            ta = ntorch.stack(ta, "time")
            tc = ntorch.stack(tc, "time")
            log_va = ntorch.stack(log_va, "time")
            va = ntorch.stack(va, "time")
            vc = ntorch.stack(vc, "time")

            ea = ea.rename("els", "e")
            ta = ta.rename("els", "t")

        return log_ea, ea, ec, log_ta, ta, tc, log_va, va, vc, output, s

    """
    def log_pc_a(self, rnn_o, rW_a):
        # rnn_o = f(y<t)
        # rW_a = g(r, a) = r[a]
        return self.Wcopy(rnn_o + rW_a).log_softmax("copy")
    """

    def log_alpha_pc(self, rnn_o, ec, tc, vc):
        # rnn_o = f(y<t)
        # condition on soft attention
        # rW_a = g(r, a) = r[a]
        # already concatenated if input feed
        return self.Wcopy(rnn_o)
        """
        return self.Wcopy(
            rnn_o if self.inputfeed else
            self.Wif(ntorch.cat([rnn_o, ec, tc, vc], "rnns"))
        ).log_softmax("copy")
        """

    # posterior
    def log_qac_y(self, emb_y, y_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        # ea: T x N x R
        log_ea, ea, ec = attn(rnn_o, self.Wie(emb_e), ue_info.mask, self.temp)
        log_ta, ta, tc = attn(rnn_o, self.Wit(emb_t), ut_info.mask, self.temp)
        log_ea = log_ea.rename("els", "e")
        log_ta = log_ta.rename("els", "t")
        log_va = log_ea + log_ta
        if self.qcrnn:
            rnn_oc, _ = self.brnnc(emb_y.detach(), None, lengths=y_info.lengths) 
        log_alpha_c = self.Wicopy(rnn_oc if self.qcrnn else rnn_o)
        # gumbel stuff
        eps = torch.finfo(log_alpha_c.values.dtype).eps
        shape = log_alpha_c.shape
        u = ntorch.rand(
            (1,) + tuple(shape.values()),
            names = ("k",) + tuple(shape.keys()),
            device = log_alpha_c.values.device
        ).clamp(min=eps, max=1-eps)
        g = -(-u.log()).log()
        logits = (log_alpha_c + g) / self.qtao
        log_c = logits.log_softmax("copy")

        return log_c, log_alpha_c, log_ea, log_ta, log_va

    # bayes rule
    def log_qacv_y(self, emb_y, y_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        # ea: T x N x R
        log_ea, ea, ec = attn(rnn_o, self.Wie(emb_e), ue_info.mask, self.temp)
        log_ta, ta, tc = attn(rnn_o, self.Wit(emb_t), ut_info.mask, self.temp)
        log_ea = log_ea.rename("els", "e")
        log_ta = log_ta.rename("els", "t")
        log_va = log_ea + log_ta

        if self.qcrnn:
            rnn_oc, _ = self.brnnc(emb_y.detach(), None, lengths=y_info.lengths) 
        log_c = self.Wicopy(rnn_oc if self.qcrnn else rnn_o).log_softmax("copy") 

        rnn_ov, _ = self.brnnv(emb_y.detach(), None, lengths=y_info.lengths)
        log_v = self.vproj(rnn_ov).log_softmax("v")

        return log_c, log_ea, log_ta, log_va, log_v

    def log_qc_y(self, emb_y, x_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        # ea: T x N x R
        log_ea, ea, ec = attn(rnn_o, emb_e, ue_info.mask)
        log_ta, ta, tc = attn(rnn_o, emb_t, ut_info.mask)
        log_ea = log_ea.rename("els", "e")
        log_ta = log_ta.rename("els", "t")
        log_va = log_ea + log_ta
        #vc = log_va.exp().dot(("t", "e"), v2dx)
        #va = log_va.exp()

        ea = ea.rename("els", "e")
        ta = ta.rename("els", "t")

        return log_ea, log_ta, log_va


    # not needed?
    def log_py_a(self):
        pass

    def log_py(self, rnn_o, ctxt, log_pa, vt, y):
        raise NotImplementedError
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


    def sample_a(self, probs, logits, K, dim, sampledim):
        a_s = NamedTensor(torch.multinomial(
            probs.stack(("batch", "time"), "samplebatch").transpose("samplebatch", dim).values,
            K,
            True,
        ), names=("samplebatch", sampledim)).chop("samplebatch", ("batch", "time"), batch=logits.shape["batch"])
        a_s_log_p = logits.gather(dim, a_s, sampledim)
        return a_s, a_s_log_p

    def forward(
        self, x, s, x_info, r, r_info, vt,
        ue, ue_info,
        ut, ut_info,
        v2d, vt2d,
        y=None, y_info=None,
        T=None, E=None, R=None,
        learn=False,
        supattn=False,
    ):
        # shared encoding
        # r: R x N x Er
        # Wa r: R x N x H
        e = self.lute(r[0]).rename("e", "r")
        #t = self.lutt(r[1]).rename("t", "r")
        #v = self.lutv(r[2]).rename("v", "r")

        #r = ntorch.cat([e, t, v], "r")
        #rW = self.Wa(r)

        emb_x = self.lutx(x)

        emb_e = self.lute(ue)
        emb_t = self.lutt(ut)
        eA = self.Wae(emb_e)
        tA = self.Wat(emb_t)

        if self.v2d:
            v2dx = self.lutv(
                v2d.stack(("t", "e"), "els")
            ).chop("els", ("t", "e"), t=v2d.shape["t"]).rename("v", "rnns")
        else:
            # vt2dx
            v2dx = self.lutx(
                vt2d.stack(("t", "e"), "time")
            ).chop("time", ("t", "e"), t=v2d.shape["t"]).rename("x", "rnns")

        v2dgx = (
            self.lutgx(
                vt2d.stack(("t", "e"), "time")
            )
                .chop("time", ("t", "e"), t=v2d.shape["t"])
                .rename("x", "rnns") if self.glove or self.untie else None
        )

        # shared attention
        # This may need to be broken up over time of not enough memory...
        # length stuff might be a bit annoying
        #log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, x_info, r_info)
        log_pe, pe, ec, log_pt, pt, tc, log_pv, pv, vc, rnn_o, s = self.pa0(
            emb_x, s, x_info, eA, ue_info, tA, ut_info, v2dx)

        # use soft attention over everything for p(c | y_<-t)
        # maybe no vc?
        log_alpha_pc = self.log_alpha_pc(rnn_o, ec, tc, vc)

        # anneal vc to 0
        if self.c_anneal_steps > 0 and self.steps > self.c_warmup_steps:
            wc = self.weight_coef(self.steps - self.c_warmup_steps, self.c_anneal_steps)
            vc = (1-wc) * vc

        K = self.K

        # Baseline if sample
        #past = self.Wc(ntorch.cat([rnn_o, vc], "rnns")).tanh().chop(
            #"batch", ("k", "batch"), k=1)
        if self.noattnvalues:
            past = self.Wc_nov(ntorch.cat([rnn_o, ec, tc], "rnns")).repeat("k", 1)
            if self.tanh:
                past = past.tanh()
        elif self.nuisance:
            past = rnn_o.repeat("k", 1).rename("rnns", "ctxt")
        elif self.wcv:
            past = self.Wcv(ntorch.cat([rnn_o, ec + tc + vc], "rnns")).repeat("k", 1)
            if self.tanh:
                past = past.tanh()
        else:
            past = self.Wc(ntorch.cat([rnn_o, ec, tc, vc], "rnns")).repeat("k", 1)
            if self.tanh:
                past = past.tanh()
        log_py_Ea = self.log_py_ac0(past, y)

        if y is not None:
            emb_y = self.lutx(y)
            log_qc, log_alpha_qc, log_qe, log_qt, log_qv = self.log_qac_y(
                emb_y, x_info, emb_e, ue_info, emb_t, ut_info, v2dx)

        nll = 0.
        kl = 0.
        klc = 0.

        log_py_a = []
        log_pa_y = []
        log_py_c0 = []
        log_py_ac1 = []

        # sample only
        v_s = [] if K > 0 else None
        v_s_log_qv = [] if K > 0 else None
        e_s = [] if K > 0 else None
        e_s_log_qe = [] if K > 0 else None
        t_s = [] if K > 0 else None
        t_s_log_qt = [] if K > 0 else None


        # cat over time
        rnn_grads = []
        log_py_Ea_grads = []

        # cat over time
        log_pc_grads = []
        log_pe_grads = []
        log_pt_grads = []
        log_qc_grads = []
        log_qe_grads = []
        log_qt_grads = []
        log_qv_y_grads = []
        c_log_qc_grads = []

        # sum over time
        v2dx_grads = []
        v2dgx_grads = []

        nwt = y.ne(1).sum().float() if y is not None else 1
        # break up over time
        # pick and use either e and t or v...
        for t, (
            rnn_t,
            log_py_Ea_t,
            log_pc_t, log_pe_t, log_pt_t,
            log_qc_t, log_qe_t, log_qt_t,
            c_log_qc_t,
            #log_qv_y_t,
            y_t,
        ) in enumerate(zip(
            rnn_o.split(T, "time"),
            log_py_Ea.split(T, "time"),
            log_alpha_pc.split(T, "time"),
            log_pe.split(T, "time"),
            log_pt.split(T, "time"),
            log_alpha_qc.split(T, "time"),
            log_qe.split(T, "time"),
            log_qt.split(T, "time"),
            log_qc.split(T, "time"),
            #log_qv_y.split(T, "time"),
            (y if y is not None else ec).split(T, "time"),
        )):
            # accumulate gradients by hand so we don't need to retain graph
            rnn_t = rnn_t._new(rnn_t.detach().values.requires_grad_(True))
            log_py_Ea_t = log_py_Ea_t._new(log_py_Ea_t.detach().values.requires_grad_(True))
            log_pc_t = log_pc_t._new(log_pc_t.detach().values.requires_grad_(True))
            log_pe_t = log_pe_t._new(log_pe_t.detach().values.requires_grad_(True))
            log_pt_t = log_pt_t._new(log_pt_t.detach().values.requires_grad_(True))
            log_qc_t = log_qc_t._new(log_qc_t.detach().values.requires_grad_(True))
            log_qe_t = log_qe_t._new(log_qe_t.detach().values.requires_grad_(True))
            log_qt_t = log_qt_t._new(log_qt_t.detach().values.requires_grad_(True))
            #log_qv_y_t = log_qv_y_t._new(log_qv_y_t.detach().values.requires_grad_(True))
            c_log_qc_t = c_log_qc_t._new(c_log_qc_t.detach().values.requires_grad_(True))
            v2dx_t = v2dx._new(v2dx.detach().values.requires_grad_(True))
            if v2dgx is not None:
                v2dgx_t = v2dgx._new(v2dx.detach().values.requires_grad_(True))

            y_maskt = y_t.ne(1)

            # just sample now
            if K > 0:
                qc_t = c_log_qc_t.exp()
                qe_t = log_qe_t.exp()
                qt_t = log_qt_t.exp()

                # actually, take top k from product dist
                # then sample from complement
                log_qv_t = (log_qe_t + log_qt_t).stack(("e", "t"), "v")
                log_pv_t = (log_pe_t + log_pt_t).stack(("e", "t"), "v")

                # for baseline
                if self.Kl > 0:
                    B_v_s_t, B_v_s_log_qv = self.sample_a(log_qv_t.exp(), log_qv_t, self.Kl, "v", "k")
                    B_v_s_log_pv = log_pv_t.gather("v", B_v_s_t, "k")

                C_v_s_log_qv, C_v_s_t = log_qv_t.topk("v", self.Ke)
                C_v_s_log_pv = log_pv_t.gather("v", C_v_s_t, "v")
                C_v_s_log_qv = C_v_s_log_qv.rename("v", "k")
                C_v_s_log_pv = C_v_s_log_pv.rename("v", "k")
                C_v_s_t = C_v_s_t.rename("v", "k")

                # weight nll by this
                C_v_s_qv = C_v_s_log_qv.exp()
                C_Zv = C_v_s_log_qv.logsumexp("k").exp()

                nC_log_qv = log_qv_t.clone()
                nC_log_qv.scatter_("v", C_v_s_t, C_v_s_log_qv.clone().fill_(float("-inf")), "k")
                # reciprocal of normalizing constant, multiply MC term by this
                nC_Zv = nC_log_qv.logsumexp("v").exp()
                nC_qv = nC_log_qv.softmax("v")

                nC_v_s_t, nC_v_s_log_qv = self.sample_a(nC_qv, log_qv_t, self.K - self.Ke, "v", "k")
                nC_v_s_log_pv = log_pv_t.gather("v", nC_v_s_t, "k")
                nC_v_s_qv = nC_v_s_log_qv.exp()

                # sampled and flatten values at timestep t
                v2dx_t_flat = (v2dx_t if not (self.glove or self.untie) else v2dgx_t).stack(("e", "t"), "r")

                if self.Kl > 0:
                    B_v_f_t = v2dx_t_flat.gather("r", B_v_s_t.rename("v", "k"), "k")
                C_v_f_t = v2dx_t_flat.gather("r", C_v_s_t.rename("v", "k"), "k")
                nC_v_f_t = v2dx_t_flat.gather("r", nC_v_s_t, "k")

                if self.Kl > 0:
                    v_f_t = ntorch.cat([B_v_f_t, C_v_f_t, nC_v_f_t], "k")
                    v_s_log_qv_t = ntorch.cat([B_v_s_log_qv, C_v_s_log_qv, nC_v_s_log_qv], "k")
                    v_s_log_pv_t = ntorch.cat([B_v_s_log_pv, C_v_s_log_pv, nC_v_s_log_pv], "k")
                else:
                    v_f_t = ntorch.cat([C_v_f_t, nC_v_f_t], "k")
                    v_s_log_qv_t = ntorch.cat([C_v_s_log_qv, nC_v_s_log_qv], "k")
                    v_s_log_pv_t = ntorch.cat([C_v_s_log_pv, nC_v_s_log_pv], "k")

                # for weighting only, detach before using in all cases
                # \sum_v q(v) g(v) + q(nC)E_{v~nC}[g(v)]
                v_s_qv_t = ntorch.cat([C_v_s_qv, nC_Zv.repeat("k", self.K - self.Ke) / (self.K - self.Ke)], "k")

                ctxt = v_f_t

                # TODO, try concatenating entity and type with value
                past = self.Wcv(ntorch.cat(
                    [
                        rnn_t.repeat("k", ctxt.shape["k"]),
                        ctxt,
                    ],
                    "rnns",
                )).tanh() if self.vctxt else ctxt.rename("rnns", "ctxt")
                if self.bil:
                    past = self.Wvy(past)
                elif self.mlp:
                    past = self.Wvy1(self.Wvy0(past).tanh()).tanh()

                log_py_ac = self.proj(
                    qc_t.get("copy", 0) * rnn_t.rename("rnns", "ctxt")
                    + qc_t.get("copy", 1) * past,
                ).log_softmax("vocab").gather("vocab", y_t.repeat("i", 1), "i").get("i", 0)

                delta_p = log_pc_t - self.tao * c_log_qc_t
                lpc_t = math.log(self.tao) + delta_p.sum("copy") - 2 * delta_p.logsumexp("copy")
                delta_q = log_qc_t - self.qtao * c_log_qc_t
                lqc_t = math.log(self.qtao) + delta_q.sum("copy") - 2 * delta_q.logsumexp("copy")

                log_ratio = lpc_t - lqc_t

                log_py_ac = log_py_ac + log_ratio

                if self.Kl > 0:
                    Bt = log_py_ac.narrow("k", 0, self.Kl).mean("k")
                    log_py_z = log_py_ac.narrow("k", self.Kl, self.K)
                else:
                    log_py_z = log_py_ac
                nll_t = -(log_py_z * v_s_qv_t.detach()).sum("k")[y_maskt].sum()

                kl_e_t = self.kl(
                    qe_t,
                    log_qe_t, log_pe_t,
                    ue_info.lengths, "e",
                ).sum()
                kl_t_t = self.kl(
                    qt_t,
                    log_qt_t, log_pt_t,
                    ut_info.lengths, "t",
                ).sum()
                kl_t =  kl_e_t + kl_t_t
                kl += kl_t.detach()

                nll += nll_t.detach()

                attn_nll_t = -(log_qe_t + log_qt_t)[vt2d == y_t].sum() if supattn else 0

                # Break up backprop over time
                if learn:
                    if self.Kl <= 0:
                        Bt = log_py_Ea_t
                    score = (log_py_z - Bt).detach()
                    self.delta = (log_py_z.exp() - Bt.exp()).detach()
                    rewardt = score * v_s_log_qv_t.narrow("k", self.Kl, self.K)
                    rewardt = -(rewardt * v_s_qv_t.detach()).sum("k")[y_maskt].sum()
                    if self.qconly:
                        rewardt = 0
                    if self.kl_anneal_steps > 0:
                        # shouldn't reach this of jointcopy + no qc
                        kl_coef = self.kl_coef(self.steps, self.kl_anneal_steps)
                        kl_t = kl_t * kl_coef
                    (rewardt + nll_t + kl_t + attn_nll_t).div(nwt).backward()
                    # acc grads
                    if rnn_t.values.grad is not None:
                        rnn_grads.append(rnn_t._new(rnn_t.values.grad))
                    if log_py_Ea_t.values.grad is not None:
                        log_py_Ea_grads.append(log_py_Ea_t._new(log_py_Ea_t.values.grad))
                    if not self.jointcopy:
                        log_pc_grads.append(log_pc_t._new(log_pc_t.values.grad))
                    if log_pe_t.values.grad is not None:
                        log_pe_grads.append(log_pe_t._new(log_pe_t.values.grad))
                    if log_pt_t.values.grad is not None:
                        log_pt_grads.append(log_pt_t._new(log_pt_t.values.grad))
                    if log_qc_t.values.grad is not None:
                        log_qc_grads.append(log_qc_t._new(log_qc_t.values.grad))
                    if log_qe_t.values.grad is not None:
                        log_qe_grads.append(log_qe_t._new(log_qe_t.values.grad))
                    if log_qt_t.values.grad is not None:
                        log_qt_grads.append(log_qt_t._new(log_qt_t.values.grad))
                    if c_log_qc_t.values.grad is not None:
                        c_log_qc_grads.append(c_log_qc_t._new(c_log_qc_t.values.grad))
                    #if log_qv_y_t.values.grad is not None:
                        #log_qv_y_grads.append(log_qv_y_t._new(log_qv_y_t.values.grad))
                    if v2dx_t.values.grad is not None:
                        v2dx_grads.append(v2dx_t._new(v2dx_t.values.grad))
                    if v2dgx is not None and v2dgx_t.values.grad is not None:
                        v2dgx_grads.append(v2dgx_t._new(v2dgx_t.values.grad))
                    #if self.steps > 200:
                        #import pdb; pdb.set_trace()
                # Add these back in later
                """
                e_s.append(e_s_t.detach())
                e_s_log_qe.append(e_s_log_qe_t.detach())
                t_s.append(t_s_t.detach())
                t_s_log_qt.append(t_s_log_qt_t.detach())
                """
                v_s.append(ntorch.cat([C_v_s_t, nC_v_s_t], "k"))
                v_s_log_qv.append(v_s_log_qv_t.detach())
                #log_py_c0.append(log_py_c0_t.detach())
                log_py_ac1.append(log_py_ac.detach())
            else:
                # exact
                raise NotImplementedError

        """
        print(
            (log_qc.exp().get("copy", 1) > 0.5).float().sum().item()
            / log_qc.get("copy", 1).values.nelement()
        )
        #import pdb; pdb.set_trace()
        """
        """
        Hqe = torch.distributions.Categorical(logits=log_qe.values).entropy()
        Hqt = torch.distributions.Categorical(logits=log_qt.values).entropy()
        Hpe = torch.distributions.Categorical(logits=log_pe.values).entropy()
        Hpt = torch.distributions.Categorical(logits=log_pt.values).entropy()
        print(Hqe[Hqe == Hqe].min().item(), Hpe[Hpe == Hpe].min().item())
        print(Hqe[Hqe == Hqe].max().item(), Hpe[Hpe == Hpe].max().item())
        import pdb; pdb.set_trace()
        """

        if learn:
            self.numexamples += 1
            self.steps += 1
            if rnn_grads:
                rnn_grads = ntorch.cat(rnn_grads, "time")
            if log_py_Ea_grads:
                log_py_Ea_grads = ntorch.cat(log_py_Ea_grads, "time")
            if not self.jointcopy:
                log_pc_grads = ntorch.cat(log_pc_grads, "time")
            if log_pe_grads:
                log_pe_grads = ntorch.cat(log_pe_grads, "time")
            if log_pt_grads:
                log_pt_grads = ntorch.cat(log_pt_grads, "time")
            if log_qc_grads:
                log_qc_grads = ntorch.cat(log_qc_grads, "time")
            if log_qe_grads:
                log_qe_grads = ntorch.cat(log_qe_grads, "time")
            if log_qt_grads:
                log_qt_grads = ntorch.cat(log_qt_grads, "time")
            if log_qv_y_grads:
                log_qv_y_grads = ntorch.cat(log_qv_y_grads, "time")
            v2dx_grads = sum(v2dx_grads)
            v2dgx_grads = sum(v2dgx_grads)

            bwd_outputs = []
            bwd_grads = []
            if rnn_grads:
                bwd_outputs.append(rnn_o.values)
                bwd_grads.append(rnn_grads.values)
            if log_pe_grads:
                bwd_outputs.append(log_pe.values)
                bwd_grads.append(log_pe_grads.values)
            if log_pt_grads:
                bwd_outputs.append(log_pt.values)
                bwd_grads.append(log_pt_grads.values)
            if log_qe_grads:
                bwd_outputs.append(log_qe.values)
                bwd_grads.append(log_qe_grads.values)
            if log_qt_grads:
                bwd_outputs.append(log_qt.values)
                bwd_grads.append(log_qt_grads.values)
            if log_qv_y_grads:
                bwd_outputs.append(log_qv_y.values)
                bwd_grads.append(log_qv_y_grads.values)
            if v2dx_grads and v2dx.values.requires_grad:
                # glove has frozen embeddings
                bwd_outputs.append(v2dx.values)
                bwd_grads.append(v2dx_grads.values)
            if v2dgx_grads and v2dgx.values.requires_grad:
                # glove has frozen embeddings
                bwd_outputs.append(v2dgx.values)
                bwd_grads.append(v2dgx_grads.values)
            if log_py_Ea_grads:
                bwd_outputs.append(log_py_Ea.values)
                bwd_grads.append(log_py_Ea_grads.values)
            if log_qc_grads:
                bwd_outputs.append(log_alpha_qc.values)
                bwd_grads.append(log_qc_grads.values)
            if not self.jointcopy:
                bwd_outputs.append(log_alpha_pc.values)
                bwd_grads.append(log_pc_grads.values)
            #import pdb; pdb.set_trace()
            backward(bwd_outputs, bwd_grads)

        """
        if K > 0:
            e_s = ntorch.cat(e_s, "time")
            e_s_log_qe = ntorch.cat(e_s_log_qe, "time")
            t_s = ntorch.cat(t_s, "time")
            t_s_log_qt = ntorch.cat(t_s_log_qt, "time")
        """
        if K > 0:
            v_s = ntorch.cat(v_s, "time")
            v_s_log_qv = ntorch.cat(v_s_log_qv, "time")
        #log_py_c0 = ntorch.cat(log_py_c0, "time")
        log_py_ac1 = ntorch.cat(log_py_ac1, "time")
        #log_py_a = ntorch.cat(log_py_a, "time")
        #log_py = ntorch.cat(log_py, "time")
        #log_pa_y = ntorch.cat(log_pa_y, "time")

        rvinfo = RvInfo(
            log_py     = None,
            log_py_a   = None,#log_py_a,
            log_py_c0  = None,#log_py_c0,
            log_py_ac1 = log_py_ac1.detach(),
            log_pe     = log_pe.detach(),
            log_pt     = log_pt.detach(),
            log_qe_y   = log_qe.detach(),
            log_qt_y   = log_qt.detach(),
            log_pc     = log_alpha_pc.detach(),
            log_qc_y   = log_alpha_qc.detach(),
            e_s        = e_s,
            e_s_log_p  = e_s_log_qe,
            t_s        = t_s,
            t_s_log_p  = t_s_log_qt,
            v_s        = v_s,
            v_s_log_p  = v_s_log_qv.detach(),
            log_py_Ea  = log_py_Ea.detach(),
        )
        #import pdb; pdb.set_trace()
        if self.steps > 1000 and False:
            teamname = self.Vt.stoi["team_name"]
            if ut.eq(teamname).any():
                idxs = [(tidx, batch) for tidx,batch in ut.eq(teamname).nonzero().tolist()]
                for tidx, batch in idxs:
                    # [time x k]
                    batch_idxs = t_s.get("batch", batch).eq(tidx).nonzero().tolist()
                    for (time, k) in batch_idxs:
                        soft = log_py_c0[{"time": time, "batch": batch}].item()
                        hard = log_py_ac1[{"time": time, "batch": batch, "k": k}].item()
                        word = self.Vx.itos[y[{"time": time, "batch": batch}].item()]
                        entity = self.Ve.itos[
                            ue[{
                                "els": e_s[{"time": time, "batch": batch, "k": k}].item(),
                                "batch": batch,
                            }].item()
                        ]
                        print(
                            f"{word} || {entity} || {soft} || {hard}"
                        )
                        import pdb; pdb.set_trace()
                    #print(f"has team name: {has_tm.item()}")
            #import pdb; pdb.set_trace()
            """
            bulls = self.Ve.stoi["bulls"]
            if ue.eq(bulls).any():
                eidx, batch = ue.eq(bulls).nonzero().tolist()[0]
                has_bulls = e_s.get("batch", batch).eq(eidx).any()
                print(f"has bulls: {has_bulls.item()}")
                #if has_bulls:
                    #import pdb; pdb.set_trace()
            """
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


    # move to lvm.py? maybe later
    def kl(self, qa, log_qa, log_pa, lens, dim):
        kl = []
        for i, l in enumerate(lens.tolist()):
            qa0 = qa.get("batch", i).narrow(dim, 0, l)
            log_qa0 = log_qa.get("batch", i).narrow(dim, 0, l)
            log_pa0 = log_pa.get("batch", i).narrow(dim, 0, l)
            kl0 =  qa0 * (log_qa0 - log_pa0)
            infmask = log_qa0 != float("-inf")
            # workaround for namedtensor bug that puts empty tensors on different devices
            kl0 = kl0.transpose("time", dim)
            kl0 = kl0._new(kl0.values.where(infmask.values, torch.zeros_like(kl0.values))).sum(dim)
            kl.append(kl0)
        return ntorch.stack(kl, "batch")

    def kl_coef(self, steps, kl_anneal_steps):
        return (1 if steps > kl_anneal_steps
            else float(steps) / float(kl_anneal_steps))

    def weight_coef(self, steps, anneal_steps):
        return (1 if steps > anneal_steps
            else float(steps) / float(anneal_steps))

