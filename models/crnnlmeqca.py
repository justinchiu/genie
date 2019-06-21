import math
from collections import defaultdict
import heapq

from .lvma import LvmA, RvInfo

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

class CrnnLmEqca(LvmA):
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
        initu = False,
        tanh = False,
        vctxt = False,
        wcv = False,
        etvctxt = False,
        bil = False,
        mlp = False,
        untie = False,
    ):
        super(CrnnLmEqca, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self._mode = "elbo"
        self._q = "qay"
        self.K = 0
        self.Kq = 0
        self.Kl = 0
        self.Kb = 0

        # should be:
        # self.nokl = not qconly
        self.nokl = True
        self.v2d = v2d
        self.initvy = initvy
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
        self.q_warmup_steps = 0
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
        self.luty = ntorch.nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        ).spec("time", "x")
        if self.initu:
            self.lutx.weight.data.uniform_(-0.1, 0.1)
        self.lutx.weight.data[Vx.stoi[self.PAD]].fill_(0)
        if self.untie:
            self.lutgx = ntorch.nn.Embedding(
                num_embeddings = len(Vx),
                embedding_dim = x_emb_sz,
                padding_idx = Vx.stoi[self.PAD],
            ).spec("time", "x")
            if self.initu:
                self.lutgx.weight.data.uniform_(-0.1, 0.1)
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
        if self.untie:
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
            if self.untie:
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

        self.ivproj = ntorch.nn.Linear(
            in_features = 2*rnn_sz,
            out_features = len(Vv),
            bias = False,
        ).spec("rnns", "v")


        self.type(self.dtype)

        # cache for likelihoods?
        self.cache = defaultdict(lambda: defaultdict(list))

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # maybe iwae later, but i don't really care about it
        if mode == "elbo":
            # elbo can be exact or approximated through sampling
            #assert(self.K >= 0)
            pass
        elif mode == "marginal":
            # only exact for marginal
            #assert(self.K == 0)
            pass
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
            log_ra = log_ea + log_ta
            rc = log_ra.exp().dot(("t", "e"), v2dx)
            ra = log_ra.exp()

            ea = ea.rename("els", "e")
            ta = ta.rename("els", "t")

            output = rnn_o
        else:
            # no need for input feeding ever
            log_ea, ea, ec = [], [], []
            log_ta, ta, tc = [], [], []
            log_ra, ra, rc = [], [], []
            out = []
            etc_t = ntorch.zeros(
                N, self.r_emb_sz, names=("batch", "rnns")
            ).to(emb_x.ralues.device)
            for t in range(T):
                etc_t = etc_t.rename("rnns", "x")
                inp = ntorch.cat([emb_x.get("time", t), etc_t], "x").repeat("time", 1)
                rnn_o, s = self.rnn(inp, s)
                rnn_o = rnn_o.get("time", 0)
                log_ea_t, ea_t, ec_t = attn(rnn_o, emb_e, ue_info.mask)
                log_ta_t, ta_t, tc_t = attn(rnn_o, emb_t, ut_info.mask)
                log_ea_t = log_ea_t.rename("els", "e")
                log_ta_t = log_ta_t.rename("els", "t")
                log_ra_t = log_ea_t + log_ta_t
                ra_t = log_ra_t.exp()
                rc_t = ra_t.dot(("t", "e"), v2dx)
                out.append(
                    self.Wif(ntorch.cat([rnn_o, vc_t, ec_t, tc_t], "rnns"))
                )

                log_ea.append(log_ea_t)
                ea.append(ea_t)
                ec.append(ec_t)
                log_ta.append(log_ta_t)
                ta.append(ta_t)
                tc.append(tc_t)
                log_ra.append(log_ra_t)
                ra.append(ra_t)
                rc.append(rc_t)

            output = ntorch.stack(out, "time")

            log_ea = ntorch.stack(log_ea, "time")
            ea = ntorch.stack(ea, "time")
            ec = ntorch.stack(ec, "time")
            log_ta = ntorch.stack(log_ta, "time")
            ta = ntorch.stack(ta, "time")
            tc = ntorch.stack(tc, "time")
            log_ra = ntorch.stack(log_ra, "time")
            ra = ntorch.stack(ra, "time")
            rc = ntorch.stack(rc, "time")

            ea = ea.rename("els", "e")
            ta = ta.rename("els", "t")

        return log_ea, ea, ec, log_ta, ta, tc, log_ra, ra, rc, output, s


    def log_pc(self, rnn_o):
        # rnn_o = f(y<t)
        return self.Wcopy(rnn_o).log_softmax("copy")

    # posterior
    def log_qac_y(self, emb_y, y_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        # ea: T x N x R
        log_ea, ea, ec = attn(rnn_o, self.Wie(emb_e), ue_info.mask, self.temp)
        log_ta, ta, tc = attn(rnn_o, self.Wit(emb_t), ut_info.mask, self.temp)
        log_ea = log_ea.rename("els", "e")
        log_ta = log_ta.rename("els", "t")
        log_ra = log_ea + log_ta
        if self.qcrnn:
            rnn_oc, _ = self.brnnc(emb_y.detach(), None, lengths=y_info.lengths) 
        log_c = self.Wicopy(rnn_oc if self.qcrnn else rnn_o).log_softmax("copy") 

        return log_c, log_ea, log_ta, log_ra, rnn_o


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
        idxs=None,
    ):
        # shared encoding
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
                .rename("x", "rnns") if self.untie else None
        )

        # shared attention
        # This may need to be broken up over time of not enough memory...
        # length stuff might be a bit annoying
        #log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, x_info, r_info)
        log_pe, pe, ec, log_pt, pt, tc, log_pa, pa, rc, rnn_o, s = self.pa0(
            emb_x, s, x_info, eA, ue_info, tA, ut_info, v2dx)

        # use soft attention over everything for p(c | y_<-t)
        # maybe no vc?
        log_pc = self.log_pc(rnn_o)

        # anneal vc to 0
        if self.c_anneal_steps > 0 and self.steps > self.c_warmup_steps:
            wc = self.weight_coef(self.steps - self.c_warmup_steps, self.c_anneal_steps)
            rc = (1-wc) * rc

        K = self.K

        # jesus, need to fix this
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
            past = self.Wcv(ntorch.cat([rnn_o, ec + tc + rc], "rnns")).repeat("k", 1)
            if self.tanh:
                past = past.tanh()
        else:
            past = self.Wc(ntorch.cat([rnn_o, ec, tc, rc], "rnns")).repeat("k", 1)
            if self.tanh:
                past = past.tanh()
        log_py_Ea = self.log_py_ac0(past, y)

        if y is not None:
            emb_y = self.luty(y)
            log_qc, log_qe, log_qt, log_qa, brnn_o = self.log_qac_y(
                emb_y, x_info, emb_e, ue_info, emb_t, ut_info, v2dx)

        nll = 0.
        kl = 0.
        klc = 0.

        log_py_a = []
        log_pa_y = []
        log_py_c0 = []
        log_py_ac1 = []

        # sample only
        a_s = [] if K > 0 else None
        a_s_log_qa = [] if K > 0 else None
        e_s = [] if K > 0 else None
        e_s_log_qe = [] if K > 0 else None
        t_s = [] if K > 0 else None
        t_s_log_qt = [] if K > 0 else None


        # cat over time
        rnn_grads = []
        brnn_grads = []
        log_py_Ea_grads = []

        # cat over time
        log_pc_grads = []
        log_pe_grads = []
        log_pt_grads = []
        log_qc_grads = []
        log_qe_grads = []
        log_qt_grads = []
        log_qa_y_grads = []

        # sum over time
        v2dx_grads = []
        v2dgx_grads = []

        nwt = y.ne(1).sum().float() if y is not None else 1
        # break up over time
        # pick and use either e and t or v...
        for t, (
            rnn_t,
            brnn_t,
            log_py_Ea_t,
            log_pc_t, log_pe_t, log_pt_t,
            log_qc_t, log_qe_t, log_qt_t,
            #log_qa_y_t,
            y_t,
        ) in enumerate(zip(
            rnn_o.split(T, "time"),
            brnn_o.split(T, "time"),
            log_py_Ea.split(T, "time"),
            log_pc.split(T, "time"),
            log_pe.split(T, "time"),
            log_pt.split(T, "time"),
            log_qc.split(T, "time"),
            log_qe.split(T, "time"),
            log_qt.split(T, "time"),
            #log_qa_y.split(T, "time"),
            (y if y is not None else ec).split(T, "time"),
        )):
            # accumulate gradients by hand so we don't need to retain graph
            rnn_t = rnn_t._new(rnn_t.detach().values.requires_grad_(True))
            brnn_t = brnn_t._new(brnn_t.detach().values.requires_grad_(True))
            log_py_Ea_t = log_py_Ea_t._new(log_py_Ea_t.detach().values.requires_grad_(True))
            log_pc_t = log_pc_t._new(log_pc_t.detach().values.requires_grad_(True))
            log_pe_t = log_pe_t._new(log_pe_t.detach().values.requires_grad_(True))
            log_pt_t = log_pt_t._new(log_pt_t.detach().values.requires_grad_(True))
            log_qc_t = log_qc_t._new(log_qc_t.detach().values.requires_grad_(True))
            log_qe_t = log_qe_t._new(log_qe_t.detach().values.requires_grad_(True))
            log_qt_t = log_qt_t._new(log_qt_t.detach().values.requires_grad_(True))
            #log_qa_y_t = log_qa_y_t._new(log_qa_y_t.detach().values.requires_grad_(True))
            v2dx_t = v2dx._new(v2dx.detach().values.requires_grad_(True))
            if v2dgx is not None:
                v2dgx_t = v2dgx._new(v2dx.detach().values.requires_grad_(True))

            y_maskt = y_t.ne(1)

            # just sample now
            if K > 0:
                qc_t = log_qc_t.exp()
                qe_t = log_qe_t.exp()
                qt_t = log_qt_t.exp()

                # actually, take top k from product dist
                # then sample from complement
                log_qa_t = (log_qe_t + log_qt_t).stack(("e", "t"), "v")
                log_pa_t = (log_pe_t + log_pt_t).stack(("e", "t"), "v")

                # for baseline
                if self.Kb > 0:
                    B_a_s_t, B_a_s_log_qa = self.sample_a(log_qa_t.exp(), log_qa_t, self.Kb, "v", "k")
                    B_a_s_log_pa = log_pa_t.gather("v", B_a_s_t, "k")

                if self.Kl > 0:
                    # get top likelihoods from cache
                    particles = []
                    for batch, s_idx in enumerate(idxs):
                        batch_p = []
                        for time in range(rnn_t.shape["time"]):
                            cache = self.cache[s_idx][time + t * T]
                            # dbg, remove this
                            cache = self.cache[2228][0]
                            lset = set(x[1] for x in cache)
                            batch_p.append(lset)
                        particles.append(batch_p)

                if self.Kq > 0:
                    # take the topk from the complement of the cache samples
                    Ks = self.Kq + self.Kl
                    C_a_s_log_qa, C_a_s_t = log_qa_t.topk("v", Ks)
                    # already sorted
                    if self.Kl > 0:
                        # UNFINISHED
                        # batch x time x K
                        batches = []
                        for batch, s_idx in enumerate(idxs):
                            times = []
                            for time in range(rnn_t.shape["time"]):
                                ok = particles[batch][time]
                                lol = C_a_s_t[{"batch": batch, "time": time}]
                                lollist = lol.tolist()
                                lolset = set(lollist)
                                diff = ok - lolset
                                cap = lolset & ok
                                take = Ks - len(diff)
                                samples = []
                                i = 0
                                while i < take:
                                    if lollist[i] not in ok:
                                        samples.append(lollist[i])
                                    i += 1
                                samples.extend(list(ok))
                                import pdb; pdb.set_trace()
                                if self.steps > 0:
                                    import pdb; pdb.set_trace()
                                times.append(samples)
                            batches.append(times)
                        shape = C_a_s_t.shape
                        C_a_s_t = NamedTensor(
                            torch.LongTensor(batches).to(C_a_s_t.values.device),
                            names = ("batch", "time", "v"),
                        )
                        C_a_s_log_qa = log_qa_t.gather("v", C_a_s_t, "v")
                        if self.steps > 0:
                            import pdb; pdb.set_trace()
                    C_a_s_log_pa = log_pa_t.gather("v", C_a_s_t, "v")
                    C_a_s_log_qa = C_a_s_log_qa.rename("v", "k")
                    C_a_s_log_pa = C_a_s_log_pa.rename("v", "k")
                    C_a_s_t = C_a_s_t.rename("v", "k")

                    # weight nll by this
                    C_a_s_qa = C_a_s_log_qa.exp()
                    C_Za = C_a_s_log_qa.logsumexp("k").exp()


                num_enum = self.Kq + self.Kl
                Kc = self.K - num_enum
                nC_log_qa = log_qa_t.clone()
                if self.Kq > 0:
                    nC_log_qa.scatter_("v", C_a_s_t, C_a_s_log_qa.clone().fill_(float("-inf")), "k")
                # reciprocal of normalizing constant, multiply MC term by this
                nC_Za = nC_log_qa.logsumexp("v").exp()
                nC_qa = nC_log_qa.softmax("v")

                nC_a_s_t, nC_a_s_log_qa = self.sample_a(nC_qa, log_qa_t, Kc, "v", "k")
                nC_a_s_log_pa = log_pa_t.gather("v", nC_a_s_t, "k")
                nC_a_s_qa = nC_a_s_log_qa.exp()


                """
                if self.Kb > 0:
                    B_v_f_t = v2dx_t_flat.gather("r", B_v_s_t, "k")
                if self.Kq > 0:
                    C_v_f_t = v2dx_t_flat.gather("r", C_v_s_t, "k")
                nC_v_f_t = v2dx_t_flat.gather("r", nC_v_s_t, "k")
                """

                samples = []
                log_probs_q = []
                log_probs_p = []
                if self.Kb > 0:
                    samples.append(B_a_s_t)
                    log_probs_q.append(B_a_s_log_qa)
                    log_probs_p.append(B_a_s_log_pa)
                if self.Kq > 0:
                    samples.append(C_a_s_t)
                    log_probs_q.append(C_a_s_log_qa)
                    log_probs_p.append(C_a_s_log_pa)
                if Kc > 0:
                    samples.append(nC_a_s_t)
                    log_probs_q.append(nC_a_s_log_qa)
                    log_probs_p.append(nC_a_s_log_pa)
                a_s_t = ntorch.cat(samples, "k")
                a_s_log_qa_t = ntorch.cat(log_probs_q, "k")
                a_s_log_pa_t = ntorch.cat(log_probs_p, "k")
                #import pdb; pdb.set_trace()

                # for weighting only, detach before using in all cases
                # \sum_a q(a) g(a) + q(nC)E_{a~nC}[g(a)]
                a_s_qa_t = ntorch.cat([C_a_s_qa, nC_Za.repeat("k", Kc) / Kc], "k")

                # sampled and flatten values at timestep t
                v2dx_t_flat = (v2dx_t if not self.untie else v2dgx_t).stack(("e", "t"), "r")
                ctxt = v2dx_t_flat.gather("r", a_s_t.rename("v", "k"), "k")

                # targets should be in vocab space
                # since v has been used for alignment, use 
                v_flat = v2d.stack(("e", "t"), "r")
                # the actual sample alignment values
                v_k_t = v_flat.gather("r", a_s_t.rename("v", "k"), "k")

                # get likelihood of v_k_t | y
                e_s_t = a_s_t.div(ut.shape["els"])
                t_s_t = a_s_t.fmod(ut.shape["els"])
                #ma = v_k_t[{"k": 40, "time": 0, "batch": 0}]
                #mb = e_k_t[{"k": 40, "time": 0, "batch": 0}]
                #mc = t_k_t[{"k": 40, "time": 0, "batch": 0}]
                # assert(ma == mb * ut.shape["els"] + mc)
                em = emb_e.gather("els", e_s_t, "k").rename("e", "rnns")
                tm = emb_t.gather("els", t_s_t, "k").rename("t", "rnns")
                log_pv = self.vproj(ntorch.cat([em, tm], "rnns")).gather(
                    "v", v_k_t.repeat("m", 1), "m",
                ).get("m", 0)
                log_qv_ay = self.ivproj(brnn_t).log_softmax("v").gather(
                    "v", v_k_t, "k",
                )

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

                #import pdb; pdb.set_trace()
                #log_py_ac0_t = self.log_py_a(past, y_t)
                log_py_c0_t = log_py_Ea_t
                log_py_ac1_t = (
                    self.proj(past) if not self.untie else self.gproj(past)
                ).log_softmax("vocab")
                log_py_ac1_t = (
                    log_py_ac1_t.gather("vocab", y_t.repeat("lol", 1), "lol").get("lol", 0)
                    if y is not None
                    else log_py_ac1_t
                )
                #log_py_ac1_t = self.log_py_ac1(v2dx_t, v_f_t, y_t)

                # DEBUG NUM COPY
                #self.copied += log_py_ac1_t.exp().mean("k").sum().item()
                """
                B_v_t = v2d_flat.gather("r", B_v_s_t.rename("v", "k"), "k")
                C_v_t = v2d_flat.gather("r", C_v_s_t.rename("v", "k"), "k")
                nC_v_t = v2d_flat.gather("r", nC_v_s_t, "k")
                import pdb; pdb.set_trace()
                """
                
                lpct = math.log(.5) if self.jointcopy else log_pc_t

                # train log_qc as well as log_qv_ay
                if self.qconly:
                    raise NotImplementedError
                    Eqac_log_py_ac_t = (
                        qc_t.get("copy", 0) * log_py_c0_t.detach()
                        + qc_t.get("copy", 1) * log_py_ac1_t.detach()
                    )
                    #import pdb; pdb.set_trace()
                    log_py_z = Eqac_log_py_ac_t
                    nll_t = -Eqac_log_py_ac_t.mean("k")[y_maskt].sum()
                    # kl[qc || pc]
                    # can sum over copy first but by fubini doesn't matter
                    klc_t = (qc_t * (log_qc_t - lpct))[y_maskt].sum()
                    #klc_t = (qc_t.detach() * (log_qc_t.detach() - lpct))[y_maskt].sum()
                    klc += klc_t

                    kl_e_t = self.kl(
                        qc_t.get("copy", 1) * qe_t.detach(),
                        log_qe_t.detach(), log_pe_t.detach(),
                        ue_info.lengths, "e",
                    ).sum()
                    kl_t_t = self.kl(
                        qc_t.get("copy", 1) * qt_t.detach(),
                        log_qt_t.detach(), log_pt_t.detach(),
                        ut_info.lengths, "t",
                    ).sum()
                    kl_t =  (kl_e_t + kl_t_t)
                    kl += kl_t.detach() + klc_t.detach()
                    #import pdb; pdb.set_trace()
                elif self.qc:
                    raise NotImplementedError
                    Eqac_log_py_ac_t = (
                        qc_t.get("copy", 0) * log_py_c0_t
                        + qc_t.get("copy", 1) * log_py_ac1_t
                    )
                    log_py_z = Eqac_log_py_ac_t
                    nll_t = -(Eqac_log_py_ac_t * v_s_qv_t.detach()).sum("k")[y_maskt].sum()
                    # kl[qc || pc]
                    # can sum over copy first but by fubini doesn't matter
                    klc_t = (qc_t * (log_qc_t - lpct))[y_maskt].sum()
                    klc += klc_t

                    kl_e_t = self.kl(
                        qc_t.get("copy", 1) * qe_t,
                        log_qe_t, log_pe_t,
                        ue_info.lengths, "e",
                    ).sum()
                    kl_t_t = self.kl(
                        qc_t.get("copy", 1) * qt_t,
                        log_qt_t, log_pt_t,
                        ut_info.lengths, "t",
                    ).sum()
                    kl_t =  (kl_e_t + kl_t_t + klc_t)
                    kl += kl_t.detach() + klc_t.detach()
                elif not self.jointcopy:
                    raise NotImplementedError
                    Eqa_log_py_a_t = logaddexp(
                        log_pc_t.get("copy", 0) + log_py_c0_t,
                        log_pc_t.get("copy", 1) + log_py_ac1_t
                            + a_s_log_pa_t - a_s_log_qa_t,
                    )
                    nll_t = -(Eqa_log_py_a_t * a_s_qa_t.detach()).sum("k")[y_maskt].sum()
                    log_py_z = Eqa_log_py_a_t

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
                    kl_t = 0
                    klc_t = 0
                    kl += kl_e_t.detach() + kl_t_t.detach()
                else:
                    #kl_coef = self.kl_coef(self.steps, self.kl_anneal_steps) if self.kl_anneal_steps > 0 else 1

                    Eqa_log_py_a_t = logaddexp(
                        log_py_c0_t + math.log(0.5),
                        log_py_ac1_t + math.log(0.5)
                            + (a_s_log_pa_t - a_s_log_qa_t),# * kl_coef,
                    )
                    if self.Kb > 0:
                        Bt = Eqa_log_py_a_t.narrow("k", 0, self.Kb).mean("k")
                        log_py_z = Eqa_log_py_a_t.narrow("k", self.Kb, K)
                    else:
                        log_py_z = Eqa_log_py_a_t
                    nll_t = -(log_py_z * a_s_qa_t.detach()).sum("k")[y_maskt].sum()

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
                    kl_t = 0
                    klc_t = 0
                    kl += kl_e_t.detach() + kl_t_t.detach()

                nll += nll_t.detach()

                attn_nll_t = -(log_qe_t + log_qt_t)[vt2d == y_t].sum() if supattn else 0

                # Break up backprop over time
                if learn:
                    if self.Kl <= 0:
                        Bt = log_py_Ea_t
                    score = (log_py_z - Bt).detach()
                    self.delta = (log_py_z.exp() - Bt.exp()).detach()
                    rewardt = score * a_s_log_qa_t.narrow("k", self.Kb, K)
                    rewardt = -(rewardt * a_s_qa_t.detach()).sum("k")[y_maskt].sum()
                    if self.qconly:
                        rewardt = 0
                    if self.kl_anneal_steps > 0:
                        # shouldn't reach this of jointcopy + no qc
                        kl_coef = self.kl_coef(self.steps, self.kl_anneal_steps)
                        kl_t = kl_t * kl_coef
                        klc_t = klc_t * kl_coef
                    if (
                        self.steps < self.q_warmup_steps
                        or (self.qsteps > 0 and self.steps % (self.qsteps + 1) != 0)
                    ):
                        (rewardt + kl_t).div(nwt).backward()
                    else:
                        (rewardt + nll_t + kl_t + attn_nll_t + klc_t).div(nwt).backward()
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
                    #if log_qv_y_t.values.grad is not None:
                        #log_qv_y_grads.append(log_qv_y_t._new(log_qv_y_t.values.grad))
                    if v2dx_t.values.grad is not None:
                        v2dx_grads.append(v2dx_t._new(v2dx_t.values.grad))
                    if v2dgx is not None and v2dgx_t.values.grad is not None:
                        v2dgx_grads.append(v2dgx_t._new(v2dgx_t.values.grad))
                    #if self.steps > 200:
                        #import pdb; pdb.set_trace()
                        
                    # cache likelihoods
                    scores, indices = log_py_z.topk("k", 16, sorted=True, largest=True)
                    if self.Kl > 0:
                        for batch, s_idx in enumerate(idxs):
                            scores_b = scores.get("batch", batch)
                            indices_b = indices.get("batch", batch)
                            for time in range(rnn_t.shape["time"]):
                                # fix time offset because of chopping
                                this_time = t * T + time
                                w_score = scores_b.get("time", time)
                                w_idx = indices_b.get("time", time)
                                for w_s, w_i in zip(w_score.tolist(), w_idx.tolist()):
                                    heapq.heappush(self.cache[s_idx][this_time], (w_s, w_i))
                                # truncate
                                self.cache[s_idx][this_time] = heapq.nlargest(
                                    self.Kl, self.cache[s_idx][this_time])
                                # maintain heap invariant
                                heapq.heapify(self.cache[s_idx][this_time])
                                # what about repeats? this caches the Kl highest likelihoods ever,
                                # with each repeat treated uniquely
                                #if self.steps > 0:
                                    #import pdb; pdb.set_trace()
                # Add these back in later
                """
                e_s.append(e_s_t.detach())
                e_s_log_qe.append(e_s_log_qe_t.detach())
                t_s.append(t_s_t.detach())
                t_s_log_qt.append(t_s_log_qt_t.detach())
                """
                a_s.append(ntorch.cat([C_a_s_t, nC_a_s_t], "k"))
                a_s_log_qa.append(a_s_log_qa_t.detach())
                log_py_c0.append(log_py_c0_t.detach())
                log_py_ac1.append(log_py_ac1_t.detach())
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
            if brnn_grads:
                brnn_grads = ntorch.cat(brnn_grads, "time")
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
            if log_qa_y_grads:
                log_qa_y_grads = ntorch.cat(log_qa_y_grads, "time")
            v2dx_grads = sum(v2dx_grads)
            v2dgx_grads = sum(v2dgx_grads)

            bwd_outputs = []
            bwd_grads = []
            if rnn_grads:
                bwd_outputs.append(rnn_o.values)
                bwd_grads.append(rnn_grads.values)
            if brnn_grads:
                bwd_outputs.append(brnn_o.values)
                bwd_grads.append(brnn_grads.values)
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
            if log_qa_y_grads:
                bwd_outputs.append(log_qa_y.values)
                bwd_grads.append(log_qa_y_grads.values)
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
                bwd_outputs.append(log_qc.values)
                bwd_grads.append(log_qc_grads.values)
            if not self.jointcopy:
                bwd_outputs.append(log_pc.values)
                bwd_grads.append(log_pc_grads.values)
            #import pdb; pdb.set_trace()
            backward(bwd_outputs, bwd_grads)

        if K > 0:
            a_s = ntorch.cat(a_s, "time")
            a_s_log_qa = ntorch.cat(a_s_log_qa, "time")
        log_py_c0 = ntorch.cat(log_py_c0, "time")
        log_py_ac1 = ntorch.cat(log_py_ac1, "time")
        #log_py_a = ntorch.cat(log_py_a, "time")
        #log_py = ntorch.cat(log_py, "time")
        #log_pa_y = ntorch.cat(log_pa_y, "time")

        rvinfo = RvInfo(
            log_py     = None,
            log_py_a   = None,#log_py_a,
            log_py_c0  = log_py_c0,
            log_py_ac1 = log_py_ac1,
            log_pe     = log_pe,
            log_pt     = log_pt,
            log_qe_y   = log_qe,
            log_qt_y   = log_qt,
            log_pc     = log_pc,
            log_qc_y   = log_qc,
            e_s        = e_s,
            e_s_log_p  = e_s_log_qe,
            t_s        = t_s,
            t_s_log_p  = t_s_log_qt,
            a_s        = a_s,
            a_s_log_p  = a_s_log_qa,
            log_py_Ea  = log_py_Ea,
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
            # factor out checks for teamnames and stuff
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

