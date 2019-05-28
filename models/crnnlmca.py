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

from .fns import attn, logaddexp


def print_mem():
    print(f"Mem (MB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cached Mem (MB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Max Mem (MB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")

class CrnnLmCa(LvmA):
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
        jointcopy = False,
        qc = True,
        qcrnn = True,
    ):
        super(CrnnLmCa, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0
        self._mode = "elbo"
        self._q = "qay"
        self.K = 0

        self.noattn = False
        self.noattnvalues = noattnvalues
        self.jointcopy = jointcopy
        self.qc = qc
        self.qcrnn = qcrnn

        self.numexamples = 0

        self.kl_anneal_steps = 0
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

    def log_pc(self, rnn_o, ec, tc, vc):
        # rnn_o = f(y<t)
        # condition on soft attention
        # rW_a = g(r, a) = r[a]
        # already concatenated if input feed
        return self.Wcopy(
            rnn_o if self.inputfeed else
            self.Wif(ntorch.cat([rnn_o, ec, tc, vc], "rnns"))
        ).log_softmax("copy")

    # posterior
    def log_qac_y(self, emb_y, y_info, emb_e, ue_info, emb_t, ut_info, v2dx):
        rnn_o, _ = self.brnn(emb_y, None, lengths=y_info.lengths)
        # ea: T x N x R
        log_ea, ea, ec = attn(rnn_o, self.Wie(emb_e), ue_info.mask)
        log_ta, ta, tc = attn(rnn_o, self.Wit(emb_t), ut_info.mask)
        log_ea = log_ea.rename("els", "e")
        log_ta = log_ta.rename("els", "t")
        log_va = log_ea + log_ta
        #vc = log_va.exp().dot(("t", "e"), v2dx)
        #va = log_va.exp()

        #ea = ea.rename("els", "e")
        #ta = ta.rename("els", "t")
        # this is probably bad? think a little more lol
        if self.qcrnn:
            rnn_oc, _ = self.brnnc(emb_y.detach(), None, lengths=y_info.lengths)
        log_c = self.Wicopy(rnn_oc if self.qcrnn else rnn_o).log_softmax("copy")

        return log_c, log_ea, log_ta, log_va

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
        v2d,
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

        v2dx = self.lutx(
            v2d.stack(("t", "e"), "time")
        ).chop("time", ("t", "e"), t=v2d.shape["t"]).rename("x", "rnns")

        # shared attention
        # This may need to be broken up over time of not enough memory...
        # length stuff might be a bit annoying
        #log_pa, pa, ec, rnn_o, s = self.pa0(emb_x, s, x_info, r_info)
        log_pe, pe, ec, log_pt, pt, tc, log_pv, pv, vc, rnn_o, s = self.pa0(
            emb_x, s, x_info, eA, ue_info, tA, ut_info, v2dx)

        # use soft attention over everything for p(c | y_<-t)
        # maybe no vc?
        log_pc = self.log_pc(rnn_o, ec, tc, vc)

        K = self.K

        # Baseline if sample
        #past = self.Wc(ntorch.cat([rnn_o, vc], "rnns")).tanh().chop(
            #"batch", ("k", "batch"), k=1)
        past = (self.Wc_nov(ntorch.cat([rnn_o, ec, tc], "rnns"))
            if self.noattnvalues
            else self.Wc(ntorch.cat([rnn_o, ec, tc, vc], "rnns"))
        ).tanh().repeat("k", 1)
        log_py_Ea = self.log_py_ac0(past, y)

        if y is not None:
            emb_y = self.lutx(y)
            log_qc, log_qe, log_qt, log_qv = self.log_qac_y(
                emb_y, x_info, emb_e, ue_info, emb_t, ut_info, v2dx)

        nll = 0.
        kl = 0.
        klc = 0.

        log_py_a = []
        log_pa_y = []
        log_py_c0 = []
        log_py_ac1 = []

        # sample only
        e_s = [] if K > 0 else None
        e_s_log_p = [] if K > 0 else None
        t_s = [] if K > 0 else None
        t_s_log_p = [] if K > 0 else None


        # cat over time
        #rnn_grads = []
        log_py_Ea_grads = []

        # cat over time
        log_pc_grads = []
        log_pe_grads = []
        log_pt_grads = []
        log_qc_grads = []
        log_qe_grads = []
        log_qt_grads = []

        # sum over time
        v2dx_grads = []

        nwt = y.ne(1).sum().float() if y is not None else 1
        # break up over time
        # pick and use either e and t or v...
        for t, (
            #rnn_t,
            log_py_Ea_t,
            log_pc_t, log_pe_t, log_pt_t,
            log_qc_t, log_qe_t, log_qt_t,
            y_t,
        ) in enumerate(zip(
            #rnn_o.split(T, "time"),
            log_py_Ea.split(T, "time"),
            log_pc.split(T, "time"),
            log_pe.split(T, "time"),
            log_pt.split(T, "time"),
            log_qc.split(T, "time"),
            log_qe.split(T, "time"),
            log_qt.split(T, "time"),
            (y if y is not None else ec).split(T, "time"),
        )):
            # accumulate gradients by hand so we don't need to retain graph
            #rnn_t = rnn_t._new(rnn_t.detach().values.requires_grad_(True))
            log_py_Ea_t = log_py_Ea_t._new(log_py_Ea_t.detach().values.requires_grad_(True))
            log_pc_t = log_pc_t._new(log_pc_t.detach().values.requires_grad_(True))
            log_pe_t = log_pe_t._new(log_pe_t.detach().values.requires_grad_(True))
            log_pt_t = log_pt_t._new(log_pt_t.detach().values.requires_grad_(True))
            log_qc_t = log_qc_t._new(log_qc_t.detach().values.requires_grad_(True))
            log_qe_t = log_qe_t._new(log_qe_t.detach().values.requires_grad_(True))
            log_qt_t = log_qt_t._new(log_qt_t.detach().values.requires_grad_(True))
            v2dx_t = v2dx._new(v2dx.detach().values.requires_grad_(True))

            y_maskt = y_t.ne(1)

            # just sample now
            if K > 0:
                qc_t = log_qc_t.exp()
                qe_t = log_qe_t.exp()
                qt_t = log_qt_t.exp()

                # Sample, average likelihood over samples
                T_size = v2dx_t.shape["t"]
                e_s_t, e_s_log_p_t = self.sample_a(qe_t, log_qe_t, self.K, "e", "k")
                t_s_t, t_s_log_p_t = self.sample_a(qt_t, log_qt_t, self.K, "t", "k")
                # linearize
                idxs = e_s_t * T_size + t_s_t
                # sampled and flatten values at timestep t
                v_f_t = v2dx_t.stack(("e", "t"), "r").gather("r", idxs, "k")
                v_f_log_p_t = e_s_log_p_t + t_s_log_p_t

                ctxt = v_f_t

                """
                past = self.Wc(ntorch.cat(
                    [
                        rnn_t.repeat("k", ctxt.shape["k"]),
                        ctxt,
                    ],
                    "rnns",
                )).tanh()
                """
                #log_py_ac0_t = self.log_py_a(past, y_t)
                log_py_c0_t = log_py_Ea_t
                log_py_ac1_t = self.proj(ctxt.rename("rnns", "ctxt")).log_softmax("vocab")
                log_py_ac1_t = (
                    log_py_ac1_t.gather("vocab", y_t.repeat("lol", 1), "lol").get("lol", 0)
                    if y is not None
                    else log_py_ac1_t
                )
                #log_py_ac1_t = self.log_py_ac1(v2dx_t, v_f_t, y_t)

                # DEBUG NUM COPY
                #self.copied += log_py_ac1_t.exp().mean("k").sum().item()

                # does it make sense to use q here...?
                if not self.jointcopy:
                    # p(y|a) = p(y|c=0)p(c=0) + p(y|a,c=1)p(c=1)
                    log_py_a_t = logaddexp(
                        log_py_c0_t + log_pc_t.get("copy", 0),
                        log_py_ac1_t + log_pc_t.get("copy", 1),
                    )
                elif self.jointcopy:
                    # p(y|a) = p(y|c=0) + p(y|a,c=1)
                    log_py_a_t = logaddexp(
                        log_py_c0_t,
                        log_py_ac1_t,
                    )
                else:
                    raise NotImplementedError

                if self.qc:
                    # train log_qc
                    Eqac_log_py_a_t = (qc_t.get("copy", 0) * log_py_c0_t.detach() +
                        qc_t.get("copy", 1) * log_py_ac1_t.detach()).mean("k")
                    qc_nll_t = -Eqac_log_py_a_t[y_maskt].sum()
                    # kl[qc || pc]
                    lpct = math.log(.5) if self.jointcopy else log_pc_t.detach()
                    # can sum over copy first but by fubini doesn't matter
                    klc_t = (qc_t * (log_qc_t - lpct)).sum()
                    klc += klc_t
                else:
                    klc = 0
                    qc_nll_t = 0

                y_maskt = y_t.ne(1)
                nll_t = -log_py_a_t.mean("k")[y_maskt].sum()
                kl_e_t = self.kl(qe_t, log_qe_t, log_pe_t, ue_info.lengths, "e").sum()
                kl_t_t = self.kl(qt_t, log_qt_t, log_pt_t, ut_info.lengths, "t").sum()
                kl_t = kl_e_t + kl_t_t
                nll += nll_t.detach()
                kl += kl_t.detach()

                attn_nll_t = -(log_qe_t + log_qt_t)[v2d == y_t].sum() if supattn else 0

                # Break up backprop over time
                if learn:
                    Bt = log_py_Ea_t
                    rewardt = (log_py_a_t - Bt).detach() * v_f_log_p_t
                    rewardt = -rewardt.mean("k")[y_maskt].sum()
                    if self.kl_anneal_steps > 0:
                        kl_coef = self.kl_coef(self.steps, self.kl_anneal_steps)
                        kl_t = kl_t * kl_coef
                    if self.qc:
                        (rewardt + nll_t + kl_t + attn_nll_t + klc_t + qc_nll_t).div(nwt).backward()
                    else:
                        (rewardt + nll_t + kl_t + attn_nll_t).div(nwt).backward()
                    #import pdb; pdb.set_trace()
                    # acc grads
                    #rnn_grads.append(rnn_t._new(rnn_t.values.grad))
                    if log_py_Ea_t.values.grad is not None:
                        log_py_Ea_grads.append(log_py_Ea_t._new(log_py_Ea_t.values.grad))
                    if not self.jointcopy:
                        log_pc_grads.append(log_pc_t._new(log_pc_t.values.grad))
                    log_pe_grads.append(log_pe_t._new(log_pe_t.values.grad))
                    log_pt_grads.append(log_pt_t._new(log_pt_t.values.grad))
                    if log_qc_t.values.grad is not None:
                        log_qc_grads.append(log_qc_t._new(log_qc_t.values.grad))
                    log_qe_grads.append(log_qe_t._new(log_qe_t.values.grad))
                    log_qt_grads.append(log_qt_t._new(log_qt_t.values.grad))
                    v2dx_grads.append(v2dx_t._new(v2dx_t.values.grad))
                    #import pdb; pdb.set_trace()
                    #if self.steps > 200:
                        #import pdb; pdb.set_trace()


                # Add these back in later
                e_s.append(e_s_t.detach())
                e_s_log_p.append(e_s_log_p_t.detach())
                t_s.append(t_s_t.detach())
                t_s_log_p.append(t_s_log_p_t.detach())
                log_py_c0.append(log_py_c0_t.detach())
                log_py_ac1.append(log_py_ac1_t.detach())
                log_py_a.append(log_py_a_t.detach())
            else:
                # exact
                raise NotImplementedError


        if learn:
            self.numexamples += 1
            self.steps += 1
            #rnn_grads = ntorch.cat(rnn_grads, "time")
            if log_py_Ea_grads:
                log_py_Ea_grads = ntorch.cat(log_py_Ea_grads, "time")
            if not self.jointcopy:
                log_pc_grads = ntorch.cat(log_pc_grads, "time")
            log_pe_grads = ntorch.cat(log_pe_grads, "time")
            log_pt_grads = ntorch.cat(log_pt_grads, "time")
            if log_qc_grads:
                log_qc_grads = ntorch.cat(log_qc_grads, "time")
            log_qe_grads = ntorch.cat(log_qe_grads, "time")
            log_qt_grads = ntorch.cat(log_qt_grads, "time")
            v2dx_grads = sum(v2dx_grads)

            bwd_outputs = [
                log_pe.values, log_pt.values,
                log_qe.values, log_qt.values,
                v2dx.values,
            ]
            bwd_grads = [
                log_pe_grads.values, log_pt_grads.values,
                log_qe_grads.values, log_qt_grads.values,
                v2dx_grads.values,
            ]
            if log_py_Ea_grads:
                bwd_outputs.append(log_py_Ea.values)
                bwd_grads.append(log_py_Ea_grads.values)
            if log_qc_grads:
                bwd_outputs.append(log_qc.values)
                bwd_grads.append(log_qc_grads.values)
            if not self.jointcopy:
                bwd_outputs.append(log_pc.values)
                bwd_grads.append(log_pc_grads.values)

            backward(bwd_outputs, bwd_grads)

        if K > 0:
            e_s = ntorch.cat(e_s, "time")
            e_s_log_p = ntorch.cat(e_s_log_p, "time")
            t_s = ntorch.cat(t_s, "time")
            t_s_log_p = ntorch.cat(t_s_log_p, "time")
        log_py_c0 = ntorch.cat(log_py_c0, "time")
        log_py_ac1 = ntorch.cat(log_py_ac1, "time")
        log_py_a = ntorch.cat(log_py_a, "time")
        #log_py = ntorch.cat(log_py, "time")
        #log_pa_y = ntorch.cat(log_pa_y, "time")

        rvinfo = RvInfo(
            log_py     = None,
            log_py_a   = log_py_a,
            log_py_c0  = log_py_c0,
            log_py_ac1 = log_py_ac1,
            log_pe     = log_pe,
            log_pt     = log_pt,
            log_qe_y   = log_qe,
            log_qt_y   = log_qt,
            log_pc     = log_pc,
            log_qc_y   = log_qc,
            e_s        = e_s,
            e_s_log_p  = e_s_log_p,
            t_s        = t_s,
            t_s_log_p  = t_s_log_p,
            log_py_Ea  = log_py_Ea,
        )
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
    def marginal_nll(self,
        x, s, x_info, r, r_info, vt, y, y_info,
        T=128, E=32, R=4, learn=False,
    ):
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
    def kl(self, qa, log_qa, log_pa, lens, dim):
        kl = []
        for i, l in enumerate(lens.tolist()):
            qa0 = qa.get("batch", i).narrow(dim, 0, l)
            log_qa0 = log_qa.get("batch", i).narrow(dim, 0, l)
            log_pa0 = log_pa.get("batch", i).narrow(dim, 0, l)
            kl0 =  qa0 * (log_qa0 - log_pa0)
            infmask = log_qa0 != float("-inf")
            # workaround for namedtensor bug that puts empty tensors on different devices
            kl0 = kl0._new(kl0.values.where(infmask.values, torch.zeros_like(kl0.values))).sum(dim)
            kl.append(kl0)
        return ntorch.stack(kl, "batch")

    def kl_coef(self, steps, kl_anneal_steps):
        return (1 if steps > kl_anneal_steps
            else float(steps) / float(kl_anneal_steps))

    def weight_coef(self, steps, anneal_steps):
        return (1 if steps > anneal_steps
            else float(steps) / float(anneal_steps))

