
from copy import deepcopy
from tqdm import tqdm

from itertools import zip_longest

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_
from torch.distributions import kl_divergence
from torch.distributions.categorical import Categorical

from .base import Lm
from .fns import logaddexp
from .lvm import Lvm


class RvInfo:
    def __init__(self,
        log_py_v     = None,
        log_py_Ev     = None,
        log_pv_y = None,
        log_pv = None,
        log_py = None,
        log_py_a   = None,
        log_py_c0  = None,
        log_py_ac1 = None,
        log_pe     = None,
        log_pt     = None,
        log_qe_y   = None,
        log_qt_y   = None,
        log_pc     = None,
        log_qc_y   = None,
        e_s        = None,
        e_s_log_p  = None,
        t_s        = None,
        t_s_log_p  = None,
        log_py_Ea  = None,
    ):
        self.log_py_v     = log_py_v
        self.log_py_Ev     = log_py_Ev
        self.log_pv_y = log_pv_y
        self.log_pv = log_pv
        self.log_py     = log_py   
        self.log_py_c0 = log_py_c0
        self.log_py_a   = log_py_a  
        self.log_py_ac1 = log_py_ac1
        self.log_pe     = log_pe   
        self.log_pt     = log_pt   
        self.log_qe_y   = log_qe_y 
        self.log_qt_y   = log_qt_y
        self.log_pc     = log_pc
        self.log_qc_y   = log_qc_y 
        self.e_s        = e_s      
        self.e_s_log_p  = e_s_log_p
        self.t_s        = t_s      
        self.t_s_log_p  = t_s_log_p
        self.log_py_Ea  = log_py_Ea


    def __str__(self):
        attrs = [
            "log_py",
            "log_py_a",
            "log_py_c0",
            "log_pya_c1",
            "log_pe",   
            "log_pt",   
            "log_qe_y",   
            "log_qt_y",   
            "log_pa",
            "log_pa_y",
            "log_pc_a",
            "log_pc_y",
            "e_s",
            "e_s_log_p",
            "t_s",
            "t_s_log_p",
        ]
        return "\n".join(
            f"{attr}: {getattr(self, attr).shape}"
            for attr in attrs
            if hasattr(self, attr) and getattr(self, attr) is not None
        )


class LvmNc(Lvm):

    def _batch(
        self,
        batch,
        T=64, E=32, R=4,
        learn = False,
        sup = False,
    ):
        text, x_info = batch.text
        text_info = deepcopy(x_info)
        mask = x_info.mask
        lens = x_info.lengths

        L = text.shape["time"]
        x = text.narrow("time", 0, L-1)
        y = text.narrow("time", 1, L-1)
        #x = text[:-1]
        #y = text[1:]
        x_info.lengths.sub_(1)

        e, e_info = batch.entities
        t, t_info = batch.types
        v, v_info = batch.values
        lene = e_info.lengths
        lent = t_info.lengths
        lenv = v_info.lengths
        #rlen, N = e.shape
        #r = torch.stack([e, t, v], dim=-1)
        r = [e, t, v]
        assert (lene == lent).all()
        lenr = lene
        r_info = e_info

        # values text
        vt, vt_info = batch.values_text
        # DBG
        #couldve_copied += (vt == y).transpose("batch", "time", "els").any(-1).sum().item()

        ue, ue_info = batch.uentities
        ut, ut_info = batch.utypes

        v2d = batch.v2d

        # should i include <eos> in ppl?
        nwords = y.ne(1).sum()
        # assert nwords == lens.sum()
        #T = y.shape["time"]
        N = y.shape["batch"]
        #if states is None:
        states = self.init_state(N)

        # should i include <eos> in ppl? no, should not.
        mask = y.ne(1) #* y.ne(3)
        nwords = mask.sum()

        # ugh.......refactor this
        if sup:
            #rvinfo, _, nll_pv, nll_py_v, nll_qv_y, v_total, y_total = self.forward_sup(
            return self.forward_sup(
                text, text_info,
                x, states, x_info, r, r_info, vt,
                ue, ue_info,
                ut, ut_info,
                v2d,
                y, x_info,
                T = T, E = E, R = R,
                learn = learn,
            )
            nll = nll_py_v
            klv = nll_pv
            kla = nll_qv_y
            N = v_total.item()
        else:
            #rvinfo, _, nll, klv, kla = self(
            return self(
                text, text_info,
                x, states, x_info, r, r_info, vt,
                ue, ue_info,
                ut, ut_info,
                v2d,
                y, x_info,
                T = T, E = E, R = R,
                learn = learn,
            )


    def _loop(
        self, iter, supiter=[], optimizer=None, clip=0,
        learn=False, re=None,
        elbo=True,
        T=64, E=32, R=4,
        supattn=False, supcopy=False,
    ):
        context = torch.enable_grad if learn else torch.no_grad

        # DBG
        self.copied = 0
        self.couldve_copied = 0

        # unsup
        cum_loss = 0.
        cum_ntokens = 0.
        cum_rx = 0.
        cum_klv = 0.
        cum_kla = 0.
        batch_loss = 0.
        batch_ntokens = 0.
        batch_rx = 0.
        batch_klv = 0.
        batch_kla = 0.
        cum_N = 0.
        batch_N = 0.

        # sup
        cum_nllpy = 0.
        cum_nllpv = 0.
        cum_nllqv = 0.
        cum_Ny = 0.
        cum_Nv = 0.
        batch_nllpy = 0.
        batch_nllpv = 0.
        batch_nllqv = 0.
        batch_Ny = 0.
        batch_Nv = 0.

        ziter = zip_longest(iter, supiter)

        states = None
        with context():
            titer = tqdm(ziter) if learn else ziter
            for i, (batch, supbatch) in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                if batch is not None:
                    rvinfo, _, nll, klv, kla, Nv, Ny = self._batch(
                        batch,
                        T = T, E = E, R = R, # ?
                        learn = learn,
                        sup = False,
                    )
                    nelbo = nll + klv + kla

                    cum_rx += nll.item()
                    batch_rx += nll.item()
                    cum_klv += klv.item()
                    batch_klv += klv.item()
                    cum_kla += kla.item()
                    batch_kla += kla.item()
                    cum_loss += nelbo.item() if not hasattr(self, "nokl") else nll.item()
                    batch_loss += nelbo.item()

                    cum_ntokens += Ny.item()
                    batch_ntokens += Ny.item()
                    cum_N += Nv.item()
                    batch_N += Nv.item()
                if supbatch is not None:
                    (
                        rvinfosup, _,
                        nll_pv, nll_py_v, nll_qv_y,
                        v_total, y_total
                    ) = self._batch(
                        supbatch,
                        T = T, E = E, R = R,
                        learn = learn,
                        sup = True,
                    )
                    cum_nllpy += nll_py_v.item()
                    cum_nllpv += nll_pv.item()
                    cum_nllqv += nll_qv_y.item()
                    cum_Ny += y_total.item()
                    cum_Nv += v_total.item()
                    batch_nllpy += nll_py_v.item()
                    batch_nllpv += nll_pv.item()
                    batch_nllqv += nll_qv_y.item()
                    batch_Ny += y_total.item()
                    batch_Nv += v_total.item()

                if learn:
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                    optimizer.step()

                if re is not None and i % re == -1 % re:
                    titer.set_postfix(
                        elbo = batch_loss / batch_ntokens if batch_loss != 0 else 0,
                        nll = batch_rx / batch_ntokens if batch_rx != 0 else 0,
                        klv = batch_klv / cum_N if batch_klv != 0 else 0,
                        kla = batch_kla / batch_N if batch_kla != 0 else 0,
                        py = batch_nllpy / batch_Ny if batch_nllpy != 0 else 0,
                        pv = batch_nllpv / batch_Nv if batch_nllpv != 0 else 0,
                        qv = batch_nllqv / batch_Nv if batch_nllqv != 0 else 0,
                        gnorm = gnorm,
                    )
                    batch_loss = 0.
                    batch_ntokens = 0.
                    batch_N = 0.
                    batch_rx = 0.
                    batch_klv = 0.
                    batch_kla = 0.
                    # sup
                    batch_nllpy = 0.
                    batch_nllpv = 0.
                    batch_nllqv = 0.
                    batch_Ny = 0.
                    batch_Nv = 0.
        #print(f"COPIED: {self.copied:,} vs {couldve_copied:,} / {cum_ntokens:,}")
        if cum_N == 0:
            cum_N = 1
        if cum_ntokens == 0:
            cum_ntokens = 1
        if cum_Ny == 0:
            cum_Ny = 1
        if cum_Nv == 0:
            cum_Nv = 1
        print(f"NLL: {cum_rx / cum_ntokens} || KLv: {cum_klv / cum_N} || KLa: {cum_kla / cum_N}")
        print(f"py: {cum_nllpy / cum_Ny} || pv: {cum_nllpv / cum_Nv} || qv: {cum_nllqv / cum_Nv}")
        return cum_loss, cum_ntokens


    def train_epoch(
        self, iter=[], supiter=[], optimizer=None, clip=0, re=None, supattn=False, supcopy=False,
        T=64, E=32, R=4,
    ):
        return self._loop(
            iter = iter,
            supiter = supiter,
            learn = True,
            optimizer=optimizer,
            clip=clip, re=re,
            supattn=supattn,
            supcopy=supcopy,
            T = T,
            E = E,
            R = R,
        )

    def validate(self, iter=[], supiter=[], T=64, E=32, R=4, sup=False,):
        return self._loop(
            iter=iter,
            supiter = supiter,
            learn=False,
            T = T,
            E = E,
            R = R,
        )

    def validate_marginal(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=False, exact=True, elbo=False, T=T, E=E, R=R)

    def train_marginal(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=True, exact=True, elbo=False, T=T, E=E, R=R)

    def validate_elbo(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=False, exact=True, elbo=True, T=T, E=E, R=R)

    def train_elbo(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=True, exact=True, elbo=True, T=T, E=E, R=R)

    def validate_elbo_sample(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=False, exact=False, elbo=True, T=T, E=E, R=R)

    def train_elbo_sample(self, iter, T=64, E=32, R=4):
        return self._loop(iter=iter, learn=True, exact=False, elbo=True, T=T, E=E, R=R)

    ## IE STUFF
    def _ie_loop(
        self, iter, optimizer=None, clip=0, learn=False, re=None,
        T=256, E=None, R=None,
    ):
        self.train(learn)
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        cum_e_correct = 0
        cum_t_correct = 0
        batch_e_correct = 0
        batch_t_correct = 0
        cum_correct = 0
        batch_correct = 0
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, x_info = batch.text
                mask = x_info.mask
                lens = x_info.lengths

                L = text.shape["time"]
                x = text.narrow("time", 0, L-1)
                y = text.narrow("time", 1, L-1)
                #x = text[:-1]
                #y = text[1:]
                x_info.lengths.sub_(1)

                e, e_info = batch.entities
                t, t_info = batch.types
                v, v_info = batch.values
                lene = e_info.lengths
                lent = t_info.lengths
                lenv = v_info.lengths
                #rlen, N = e.shape
                #r = torch.stack([e, t, v], dim=-1)
                r = [e, t, v]
                assert (lene == lent).all()
                lenr = lene
                r_info = e_info

                # values text
                vt, vt_info = batch.values_text

                ue, ue_info = batch.uentities
                ut, ut_info = batch.utypes

                v2d = batch.v2d

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                N = y.shape["batch"]
                #if states is None:
                self.K = 1 # whatever
                states = self.init_state(N)
                rvinfo, _, nll, kl = self(
                    x, states, x_info, r, r_info, vt,
                    ue, ue_info, ut, ut_info, v2d,
                    y = y, y_info = x_info,
                    T = T,
                )

                # only for crnnlma though
                """
                cor, ecor, tcor, tot = Ie.pat1(
                    self.ea.rename("els", "e"),
                    self.ta.rename("els", "t"),
                    batch.ie_d,
                )
                """

                ds = batch.ie_d

                #num_cells = batch.num_cells
                num_cells = float(sum(len(d) for d in ds))
                log_pe = rvinfo.log_qe_y.rename("els", "e") if not self.evalp else rvinfo.log_pe.rename("els", "e")
                log_pt = rvinfo.log_qt_y.rename("els", "t") if not self.evalp else rvinfo.log_pt.rename("els", "t")

                # calculate accuracy
                #import pdb; pdb.set_trace()
                for batch, d in enumerate(ds):
                    for t, (es, ts) in d.items():
                        #t = t + 1
                        _, e_max = log_pe.get("batch", batch).get("time", t).max("e")
                        _, t_max = log_pt.get("batch", batch).get("time", t).max("t")
                        e_preds = ue.get("batch", batch).get("els", e_max.item())
                        t_preds = ut.get("batch", batch).get("els", t_max.item())

                        correct = (es.eq(e_preds) * ts.eq(t_preds)).any().float().item()
                        e_correct = es.eq(e_preds).any().float().item()
                        t_correct = ts.eq(t_preds).any().float().item()
                        #import pdb; pdb.set_trace()

                        cum_correct += correct
                        batch_correct += correct
                        cum_e_correct += e_correct
                        cum_t_correct += t_correct
                        batch_e_correct += e_correct
                        batch_t_correct += t_correct
                #import pdb; pdb.set_trace()
                if learn:
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nll.item()
                cum_ntokens += num_cells
                batch_loss += nll.item()
                batch_ntokens += num_cells
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(
                        loss = batch_loss / batch_ntokens,
                        gnorm = gnorm,
                        acc = batch_correct / batch_ntokens,
                        e = batch_e_correct / batch_ntokens,
                        t = batch_t_correct / batch_ntokens,
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_correct = 0
                    batch_e_correct = 0
                    batch_t_correct = 0

        print(f"acc: {cum_correct / cum_ntokens} | E acc: {cum_e_correct / cum_ntokens} | T acc: {cum_t_correct / cum_ntokens}")
        print(f"total supervised cells: {cum_ntokens}")
        return cum_loss, cum_ntokens

