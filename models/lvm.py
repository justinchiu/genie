
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_
from torch.distributions import kl_divergence
from torch.distributions.categorical import Categorical

from .base import Lm
from .fns import logaddexp


class RvInfo:
    def __init__(self,
        log_py     = None,
        log_py_a   = None,
        log_py_ac0 = None,
        log_py_ac1 = None,
        log_pa     = None,
        log_pa_y   = None,
        log_pc_a   = None,
        log_pc_y   = None,
        a_s        = None,
        a_s_log_p  = None,
        log_py_Ea  = None,
    ):
        self.log_py     = log_py   
        self.log_py_ac0 = log_py_ac0
        self.log_py_a   = log_py_a  
        self.log_py_ac1 = log_py_ac1
        self.log_pa     = log_pa   
        self.log_pa_y   = log_pa_y 
        self.log_pc_a   = log_pc_a
        self.log_pc_y   = log_pc_y 
        self.a_s        = a_s      
        self.a_s_log_p  = a_s_log_p
        self.log_py_Ea  = log_py_Ea


    def __str__(self):
        attrs = [
            "log_py",
            "log_py_a",
            "log_py_ac0",
            "log_pya_c1",
            "log_pa",
            "log_pa_y",
            "log_pc_a",
            "log_pc_y",
            "a_s",
            "a_s_log_p",
        ]
        return "\n".join(
            f"{attr}: {getattr(self, attr).shape}"
            for attr in attrs
            if hasattr(self, attr) and getattr(self, attr) is not None
        )


class Lvm(Lm):
    def _loop(
        self, iter, optimizer=None, clip=0,
        learn=False, re=None,
        exact=False, elbo=True,
        T=64, E=128,
        supattn=False, supcopy=False,
    ):
        context = torch.enable_grad if learn else torch.no_grad

        # DBG
        self.copied = 0
        couldve_copied = 0

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        batch_rx = 0
        batch_kl = 0
        states = None
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
                # DBG
                couldve_copied += (vt == y).sum().item()

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
                if exact:
                    rvinfo, states = self.marginal_nll(
                        x, states, x_info, r, r_info, vt, y, x_info, T=T, E=E, learn=learn)
                    nll = -rvinfo.log_py[mask].sum()
                    cum_loss += nll.item()
                    batch_loss += nll.item()
                else:
                    rvinfo, _, nll, kl = self(
                        x, states, x_info, r, r_info, vt, y, x_info, T=T, E=E, learn=learn)

                    nelbo = nll + kl
                    if learn:
                        if clip > 0:
                            gnorm = clip_(self.parameters(), clip)
                        optimizer.step()
                    cum_rx += nll.item()
                    batch_rx += nll.item()
                    cum_kl += kl.item()
                    batch_kl += kl.item()
                    cum_loss += nelbo.item()
                    batch_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(
                        elbo = batch_loss / batch_ntokens,
                        nll = batch_rx / batch_ntokens,
                        kl = batch_kl / batch_ntokens,
                        gnorm = gnorm,
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_rx = 0
                    batch_kl = 0
        print(f"COPIED: {self.copied:,} vs {couldve_copied:,} / {cum_ntokens:,}")
        print(f"NLL: {cum_rx / cum_ntokens} || KL: {cum_kl / cum_ntokens}")
        return cum_loss, cum_ntokens


    def validate_marginal(self, iter, T=64, E=128):
        return self._loop(iter=iter, learn=False, exact=True, elbo=False, T=T, E=E)

    def train_marginal(self, iter, T=64, E=64):
        return self._loop(iter=iter, learn=True, exact=True, elbo=False, T=T, E=E)

    def validate_elbo(self, iter, T=64, E=64):
        return self._loop(iter=iter, learn=False, exact=True, elbo=True, T=T, E=E)

    def train_elbo(self, iter, T=64, E=64):
        return self._loop(iter=iter, learn=True, exact=True, elbo=True, T=T, E=E)

    def validate_elbo_sample(self, iter, T=64, E=64):
        return self._loop(iter=iter, learn=False, exact=False, elbo=True, T=T, E=E)

    def train_elbo_sample(self, iter, T=64, E=64):
        return self._loop(iter=iter, learn=True, exact=False, elbo=True, T=T, E=E)

