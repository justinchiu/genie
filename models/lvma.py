
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
from .lvm import Lvm


class RvInfo:
    def __init__(self,
        log_py     = None,
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
        v_s        = None,
        v_s_log_p  = None,
        log_py_Ea  = None,
    ):
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
        self.v_s        = v_s      
        self.v_s_log_p  = v_s_log_p
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
            "v_s",
            "v_s_log_p",
        ]
        return "\n".join(
            f"{attr}: {getattr(self, attr).shape}"
            for attr in attrs
            if hasattr(self, attr) and getattr(self, attr) is not None
        )


class LvmA(Lvm):
    def _loop(
        self, iter, optimizer=None, clip=0,
        learn=False, re=None,
        exact=False, elbo=True,
        T=64, E=32, R=4,
        supattn=False, supcopy=False,
    ):
        context = torch.enable_grad if learn else torch.no_grad

        # DBG
        self.copied = 0
        couldve_copied = 0

        nthe = 0
        n4 = 0
        nfour = 0
        n6 = 0
        nsix = 0
        n8 = 0
        neight = 0
        nchi = 0
        nleb = 0

        bthe = 0
        b4 = 0
        bfour = 0
        b6 = 0
        bsix = 0
        b8 = 0
        beight = 0
        bchi = 0
        bleb = 0

        athe = 0
        a4 = 0
        afour = 0
        a6 = 0
        asix = 0
        a8 = 0
        aeight = 0
        achi = 0
        aleb = 0

        abthe = 0
        ab4 = 0
        abfour = 0
        ab6 = 0
        absix = 0
        ab8 = 0
        abeight = 0
        abchi = 0
        ableb = 0

        d1 = 0
        d2 = 0
        d4 = 0
        dn = 0

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
                couldve_copied += (vt == y).transpose("batch", "time", "els").any(-1).sum().item()

                ue, ue_info = batch.uentities
                ut, ut_info = batch.utypes

                v2d = batch.v2d
                vt2d = batch.vt2d

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
                        x, states, x_info, r, r_info, vt,
                        ue, ue_info,
                        ut, ut_info,
                        v2d, vt2d,
                        y, x_info, T=T, E=E, R=R, learn=learn
                    )
                    nll = -rvinfo.log_py[mask].sum()
                    cum_loss += nll.item()
                    batch_loss += nll.item()
                else:
                    rvinfo, _, nll, kl = self(
                        x, states, x_info, r, r_info, vt,
                        ue, ue_info,
                        ut, ut_info,
                        v2d, vt2d,
                        y, x_info, T=T, E=E, R=R, learn=learn,
                        supattn=supattn, # hacky
                        idxs = batch.idxs,
                    )
                    if rvinfo.log_qc_y is None:
                        self.copied += (rvinfo.log_pc.exp().get("copy", 1) > 0.5).sum().item()
                    else:
                        self.copied += (rvinfo.log_qc_y.exp().get("copy", 1) > 0.5).sum().item()

                    nelbo = nll + kl

                    # check p(c|y) for subset of words
                    # likelihood ratio of content vs noncontent
                    pyn = rvinfo.log_py_c0
                    pyc = rvinfo.log_py_ac1

                    ythe = y.eq(self.Vx.stoi["the"])
                    # find "4" and "four"
                    y4 = y.eq(self.Vx.stoi["4"])
                    yfour = y.eq(self.Vx.stoi["four"])
                    # find "6" and "six"
                    y6 = y.eq(self.Vx.stoi["6"])
                    ysix = y.eq(self.Vx.stoi["six"])
                    # find "8" and "eight"
                    y8 = y.eq(self.Vx.stoi["8"])
                    yeight = y.eq(self.Vx.stoi["eight"])
                    ychi = y.eq(self.Vx.stoi["chicago"])
                    yleb = y.eq(self.Vx.stoi["lebron"])


                    nthe += ythe.sum()
                    n4 += y4.sum()
                    nfour += yfour.sum()
                    n6 += y6.sum()
                    nsix += ysix.sum()
                    n8 += y8.sum()
                    neight += yeight.sum()
                    nchi += ychi.sum()
                    nleb += yleb.sum()

                    vt2d_f = vt2d.stack(("e", "t"), "r")
                    alignments = vt2d_f.gather("r", rvinfo.v_s, "k")

                    a4 += alignments.eq(self.Vx.stoi["4"]).values.any(0)[y4.values].sum()
                    afour += alignments.eq(self.Vx.stoi["4"]).values.any(0)[yfour.values].sum()
                    a6 += alignments.eq(self.Vx.stoi["6"]).values.any(0)[y6.values].sum()
                    asix += alignments.eq(self.Vx.stoi["6"]).values.any(0)[ysix.values].sum()
                    a8 += alignments.eq(self.Vx.stoi["8"]).values.any(0)[y8.values].sum()
                    aeight += alignments.eq(self.Vx.stoi["8"]).values.any(0)[yeight.values].sum()
                    achi += alignments.eq(self.Vx.stoi["chicago"]).values.any(0)[ychi.values].sum()
                    aleb += alignments.eq(self.Vx.stoi["lebron"]).values.any(0)[yleb.values].sum()

                    if pyn is not None:
                        better = (pyn < pyc).narrow("k", self.Kl, self.K)

                        betterthe = (pyn.get("k", 0) < pyc.get("k", 0))[ythe]
                        better4 = (pyn.get("k", 0) < pyc.get("k", 0))[y4]
                        betterfour = (pyn.get("k", 0) < pyc.get("k", 0))[yfour]
                        better6 = (pyn.get("k", 0) < pyc.get("k", 0))[y6]
                        bettersix = (pyn.get("k", 0) < pyc.get("k", 0))[ysix]
                        better8 = (pyn.get("k", 0) < pyc.get("k", 0))[y8]
                        bettereight = (pyn.get("k", 0) < pyc.get("k", 0))[yeight]
                        betterchi = (pyn.get("k", 0) < pyc.get("k", 0))[ychi]
                        betterleb = (pyn.get("k", 0) < pyc.get("k", 0))[yleb]

                        bthe += betterthe.sum()
                        b4 += better4.sum()
                        bfour += betterfour.sum()
                        b6 += better6.sum()
                        bsix += bettersix.sum()
                        b8 += better8.sum()
                        beight += bettereight.sum()
                        bchi += betterchi.sum()
                        bleb += betterleb.sum()

                        # if any alignments have the right word AND are better
                        ab4 += (alignments.eq(self.Vx.stoi["4"]) * better).values.any(0)[y4.values].sum()
                        abfour += (alignments.eq(self.Vx.stoi["4"]) * better).values.any(0)[yfour.values].sum()
                        ab6 += (alignments.eq(self.Vx.stoi["6"]) * better).values.any(0)[y6.values].sum()
                        absix += (alignments.eq(self.Vx.stoi["6"]) * better).values.any(0)[ysix.values].sum()
                        ab8 += (alignments.eq(self.Vx.stoi["8"]) * better).values.any(0)[y8.values].sum()
                        abeight += (alignments.eq(self.Vx.stoi["8"]) * better).values.any(0)[yeight.values].sum()
                        abchi += (alignments.eq(self.Vx.stoi["chicago"]) * better).values.any(0)[ychi.values].sum()
                        ableb += (alignments.eq(self.Vx.stoi["lebron"]) * better).values.any(0)[yleb.values].sum()
                    else:
                        qc = rvinfo.log_qc_y.softmax("copy")
                        better = qc.get("copy", 1) > 0.5
                        bthe += better[ythe].sum()
                        b4 += better[y4].sum()
                        bfour += better[yfour].sum()
                        b6 += better[y6].sum()
                        bsix += better[ysix].sum()
                        b8 += better[y8].sum()
                        beight += better[yeight].sum()
                        bchi += better[ychi].sum()
                        bleb += better[yleb].sum()

                        # if any alignments have the right word AND are better
                        ab4 += (alignments.eq(self.Vx.stoi["4"]) * better).values.any(0)[y4.values].sum()
                        abfour += (alignments.eq(self.Vx.stoi["4"]) * better).values.any(0)[yfour.values].sum()
                        ab6 += (alignments.eq(self.Vx.stoi["6"]) * better).values.any(0)[y6.values].sum()
                        absix += (alignments.eq(self.Vx.stoi["6"]) * better).values.any(0)[ysix.values].sum()
                        ab8 += (alignments.eq(self.Vx.stoi["8"]) * better).values.any(0)[y8.values].sum()
                        abeight += (alignments.eq(self.Vx.stoi["8"]) * better).values.any(0)[yeight.values].sum()
                        abchi += (alignments.eq(self.Vx.stoi["chicago"]) * better).values.any(0)[ychi.values].sum()
                        ableb += (alignments.eq(self.Vx.stoi["lebron"]) * better).values.any(0)[yleb.values].sum()

                    maxd, _ = self.delta.max("k")
                    d1 += (maxd > 0.1).sum()
                    d2 += (maxd > 0.2).sum()
                    d4 += (maxd > 0.4).sum()
                    dn += maxd.values.nelement()

                    """
                    print(f"Proportion greater than .1: {float(d1.item()) / float(dn)}")
                    print(f"Proportion greater than .2: {float(d2.item()) / float(dn)}")
                    print(f"Proportion greater than .4: {float(d4.item()) / float(dn)}")
                    """


                    if learn:
                        if clip > 0:
                            gnorm = clip_(self.parameters(), clip)
                        optimizer.step()
                    cum_rx += nll.item()
                    batch_rx += nll.item()
                    cum_kl += kl.item()
                    batch_kl += kl.item()
                    cum_loss += nelbo.item() if not hasattr(self, "nokl") else nll.item()
                    batch_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(
                        elbo = batch_loss / batch_ntokens,
                        nll = batch_rx / batch_ntokens,
                        kl = batch_kl / batch_ntokens,
                        gnorm = gnorm,
                        #b4 = b4.item() / n4.item(),
                        #bfour = bfour.item() / nfour.item(),
                        #b6 = b6.item() / n6.item(),
                        #bsix = bsix.item() / nsix.item(),
                        #b8 = b8.item() / n8.item(),
                        #beight = beight.item() / neight.item(),
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_rx = 0
                    batch_kl = 0
                if re is not None and i % 100 == -1 % 100:
                #if re is not None and i % 10 == -1 % 10:
                    print("better prob")
                    print(f"the: {bthe.item()} / {nthe.item()}")
                    print(f"4: {b4.item()} / {n4.item()}")
                    print(f"four: {bfour.item()} / {nfour.item()}")
                    print(f"6: {b6.item()} / {n6.item()}")
                    print(f"six: {bsix.item()} / {nsix.item()}")
                    print(f"8: {b8.item()} / {n8.item()}")
                    print(f"eight: {beight.item()} / {neight.item()}")
                    print(f"chicago: {bchi.item()} / {nchi.item()}")
                    print(f"lebron: {bleb.item()} / {nleb.item()}")
                    print("correct alignment")
                    print(f"4: {a4.item()} / {n4.item()}")
                    print(f"four: {afour.item()} / {nfour.item()}")
                    print(f"6: {a6.item()} / {n6.item()}")
                    print(f"six: {asix.item()} / {nsix.item()}")
                    print(f"8: {a8.item()} / {n8.item()}")
                    print(f"eight: {aeight.item()} / {neight.item()}")
                    print(f"chi: {achi.item()} / {nchi.item()}")
                    print(f"leb: {aleb.item()} / {nleb.item()}")
                    print("correct alignment and better prob")
                    print(f"4: {ab4.item()} / {n4.item()}")
                    print(f"four: {abfour.item()} / {nfour.item()}")
                    print(f"6: {ab6.item()} / {n6.item()}")
                    print(f"six: {absix.item()} / {nsix.item()}")
                    print(f"8: {ab8.item()} / {n8.item()}")
                    print(f"eight: {abeight.item()} / {neight.item()}")
                    print(f"chi: {abchi.item()} / {nchi.item()}")
                    print(f"leb: {ableb.item()} / {nleb.item()}")

                    print(f"Proportion greater than .1: {float(d1.item()) / float(dn)}")
                    print(f"Proportion greater than .2: {float(d2.item()) / float(dn)}")
                    print(f"Proportion greater than .4: {float(d4.item()) / float(dn)}")

                    words = ["2", "4", "6", "8", "bulls", "lebron", "chicago"]
                    for word in words:
                        if self.v2d:
                            input = self.lutv.weight[self.Vv.stoi[word]]
                        elif self.untie:
                            input = self.lutgx.weight[self.Vx.stoi[word]]
                        else:
                            input = self.lutx.weight[self.Vx.stoi[word]]
                        weight = self.lutx.weight if not self.untie else self.lutgx.weight
                        if self.mlp:
                            probs, idx = (weight @ self.Wvy1(self.Wvy0(NamedTensor(
                                input, names=("ctxt",),
                            )).tanh()).tanh().values).softmax(0).topk(5)
                        else:
                            probs, idx = (weight @ input).softmax(0).topk(5)
                        print(f"{word} probs "+ " || ".join(f"{self.Vx.itos[x]}: {p:.2f}" for p,x in zip(probs.tolist(), idx.tolist())))

        print("better prob")
        print(f"the: {bthe.item()} / {nthe.item()}")
        print(f"4: {b4.item()} / {n4.item()}")
        print(f"four: {bfour.item()} / {nfour.item()}")
        print(f"6: {b6.item()} / {n6.item()}")
        print(f"six: {bsix.item()} / {nsix.item()}")
        print(f"8: {b8.item()} / {n8.item()}")
        print(f"eight: {beight.item()} / {neight.item()}")
        print("correct alignment")
        print(f"4: {a4.item()} / {n4.item()}")
        print(f"four: {afour.item()} / {nfour.item()}")
        print(f"6: {a6.item()} / {n6.item()}")
        print(f"six: {asix.item()} / {nsix.item()}")
        print(f"8: {a8.item()} / {n8.item()}")
        print(f"eight: {aeight.item()} / {neight.item()}")
        print("correct alignment and better prob")
        print(f"4: {ab4.item()} / {n4.item()}")
        print(f"four: {abfour.item()} / {nfour.item()}")
        print(f"6: {ab6.item()} / {n6.item()}")
        print(f"six: {absix.item()} / {nsix.item()}")
        print(f"8: {ab8.item()} / {n8.item()}")
        print(f"eight: {abeight.item()} / {neight.item()}")
        print(f"Proportion greater than .1: {float(d1.item()) / float(dn)}")
        print(f"Proportion greater than .2: {float(d2.item()) / float(dn)}")
        print(f"Proportion greater than .4: {float(d4.item()) / float(dn)}")
        #print(f"COPIED: {self.copied:,} vs {couldve_copied:,} / {cum_ntokens:,}")
        print(f"NLL: {cum_rx / cum_ntokens} || KL: {cum_kl / cum_ntokens}")
        return cum_loss, cum_ntokens

    def train_epoch(self, iter, optimizer, clip=0, re=None, supattn=False, supcopy=False, T=64, E=32, R=4):
        return self._loop(
            iter=iter, learn=True,
            optimizer=optimizer, clip=clip, re=re,
            supattn=supattn,
            supcopy=supcopy,
            T = T,
            E = E,
            R = R
        )

    def validate(self, iter, T=64, E=32, R=4):
        return self._loop(
            iter=iter, learn=False,
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
                vt2d = batch.vt2d

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                N = y.shape["batch"]
                #if states is None:
                self.K = 1 # whatever
                states = self.init_state(N)
                rvinfo, _, nll, kl = self(
                    x, states, x_info, r, r_info, vt,
                    ue, ue_info, ut, ut_info, v2d, vt2d,
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

                #ds = batch.ie_d
                ets = batch.ie_etv

                #num_cells = batch.num_cells
                log_pe = rvinfo.log_qe_y.rename("els", "e") if not self.evalp else rvinfo.log_pe.rename("els", "e")
                log_pt = rvinfo.log_qt_y.rename("els", "t") if not self.evalp else rvinfo.log_pt.rename("els", "t")

                log_pe = log_pe.cpu()
                log_pt = log_pt.cpu()
                ue = ue.cpu()
                ut = ut.cpu()

                #import pdb; pdb.set_trace()

                # calculate accuracy
                #import pdb; pdb.set_trace()
                ds = batch.ie_et_d
                num_cells = float(sum(len(d) for d in ds))

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

