from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#from sklearn.metrics import f1_score, precision_score, recall_score

from torch.nn.utils import clip_grad_norm_ as clip_

from namedtensor import NamedTensor

from .iebase import Ie
from .fns import pr

class Lm(nn.Module):
    PAD = "<pad>"

    def _loop(
        self, iter, optimizer=None, clip=0, learn=False, re=None, supattn=False, supcopy=False,
        T=None, E=None, R=None,
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

                ue, ue_info = batch.uentities
                ut, ut_info = batch.utypes

                v2d = batch.v2d
                vt2d = batch.vt2d

                vt, vt_info = batch.values_text

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                T = y.shape["time"]
                N = y.shape["batch"]
                #if states is None:
                states = self.init_state(N)
                logits, _ = self(
                    x, states, x_info, r, r_info, vt,
                    ue, ue_info, ut, ut_info,
                    v2d, vt2d, y
                )

                nll = self.loss(logits, y)

                if self.maskedc:
                    slist = [
                        "one", "two", "three", "four", "five",
                        "six", "seven", "eight", "nine",
                        "1", "2", "3", "4", "5",
                        "6", "7", "8", "9",
                    ]
                    mask = sum(y.eq(self.Vx.stoi[x]) for x in slist)
                    nllc = -self.log_pc[mask].sum()
                else:
                    nllc = 0

                kl = 0
                nelbo = nll + kl
                if learn:
                    (nelbo + nllc).div(nwords.item()).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_loss += nelbo.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        return cum_loss, cum_ntokens

    def train_epoch(
        self, iter, optimizer, clip=0, re=None, supattn=False, supcopy=False,
        T=64, E=32, R=4,
    ):
        return self._loop(
            iter=iter, learn=True,
            optimizer=optimizer, clip=clip, re=re,
            supattn=supattn,
            supcopy=supcopy,
            T = T,
            E = E,
            R = R,
        )

    def validate(self, iter, T=64, E=32, R=4):
        return self._loop(
            iter=iter, learn=False,
            T = T,
            E = E,
            R = R,
        )

    def forward(self):
        raise NotImplementedError

    def rnn_parameters(self):
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def loss(self, logits, y):
        yflat = (y
            .stack(("batch", "time"), "batch")
            .chop("batch", ("batch", "vocab"), vocab=1))
        lflat = logits.stack(("batch", "time"), "batch")
        return -(lflat
            .log_softmax("vocab")
            .gather("vocab", yflat, "vocab")[yflat != 1]
            .sum())

    def _ie_loop(
        self, iter, optimizer=None, clip=0, learn=False, re=None,
        T=None, E=None, R=None,
    ):
        self.train(learn)
        context = torch.enable_grad if learn else torch.no_grad

        if not self.v2d:
            self.copy_x_to_v()

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        cum_TP, cum_FP, cum_FN, cum_TM, cum_TG = 0, 0, 0, 0, 0
        batch_TP, batch_FP, batch_FN, batch_TM, batch_TG = 0, 0, 0, 0, 0
        # for debugging
        cum_e_correct = 0
        cum_t_correct = 0
        cum_correct = 0
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

                vt, vt_info = batch.values_text

                ue, ue_info = batch.uentities
                ut, ut_info = batch.utypes

                v2d = batch.v2d
                vt2d = batch.vt2d

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                T = y.shape["time"]
                N = y.shape["batch"]
                #if states is None:
                states = self.init_state(N)
                logits, _ = self(
                    x, states, x_info, r, r_info, vt,
                    ue, ue_info, ut, ut_info, v2d, vt2d)
                nll = self.loss(logits, y)

                # only for crnnlma though
                """
                cor, ecor, tcor, tot = Ie.pat1(
                    self.ea.rename("els", "e"),
                    self.ta.rename("els", "t"),
                    batch.ie_d,
                )
                """

                #ds = batch.ie_d
                etvs = batch.ie_etv

                #num_cells = batch.num_cells
                num_cells = float(sum(len(d) for d in etvs))
                log_pe = self.ea.cpu()
                log_pt = self.ta.cpu()
                log_pc = self.log_pc.cpu()
                ue = ue.cpu()
                ut = ut.cpu()

                # need log_pv, need to check if self.v2d
                # batch x time x hid
                h = self.lutx(y).values
                log_pv = torch.einsum("nth,vh->ntv", [h, self.lutv.weight.data]).log_softmax(-1)
                log_pv = NamedTensor(log_pv, names=("batch", "time", "v"))

                tp, fp, fn, tm, tg = pr(
                    etvs, ue, ut, log_pe, log_pt, log_pv,
                    log_pc = log_pc,
                    lens = lens,
                    Ve = self.Ve,
                    Vt = self.Vt,
                    Vv = self.Vv,
                    Vx = self.Vx,
                    text = text,
                )
                cum_TP += tp
                cum_FP += fp
                cum_FN += fn
                cum_TM += tm
                cum_TG += tg

                batch_TP += tp
                batch_FP += fp
                batch_FN += fn
                batch_TM += tm
                batch_TG += tg

                # For debugging
                ds = batch.ie_et_d
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

                        cum_correct += correct
                        cum_e_correct += e_correct
                        cum_t_correct += t_correct

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
                        p = batch_TP / (batch_TP + batch_FP),
                        r = batch_TP / (batch_TP + batch_FN),
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_TP, batch_FP, batch_FN, batch_TM, batch_TG = 0, 0, 0, 0, 0

        print(f"DBG acc: {cum_correct / cum_ntokens} | E acc: {cum_e_correct / cum_ntokens} | T acc: {cum_t_correct / cum_ntokens}")
        print(f"p: {cum_TP / (cum_TP + cum_FP)} || r: {cum_TP / (cum_TP + cum_FN)}")
        print(f"total supervised cells: {cum_ntokens}")
        return cum_loss, cum_ntokens

    # IE stuff?
    def train_ie_epoch(self, iter, optimizer, clip=0, re=None, supattn=False, supcopy=False, T=64, E=32, R=4):
        return self._ie_loop(
            iter=iter, learn=True,
            optimizer=optimizer, clip=clip, re=re,
            T = T,
            E = E,
            R = R,
        )

    def validate_ie(self, iter, T=64, E=128, R=4):
        return self._ie_loop(
            iter=iter, learn=False,
            T = T,
            E = E,
            R = R,
        )

    def copy_x_to_v(self):
        self.lutv.weight.data.copy_(
            self.lutx.weight.data.index_select(
                0,
                torch.LongTensor([self.Vx.stoi[word] for word in self.Vv.itos])
                    .to(self.lutx.weight.data.device),
            )
        )
