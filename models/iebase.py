from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_

class Ie(nn.Module):
    PAD = "<pad>"
    NONE = "<none>"

    def _ie_loop(
        self, iter, optimizer=None, clip=0, learn=False, re=None,
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
        cum_TP, cum_FP, cum_FN, cum_TM, cum_TG = 0, 0, 0, 0, 0
        batch_TP, batch_FP, batch_FN, batch_TM, batch_TG = 0, 0, 0, 0, 0
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, x_info = batch.text
                mask = x_info.mask
                lens = x_info.lengths

                x = text

                # ie_idx is off by one because of bos
                #ie_idx = batch.ie_idx
                #ie_e = batch.ie_e
                #ie_t = batch.ie_t

                etvs = batch.ie_etv

                #num_cells = batch.num_cells
                num_cells = float(sum(len(d) for d in etvs))

                nll, log_pe, log_pt, log_pv = self(
                    x, x_info, etvs, learn=learn)


                # gather predictions
                batch_positives = []
                for batch, etv in enumerate(etvs):
                    # Get model positives
                    positives = {}
                    T = lens.get("batch", batch).item()
                    for t in range(T):
                        e_p, e_max = log_pe.get("batch", batch).get("time", t).max("e")
                        t_p, t_max = log_pt.get("batch", batch).get("time", t).max("t")
                        v_p, v_preds = log_pv.get("batch", batch).get("time", t).max("v")

                        e_pred = e_max.item()
                        t_pred = t_max.item()
                        v_pred = v_preds.item()
                        if (
                            e_pred != self.Ve.stoi[self.NONE]
                            and t_pred != self.Vt.stoi[self.NONE]
                            and v_pred != self.Vv.stoi[self.NONE]
                        ):
                            key = (e_pred, t_pred, v_pred)
                            score = (e_p + t_p + v_p).item()
                            if key not in positives or score > positives[key]:
                                positives[key] = score
                    batch_positives.append(positives)

                    # Compare against true positives
                    true_positives = set()
                    for es, ts, vs, _ in etv:
                        true_positives |= set(zip(es.tolist(), ts.tolist(), vs.tolist()))

                    total_m = len(positives)
                    total_g = len(true_positives)
                    tp = len(set(positives) & true_positives)
                    fp = len(set(positives) - true_positives)
                    fn = len(true_positives - set(positives))

                    cum_TP += tp
                    cum_FP += fp
                    cum_FN += fn
                    cum_TM += total_m
                    cum_TG += total_g

                    batch_TP += tp
                    batch_FP += fp
                    batch_FN += fn
                    batch_TM += total_m
                    batch_TG += total_g

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
                        p = batch_TP / (batch_TP + batch_FP) if batch_TP > 0 else 0,
                        r = batch_TP / (batch_TP + batch_FN) if batch_TP > 0 else 0,
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_TP, batch_FP, batch_FN, batch_TM, batch_TG = 0, 0, 0, 0, 0

        print(f"p: {cum_TP / (cum_TP + cum_FP) if cum_TP > 0 else 0} || r: {cum_TP / (cum_TP + cum_FN) if cum_TP > 0 else 0}")
        print(f"total supervised cells: {cum_ntokens}")
        return cum_loss, cum_ntokens, cum_TP, cum_FP, cum_FN, cum_TM, cum_TG

    def _vie_loop(
        self, iter, optimizer=None, clip=0, learn=False, re=None,
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
        cum_e_correct = 0
        cum_t_correct = 0
        batch_e_correct = 0
        batch_t_correct = 0
        cum_correct = 0
        batch_correct = 0
        cum_copyable = 0
        batch_copyable = 0
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, x_info = batch.text
                mask = x_info.mask
                lens = x_info.lengths

                x = text

                # ie_idx is off by one because of bos
                #ie_idx = batch.ie_idx
                #ie_e = batch.ie_e
                #ie_t = batch.ie_t

                #ds = batch.ie_d
                ets = batch.ie_et

                #num_cells = batch.num_cells
                #num_cells = float(sum(len(d) for d in ds))

                e, r_info = batch.entities
                t, _ = batch.types
                v, _ = batch.values
                vt, _ = batch.values_text
                num_cells = r_info.mask.sum()

                nll, log_pv, attn = self(
                    x, x_info, e=e, t=t, v=v, r_info=r_info, ets=ets, learn=learn)

                hv = log_pv.max("v")[1]
                correct = (hv == v)[r_info.mask].sum()
                batch_correct += correct
                cum_correct += correct

                vt_nopad = vt.clone()
                vt_nopad[vt_nopad == 1] = -1
                num_copyable = vt_nopad.eq(x).sum()

                batch_copyable += num_copyable
                cum_copyable += num_copyable

                # Deal with n/a tokens
                na_v_idx = self.Vv.stoi["n/a"]
                #import pdb; pdb.set_trace()

                if learn:
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nll
                cum_ntokens += num_cells
                batch_loss += nll
                batch_ntokens += num_cells
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(
                        loss = batch_loss.item() / batch_ntokens.item(),
                        gnorm = gnorm,
                        acc = batch_correct.item() / batch_ntokens.item(),
                        copyable = batch_copyable.item() / batch_ntokens.item(),
                    )
                    batch_loss = 0
                    batch_ntokens = 0
                    batch_correct = 0
                    batch_copyable = 0

        print(f"acc: {cum_correct.item() / cum_ntokens.item()} || copyable: {cum_copyable.item() / cum_ntokens.item()}")
        print(f"total supervised cells: {cum_ntokens.item()}")
        return cum_loss.item(), cum_ntokens.item()


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

    def train_vie_epoch(self, iter, optimizer, clip=0, re=None, supattn=False, supcopy=False, T=64, E=32, R=4):
        return self._vie_loop(
            iter=iter, learn=True,
            optimizer=optimizer, clip=clip, re=re,
            T = T,
            E = E,
            R = R,
        )

    def validate_vie(self, iter, T=64, E=128, R=4):
        return self._vie_loop(
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

    @staticmethod
    def pat1(log_pe, log_pt, ds):
        total_correct, total_e_correct, total_t_correct, total = 0, 0, 0, 0
        for batch, d in enumerate(ds):
            for t, (es, ts) in d.items():
                t = t + 1 # off by one because of <bos>
                _, e_preds = log_pe.get("batch", batch).get("time", t).max("e")
                _, t_preds = log_pt.get("batch", batch).get("time", t).max("t")

                correct = (es.eq(e_preds) * ts.eq(t_preds)).any().float().item()
                e_correct = es.eq(e_preds).any().float().item()
                t_correct = ts.eq(t_preds).any().float().item()

                total_correct += correct
                total_e_correct += e_correct
                total_t_correct += t_correct
                total += 1
        return total_correct, total_e_correct, total_t_correct, total
