import torch
import torch.nn as nn
import torch.nn.functional as F

from namedtensor import ntorch, NamedTensor


def attn(x, r, mask, temp=1):
    logits = x.dot("rnns", r) if temp == 1 else x.dot("rnns", r) / temp
    logits.masked_fill_(1-mask, float("-inf"))
    log_a = logits.log_softmax("els")
    a = log_a.exp()
    c = r.dot("els", a)
    #c = torch.einsum("...nr,rnh->...nh", [a, r])
    return log_a, a, c

def attn2(x, r, mask):
    logits = x.dot("rnns", r)
    logits.masked_fill_(1-mask, float("-inf"))
    log_a = logits.log_softmax("els")
    a = log_a.exp()
    c = r.dot("els", a)
    #c = torch.einsum("...nr,rnh->...nh", [a, r])
    return log_a, a, c, logits

def logaddexp(log_a, log_b):
    order = log_a._broadcast_order(log_b)
    log_a = log_a._force_order(order)
    log_b = log_b._force_order(order)
    m = log_a._new(log_a.values.max(log_b.values))
    m.masked_fill_(m.abs() == float("inf"), 0)
    return m + ((log_a - m).exp() + (log_b - m).exp()).log()

def concrete_attn(x, r, mask, tao, k=1):
    eps = torch.finfo(x.values.dtype).eps
    log_alpha = x.dot("rnns", r)
    if k == 0:
        log_alpha.masked_fill_(1-mask, float("-inf"))
        return log_alpha
    shape = log_alpha.shape
    u = ntorch.rand(
        (k,) + tuple(shape.values()),
        names = ("k",) + tuple(shape.keys()),
        device = x.values.device
    ).clamp(min=eps, max=1-eps)
    g = -(-u.log()).log()
    logits = (log_alpha + g) / tao
    log_alpha.masked_fill_(1-mask, float("-inf"))
    logits.masked_fill_(1-mask, float("-inf"))
    log_a = logits.log_softmax("els")
    return log_a, log_alpha, g

def kl_qz(qz, log_qz, log_pz, lens, dim, batchdim="time"):
    # batchdim other than "batch", could be time or els
    kl = []
    for i, l in enumerate(lens.tolist()):
        qz0 = qz.get("batch", i).narrow(dim, 0, l)
        log_qz0 = log_qz.get("batch", i).narrow(dim, 0, l)
        log_pz0 = log_pz.get("batch", i).narrow(dim, 0, l)
        kl0 =  qz0 * (log_qz0 - log_pz0)
        infmask = log_qz0 != float("-inf")
        # workaround for namedtensor bug that puts empty tensors on different devices
        kl0 = kl0.transpose(batchdim, dim)
        kl0 = kl0._new(kl0.values.where(infmask.values, torch.zeros_like(kl0.values))).sum(dim)
        kl.append(kl0)
    return ntorch.stack(kl, "batch")


NONE = "<none>"
# Micro precision and recall
def pr(etvs, ue, ut, log_pe, log_pt, log_pv, log_pc, lens, Ve, Vt, Vv, Vx, text=None):
    cum_TP, cum_FP, cum_FN, cum_TM, cum_TG = 0, 0, 0, 0, 0
    # gather predictions
    for batch, etv in enumerate(etvs):
        # Get model positives
        positives = {}
        T = lens.get("batch", batch).item()
        for t in range(T):
            e_p, e_max = log_pe.get("batch", batch).get("time", t).max("e")
            t_p, t_max = log_pt.get("batch", batch).get("time", t).max("t")
            v_p, v_preds = log_pv.get("batch", batch).get("time", t).max("v")
            lc = log_pc.get("batch", batch).get("time", t).get("copy", 1)

            e_preds = ue.get("batch", batch).get("els", e_max.item())
            t_preds = ut.get("batch", batch).get("els", t_max.item())

            e_pred = e_preds.item()
            t_pred = t_preds.item()
            v_pred = v_preds.item()
            c = lc.exp().item()

            """
            print(f"x: {Vx.itos[text.get('batch', batch).get('time', t).item()]}")
            print(f"y: {Vx.itos[text.get('batch', batch).get('time', t+1).item()]}")
            print(f"e: {Ve.itos[e_pred]}")
            print(f"t: {Vt.itos[t_pred]}")
            print(f"v: {Vv.itos[v_pred]}")
            print(c)
            if c > 0.4:
                words = [Vx.itos[x] for x in text.get("batch", batch).tolist()]
                import pdb; pdb.set_trace()
            """
            if (
                e_pred != Ve.stoi[NONE]
                and t_pred != Vt.stoi[NONE]
                and v_pred != Vv.stoi[NONE]
                and c > 0.5
            ):
                key = (e_pred, t_pred, v_pred)
                score = (e_p + t_p + v_p + lc.item()).item()
                if key not in positives or score > positives[key]:
                    positives[key] = score
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

    return cum_TP, cum_FP, cum_FN, cum_TM, cum_TG


if __name__ == "__main__":
    # Ensure the broadcasting is correct
    T, N, H, R = 5, 3, 13, 7
    x, r = torch.randn(T, N, H), torch.randn(R, N, H)
    lenr = torch.LongTensor(N).fill_(R)
    z = attn(x, r, lenr)
    zs = torch.stack([attn(x[t], r, lenr) for t in range(T)], dim=0)
    print((z - zs).abs().max())


