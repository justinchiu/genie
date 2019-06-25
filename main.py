
import argparse
import random
import json
import sys
import os
import pathlib

import numpy as np


import torch
import torch.optim as optim

#from torchtext.data import BucketIterator

import data.rotowire as data
from data.rotowire import RotoExample

import data.nyt as nytdata
from data.nyt import NytExample

from models.rnnlm import RnnLm

from models.crnnlm import CrnnLm

from models.crnnlma import CrnnLmA
from models.crnnlmsa import CrnnLmSa
from models.crnnlmca import CrnnLmCa
from models.crnnlmeqca import CrnnLmEqca

from models.lvm import Lvm

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# debug
#torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="rotowire",
        # "nyt/nyt10"
        type=str,
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "rotowire",
            "nyt",
        ],
        default="rotowire",
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--bsz", default=32, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--exactepochs", default=4, type=int)

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--pat", default=0, type=int)
    parser.add_argument("--dp", default=0, type=float)
    parser.add_argument("--wdp", default=0, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    parser.add_argument("--klannealsteps", default=0, type=int)

    parser.add_argument("--maxlen", default=-1, type=int)
    parser.add_argument("--sentences", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--minfreq", default=1, type=int)

    parser.add_argument("--T", default=64, type=int, help="Time split")
    parser.add_argument("--E", default=32, type=int, help="Entity split")
    parser.add_argument("--R", default=4, type=int, help="Relation (type) split")
    parser.add_argument("--K", default=16, type=int, help="total number of particles")
    parser.add_argument("--Kq", default=16, type=int, help="top Kq from q(z) to enumerate over")
    parser.add_argument("--Kl", default=16, type=int, help="top Kl from l(z)")
    parser.add_argument("--Kb", default=16, type=int, help="Samples for baseline if not LOO")
    parser.add_argument("--loo", action="store_true", help="LOO baseline")
    parser.add_argument("--mode", choices=["elbo", "marginal", "iwae"], default="elbo")
    parser.add_argument("--q", choices=["pa", "qay", "pay"], default="qay")

    parser.add_argument("--optim", choices=["Adam", "SGD"])

    # Adam
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD
    parser.add_argument("--mom", type=float, default=0)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Model
    parser.add_argument(
        "--model",
        choices=[
            "sa",
            "rnnlm", "crnnlm", "crnnmlm", "HMM", "HSMM",
            "crnnlma",
            "crnnlmsa", # soft (lm + soft attn)
            "crnnlmca",
            "crnnlmeqca", # hard (lm + hard attn)
            "nccrnnlm",
        ],
        default="rnnlm"
    )

    parser.add_argument("--attn", choices=["hard", "gumbel"], default="hard")
    parser.add_argument("--noattndbg", action="store_true")
    parser.add_argument("--noattnvalues", action="store_true")
    parser.add_argument("--v2d", action="store_true")
    parser.add_argument("--untie", action="store_true")
    parser.add_argument("--vctxt", action="store_true")
    parser.add_argument("--etvctxt", action="store_true",
        help="unused, but was supposed to be to let content model see entity and value")
    parser.add_argument("--wcv", action="store_true")
    parser.add_argument("--initvy", action="store_true")
    parser.add_argument("--nuisance", action="store_true")
    parser.add_argument("--initu", action="store_true")
    parser.add_argument("--initg", action="store_true")
    parser.add_argument("--tanh", action="store_true")
    parser.add_argument("--bil", action="store_true")
    parser.add_argument("--mlp", action="store_true")

    # dbg crnnlmsa
    parser.add_argument("--hardc", action="store_true")
    parser.add_argument("--fixedc", action="store_true")
    parser.add_argument("--maskedc", action="store_true")

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=256, type=int)
    parser.add_argument("--rnn-sz", default=256, type=int)
    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--brnn", action="store_true")
    parser.add_argument("--inputfeed", action="store_true")
    parser.add_argument("--supattn", action="store_true")
    parser.add_argument("--supcopy", action="store_true")
    parser.add_argument("--jointcopy", action="store_true")
    parser.add_argument("--qc", action="store_true", help="learn qc")
    parser.add_argument("--qcrnn", action="store_true", help="use a sep brnn for qc")
    parser.add_argument("--qconly", action="store_true", help="only train inf net")

    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--tao", type=float, default=1)
    parser.add_argument("--qtao", type=float, default=0.9)
    parser.add_argument("--taoannealsteps", default=0, type=int)
    parser.add_argument("--qwarmupsteps", default=0, type=int)
    parser.add_argument("--qsteps", default=0, type=int)

    parser.add_argument("--cannealsteps", default=0, type=int)
    parser.add_argument("--cwarmupsteps", default=0, type=int)


    parser.add_argument("--save", action="store_true")
    parser.add_argument("--eval-only", type=str, default=None)
    parser.add_argument("--train-from", type=str, default=None)
    parser.add_argument("--only-emb", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--pretrain-emission", type=int, default=0)

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    parser.add_argument("--old-model", action="store_true")
    return parser.parse_args()


args = get_args()
print(args)
#modelname = f"{args.prefix}-{args.dataset}-{args.model}-{args.mode}-es{args.emb_sz}-rs{args.rnn_sz}-b{args.bsz}-lr{args.lr}-lrd{args.lrd}-dp{args.dp}-tw{args.tieweights}-K{args.K}-Ke{args.Ke}-Kl{args.Kl}-ks{args.klannealsteps}-sc{args.supcopy}-q{args.q}-nv{args.noattnvalues}-jc{args.jointcopy}-tf{args.train_from is not None}-qc{args.qc}-qr{args.qcrnn}-t{args.temp}-v2d{args.v2d}-vy{args.initvy}-n{args.nuisance}-iu{args.initu}-ig{args.initg}-t{args.tanh}-b{args.bil}-m{args.mlp}-cs{args.cannealsteps}-cw{args.cwarmupsteps}-pe{args.pretrain_emission}-u{args.untie}-bv{args.bayesv}"
modelname = f"{args.prefix}-{args.model}-es{args.emb_sz}-rs{args.rnn_sz}-b{args.bsz}-lr{args.lr}-dp{args.dp}-tw{args.tieweights}-K{args.K}-Kq{args.Kq}-Kl{args.Kl}-Kb{args.Kb}-ks{args.klannealsteps}-sc{args.supcopy}-q{args.q}-nv{args.noattnvalues}-jc{args.jointcopy}-tf{args.train_from is not None}-qc{args.qc}-qr{args.qcrnn}-t{args.temp}-v2d{args.v2d}-vy{args.initvy}-n{args.nuisance}-t{args.tanh}-b{args.bil}-m{args.mlp}-cs{args.cannealsteps}-cw{args.cwarmupsteps}-pe{args.pretrain_emission}-u{args.untie}-qs{args.qwarmupsteps}-{args.qsteps}"
if args.model == "crnnlmsa":
    modelname = f"dbg-crnnlmsa-{args.prefix}-{args.model}-v2d{args.v2d}-mlp{args.mlp}-hard{args.hardc}-fix{args.fixedc}-mask{args.maskedc}-pe{args.pretrain_emission}"
if args.maxlen > 0:
    modelname = f"dbg-model-maxlen{args.maxlen}"
if args.save:
    pathlib.Path(modelname).mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid >= 0 else "cpu")

# Data

dataset = data.RotoDataset if args.dataset == "rotowire" else nytdata.NytDataset
ENT, TYPE, VALUE, VALUE_TEXT, TEXT = data.make_fields(args.maxlen + 1)
train, valid, test = dataset.splits(
    ENT, TYPE, VALUE, VALUE_TEXT, TEXT,
    path=args.filepath,
    sentences=args.sentences,
    reset=args.reset,
)

data.build_vocab(ENT, TYPE, VALUE, TEXT, train, min_freq=args.minfreq)

# is this enough?
TEXT.vocab.extend(VALUE.vocab)
VALUE_TEXT.vocab = TEXT.vocab

iterator = data.RotowireIterator if args.dataset == "rotowire" else nytdata.NytIterator
train_iter, valid_iter, test_iter = iterator.splits(
    (train, valid, test),
    batch_size = args.bsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)

model_state = None
if args.eval_only:
    eval_args = args
    thing = torch.load(args.eval_only, map_location="cpu")
    model_state, args = thing["model"], thing["args"]
    args.eval_only = True
    # Use new T and E
    args.T = eval_args.T
    args.E = eval_args.E

# Model
if args.model == "rnnlm":
    model = RnnLm(
        V       = TEXT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
    )
elif args.model == "crnnlm":
    model = CrnnLm(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        r_emb_sz = args.emb_sz,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
        inputfeed = args.inputfeed,
        noattn = args.noattndbg,
    )
elif args.model == "crnnlma":
    model = CrnnLmA(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        r_emb_sz = args.emb_sz,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
        inputfeed = args.inputfeed,
        noattnvalues = args.noattnvalues,
        initu = args.initu,
        initg = args.initg,
        v2d = args.v2d,
        initvy = args.initvy,
    )
elif args.model == "crnnlmsa":
    model = CrnnLmSa(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        r_emb_sz = args.emb_sz,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
        inputfeed = args.inputfeed,
        noattnvalues = args.noattnvalues,
        initu = args.initu,
        initg = args.initg,
        v2d = args.v2d,
        initvy = args.initvy,
        mlp = args.mlp,
        hardc = args.hardc,
        fixedc = args.fixedc,
        maskedc = args.maskedc,
    )
elif args.model == "crnnlmeqca":
    model = CrnnLmEqca(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        r_emb_sz = args.emb_sz,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
        inputfeed = args.inputfeed,
        noattnvalues = args.noattnvalues,
        jointcopy = args.jointcopy,
        qc = args.qc,
        qcrnn = args.qcrnn,
        temp = args.temp,
        qconly = args.qconly,
        v2d = args.v2d,
        initvy = args.initvy,
        nuisance = args.nuisance,
        initu = args.initu,
        tanh = args.tanh,
        vctxt = args.vctxt,
        wcv = args.wcv,
        etvctxt = args.etvctxt,
        bil = args.bil,
        mlp = args.mlp,
        untie = args.untie,
    )
model.to(device)
print(model)

if args.train_from:
    thing = torch.load(args.train_from, map_location="cpu")
    model_state, _ = thing["model"], thing["args"]
    if args.only_emb:
        model.lutx.weight.data.copy_(model_state["lutx.weight"])
        model.proj.weight.data.copy_(model_state["proj.weight"])
    else:
        model.load_state_dict(model_state, strict=False)
    if model.untie:
        model.lutgx.weight.data.copy_(model_state["lutx.weight"])
        model.gproj.weight.data.copy_(model_state["proj.weight"])

if args.klannealsteps > 0:
    model.kl_anneal_steps = args.klannealsteps
if args.cannealsteps > 0:
    model.c_anneal_steps = args.cannealsteps
if args.cwarmupsteps > 0:
    model.c_warmup_steps = args.cwarmupsteps

model.K = args.K
model.Kq = args.Kq
model.Kl = args.Kl
model.Kb = args.Kb

model.mode = args.mode
model.q = args.q

model.tao = args.tao
model.qtao = args.qtao

if args.taoannealsteps > 0:
    model.tao_anneal_steps = args.taoannealsteps

if args.qwarmupsteps > 0:
    model.q_warmup_steps = args.qwarmupsteps
model.qsteps = args.qsteps

#dbg
model.ENT = ENT.vocab
model.TYPE = TYPE.vocab
model.VALUE = VALUE.vocab
model.TEXT = TEXT.vocab

if args.eval_only:
    model.load_state_dict(model_state)
    if isinstance(model, Lvm):
        valid_loss, ntok = model.validate_marginal(valid_iter, T=args.T, E=args.E)
    else:
        valid_loss, ntok = model.validate(valid_iter, T=args.T, E=args.E)
    print(f"loss: {valid_loss / ntok}")
    sys.exit(0)

if args.pretrain_emission:
    # pretrain the content emission
    # the lexical match prior is necessary for training the alignment model
    params = list(model.parameters())
    optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))

    print("pretraining content emission model")

    # get lexical match between value and words
    inputs = []
    labels = []
    for i, v in enumerate(VALUE.vocab.itos):
        label = v.split()
        input = [v] * len(label)

        inputs.extend(input)
        labels.extend(label)

    v = VALUE.process(
        [inputs], device=device
    )[0]
    y = TEXT.process(
        [labels], device=device
    )[0].narrow("time", 1, len(labels))

    for e in range(args.pretrain_emission):
        optimizer.zero_grad()
        ev = model.lutv(v).rename("els", "time").rename("v", "ctxt")
        out = model.Wvy1(
            model.Wvy0(ev).tanh()
        ).tanh() if args.mlp else ev
        log_py = model.proj(out).log_softmax("vocab")
        nll = -log_py.gather("vocab", y.repeat("k", 1), "k").sum()
        nll.backward()
        optimizer.step()

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
schedule = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.pat, factor=args.lrd, threshold=1e-3)
#batch = next(iter(train_iter))
# TODO: try truncating sequences early on?
#import pdb; pdb.set_trace()
best_val = float("inf")
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    # Train
    train_loss, tntok = model.train_epoch(
        iter      = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
        supattn   = args.supattn,
        supcopy   = args.supcopy,
        T         = args.T,
        E         = args.E,
        R         = args.R,
    )

    # Validate
    valid_loss, ntok = model.validate(valid_iter, T=args.T, E=args.E)
    print(f"Epoch {e} train loss: {train_loss / tntok} valid loss: {valid_loss / ntok}")
    old_lr = optimizer.param_groups[0]['lr']
    schedule.step(valid_loss / ntok)
    new_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        model.annealed = True
        model.temp = 1

    if valid_loss < best_val:
        best_val = valid_loss
    if args.save:
        savestring = f"{modelname}/{modelname}-e{e}-vloss{valid_loss / ntok:.2f}.pt"
        torch.save({"model": model.state_dict(), "args": args}, savestring)
        print(f"Saved to {savestring}")
