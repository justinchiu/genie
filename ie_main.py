
import argparse
import random
import json
import sys
import os
import pathlib


import torch
import torch.optim as optim

import data
from data import RotoExample
import nytdata
from nytdata import NytExample

from models.rnnie import RnnIe
from models.rnnaie import RnnAie
from models.rnnvie import RnnVie

from models.crnnlma import CrnnLmA
from models.crnnlmsa import CrnnLmSa
from models.crnnlmca import CrnnLmCa
from models.crnnlmqca import CrnnLmQca
from models.crnnlmeqca import CrnnLmEqca
from models.crnnlmesqca import CrnnLmEsqca
from models.crnnlmcqca import CrnnLmCqca
from models.crnnlmecqca import CrnnLmEcqca

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

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

    parser.add_argument("--minfreq", default=1, type=int)

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--pat", default=0, type=int)
    parser.add_argument("--dp", default=0.3, type=float)
    parser.add_argument("--wdp", default=0, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    parser.add_argument("--klannealsteps", default=120000, type=int)

    parser.add_argument("--maxlen", default=-1, type=int)
    parser.add_argument("--sentences", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--numericvalues", action="store_true")
    parser.add_argument("--vie", action="store_true", help="Full slot filling task")

    parser.add_argument("--T", default=64, type=int, help="Time split")
    parser.add_argument("--E", default=32, type=int, help="Entity split")
    parser.add_argument("--R", default=4, type=int, help="Relation (type) split")
    parser.add_argument("--K", default=1, type=int)
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
            "rnnie",
            "rnnaie",
            "crnnlma",
            "rnnvie",
            "crnnlmca",
            "crnnlmsa",
            "crnnlmqca",
            "crnnlmeqca",
            "crnnlmcqca",
            "crnnlmecqca",
        ],
        default="rnnie"
    )

    parser.add_argument("--attn", choices=["emb", "rnn"], default="emb")
    parser.add_argument("--joint", action="store_true")

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=256, type=int)
    parser.add_argument("--rnn-sz", default=256, type=int)
    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--inputfeed", action="store_true")
    parser.add_argument("--supattn", action="store_true")
    parser.add_argument("--supcopy", action="store_true")
    parser.add_argument("--evalp", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--eval-only", type=str, default=None)

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    parser.add_argument("--old-model", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()


args = get_args()
print(args)
modelname = f"{args.prefix}-{args.dataset}-{args.model}-s{args.sentences}-rs{args.rnn_sz}-b{args.bsz}-lr{args.lr}-lrd{args.lrd}-dp{args.dp}-a{args.attn}"
if args.maxlen > 0:
    modelname = f"dbg-model-maxlen{args.maxlen}"
pathlib.Path(modelname).mkdir(parents=True, exist_ok=True)

if args.old_model:
    from models.crnnmlmc_old import CrnnMlmC
else:
    from models.crnnmlmc import CrnnMlmC

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid >= 0 else "cpu")

# Data
dataset = data.RotoDataset if args.dataset == "rotowire" else nytdata.NytDataset
ENT, TYPE, VALUE, VALUE_TEXT, TEXT = data.make_fields(args.maxlen + 1)
train, valid, test = dataset.splits(
    ENT, TYPE, VALUE, VALUE_TEXT, TEXT,
    path = args.filepath,
    sentences = args.sentences,
    reset = args.reset,
    numericvalues = args.numericvalues,
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
    args.evalp = eval_args.evalp
    print("eval args:")
    print(args)


# Model
if args.model == "rnnie":
    model = RnnIe(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        joint   = args.joint,
    )
elif args.model == "rnnaie":
    model = RnnAie(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        attn    = args.attn,
        joint   = args.joint,
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
elif args.model == "rnnvie":
    model = RnnVie(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        x_emb_sz = args.emb_sz,
        r_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        attn    = args.attn,
        joint   = args.joint,
    )
elif args.model == "crnnlmca":
    model = CrnnLmCa(
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
        qc = args.qc if hasattr(args, "qc") else False,
        qcrnn = args.qcrnn if hasattr(args, "qcrnn") else False,
    )
elif args.model == "crnnlmqca":
    model = CrnnLmQca(
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
        qc = args.qc if hasattr(args, "qc") else False,
        qcrnn = args.qcrnn if hasattr(args, "qcrnn") else False,
        v2d = args.v2d if hasattr(args, "v2d") else False
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
        glove = args.glove,
        nuisance = args.nuisance,
        initu = args.initu,
        tanh = args.tanh,
        vctxt = args.vctxt if hasattr(args, "vctxt") else False,
        wcv = args.wcv if hasattr(args, "wcv") else False,
        etvctxt = args.etvctxt if hasattr(args, "etvctxt") else False,
        bil = args.bil if hasattr(args, "bil") else False,
    )
elif args.model == "crnnlmecqca":
    model = CrnnLmEcqca(
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
        glove = args.glove,
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
elif args.model == "crnnlmesqca":
    model = CrnnLmEsqca(
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
        glove = args.glove,
        nuisance = args.nuisance,
        initu = args.initu,
        tanh = args.tanh,
        vctxt = args.vctxt,
        wcv = args.wcv,
        etvctxt = args.etvctxt,
        bil = args.bil,
    )
elif args.model == "crnnlmcqca":
    model = CrnnLmCqca(
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
        qc = args.qc if hasattr(args, "qc") else False,
        qcrnn = args.qcrnn if hasattr(args, "qcrnn") else False,
        v2d = args.v2d if hasattr(args, "v2d") else False,
        nuisance = args.nuisance if hasattr(args, "nuisance") else False,
    )

model.to(device)
print(model)

if args.klannealsteps > 0:
    model.kl_anneal_steps = args.klannealsteps

model.evalp = args.evalp

model.K = 2
model.Ke = 0
model.mode = args.mode
model.q = args.q

model.tao = 1
model.qtao = 1

#dbg
model.ENT = ENT.vocab
model.TYPE = TYPE.vocab
model.VALUE = VALUE.vocab
model.TEXT = TEXT.vocab

if args.eval_only:
    model.load_state_dict(model_state, strict=False)
    #train_loss, ntok, tp, fp, fn, tm, tg = model.validate_ie(train_iter, T=args.T, E=args.E, R=args.R)
    #train_loss, ntok = model.validate_ie(train_iter, T=args.T, E=args.E, R=args.R)
    #print(f"train loss: {train_loss / ntok}")
    #valid_loss, ntok, tp, fp, fn, tm, tg = model.validate_ie(valid_iter, T=args.T, E=args.E, R=args.R)
    valid_loss, ntok = model.validate_ie(valid_iter, T=args.T, E=args.E, R=args.R)
    print(f"valid loss: {valid_loss / ntok}")
    sys.exit(0)

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
schedule = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.pat, factor=args.lrd, threshold=1e-3)
#batch = next(iter(train_iter))
# TODO: try truncating sequences early on?

best_val = float("inf")
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    train_fn = model.train_ie_epoch if not args.vie else model.train_vie_epoch
    validate_fn = model.validate_ie if not args.vie else model.validate_vie

    # Train
    #train_loss, tntok, tp, fp, fn, tm, tg = train_fn(
    train_loss, tntok = train_fn(
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
    #valid_loss, ntok, tp, fp, fn, tm, tg = validate_fn(valid_iter, T=args.T, E=args.E)
    valid_loss, ntok = validate_fn(valid_iter, T=args.T, E=args.E)
    print(f"Epoch {e} train loss: {train_loss / tntok} valid loss: {valid_loss / ntok}")
    schedule.step(valid_loss / ntok)

    if args.save and valid_loss < best_val:
        best_val = valid_loss
        savestring = f"{modelname}/{modelname}-e{e}.pt"
        torch.save({"model": model.state_dict(), "args": args}, savestring)
