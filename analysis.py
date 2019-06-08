import argparse
import random
import json
import sys

from copy import deepcopy

import torch
import torch.optim as optim

from namedtensor import ntorch, NamedTensor

import data
from data import RotoExample

from models.crnnlm import CrnnLm
from models.crnnmlmc import CrnnMlmC
from models.crnnmlmc_old import CrnnMlmC as Old
from models.lvm import Lvm
from models.lvm_old import Lvm as LvmOld

from models.crnnlma import CrnnLmA
from models.crnnlmsa import CrnnLmSa
from models.crnnlmca import CrnnLmCa
from models.crnnlmb import CrnnLmB
from models.crnnlmqca import CrnnLmQca
from models.crnnlmeqca import CrnnLmEqca
from models.crnnlmcqca import CrnnLmCqca

from models.rnnie import RnnIe

torch.backends.cudnn.enabled = False
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.deterministic = True

# debug
parser = argparse.ArgumentParser()
parser.add_argument("--devid", default=0, type=int)
parser.add_argument("--bsz", default=6, type=int)
parser.add_argument("--split-attn", type=str,
    #default="v3-d-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K0-scFalse-qqay/v3-d-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K0-scFalse-qqay-e22.pt")
    #default="ie-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-ks0-scFalse-qqay-nvFalse-jcFalse-tfFalse-qcFalse-qrFalse/ie-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-ks0-scFalse-qqay-nvFalse-jcFalse-tfFalse-qcFalse-qrFalse-e22.pt")
    #default="iea-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-ks0-scFalse-qqay-nvFalse-jcFalse-tfFalse-qcFalse-qrFalse/iea-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-ks0-scFalse-qqay-nvFalse-jcFalse-tfFalse-qcFalse-qrFalse-e22.pt")
    default="iea-dbg-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse/iea-dbg-crnnlma-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-e0.pt")
parser.add_argument("--ca", type=str,
    # K8
    # this one is good but check out qc first
    #default="ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue/ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-e26.pt")
    #default="ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue/ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue-e29.pt")
   default="iea-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue/iea-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue-e24.pt")
parser.add_argument("--sa", type=str,
   default="dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dTrue-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse/dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dTrue-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse-e3-vloss2.99.pt")
parser.add_argument("--canov", type=str,
    #default="ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvTrue-jcTrue/ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvTrue-jcTrue-e27.pt")
    #default="ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvTrue-jcTrue-tfFalse-qcTrue-qrTrue/ie-d-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvTrue-jcTrue-tfFalse-qcTrue-qrTrue-e26.pt")
    # for now
    default="iea-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue/iea-crnnlmca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks10000-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcTrue-qrTrue-e13.pt")
parser.add_argument("--qca", type=str,
    default="fuck3-eqca-crnnlmeqca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K64-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse/fuck3-eqca-crnnlmeqca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K64-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-e31-vloss691212.8107910156.pt")
parser.add_argument("--cqca", type=str,
    default="gumbel-rotowire-crnnlmcqca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-Ke16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nTrue-iuFalse-igFalse-tFalse-vcFalse-wcvFalse-bFalse/gumbel-rotowire-crnnlmcqca-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K1-Ke16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nTrue-iuFalse-igFalse-tFalse-vcFalse-wcvFalse-bFalse-e31-vloss2.61.pt")
parser.add_argument("--b", type=str,
    default="ie-d-crnnlmb-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks0-scFalse-qqay-nvFalse-jcFalse-sb2.0-tfFalse/ie-d-crnnlmb-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K8-ks0-scFalse-qqay-nvFalse-jcFalse-sb2.0-tfFalse-e17.pt",
)
parser.add_argument("--rie", type=str,
    default="ie-rnnie-sFalse-rs256-b16-lr0.001-lrd0.1-dp0.0-aemb/ie-rnnie-sFalse-rs256-b16-lr0.001-lrd0.1-dp0.0-aemb-e8.pt",
    # need to update this as well
)
parser.add_argument("--numericvalues", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--save", action="store_true")
test_args = parser.parse_args()

bsz = test_args.bsz
device = torch.device(f"cuda:{test_args.devid}" if test_args.devid >= 0 else "cpu")

#split_thing = torch.load(test_args.split_attn, map_location="cpu")
#split_model_state, split_args = split_thing["model"], split_thing["args"]
#args = split_args

#ca_thing = torch.load(test_args.ca, map_location="cpu")
#ca_model_state, ca_args = ca_thing["model"], ca_thing["args"]
sa_thing = torch.load(
    #"dbg-soft-crnnlmsa-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K64-Ke16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nFalse-iuFalse-tFalse/dbg-soft-crnnlmsa-elbo-es256-rs256-b6-lr0.001-lrd0.1-dp0.0-twTrue-ifFalse-saFalse-K64-Ke16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nFalse-iuFalse-tFalse-e31-vloss628116.1181640625.pt",
    # crnnlmsa soft content soft value
    #"dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse/dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dFalse-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse-e14-vloss2.57.pt",
    # crnnlmsa hard content soft value
    #"dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dTrue-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse/dbgs-crnnlmsa-es256-rs256-b6-lr0.001-dp0.0-twTrue-K64-Ke16-Kl16-ks0-scFalse-qqay-nvFalse-jcTrue-tfFalse-qcFalse-qrFalse-t1-v2dTrue-vyFalse-gFalse-nFalse-tFalse-bFalse-mFalse-cs0-cw0-pe0-uFalse-bvFalse-e3-vloss2.99.pt",
    test_args.sa,
    map_location="cpu")
sa_model_state, sa_args = sa_thing["model"], sa_thing["args"]

# dbg for now
#canov_thing = torch.load(test_args.canov, map_location="cpu")
#canov_model_state, canov_args = canov_thing["model"], canov_thing["args"]

#rie_thing = torch.load(test_args.rie, map_location="cpu")
#rie_model_state, rie_args = rie_thing["model"], rie_thing["args"]

qca_thing = torch.load(test_args.qca, map_location="cpu")
qca_model_state, qca_args = qca_thing["model"], qca_thing["args"]

#cqca_thing = torch.load(test_args.cqca, map_location="cpu")
#cqca_model_state, cqca_args = cqca_thing["model"], cqca_thing["args"]

args = qca_args

E = 64
T = 32
maxlen = -1

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Data
ENT, TYPE, VALUE, VALUE_TEXT, TEXT = data.make_fields(maxlen)
train, valid, test = data.RotoDataset.splits(
    ENT, TYPE, VALUE, VALUE_TEXT, TEXT, path=args.filepath,
    numericvalues = test_args.numericvalues,
)

data.build_vocab(ENT, TYPE, VALUE, TEXT, train)
# is this enough?
TEXT.vocab.extend(VALUE.vocab)
VALUE_TEXT.vocab = TEXT.vocab

train_iter, valid_iter, test_iter = data.RotowireIterator.splits(
    (train, valid, test),
    batch_size = args.bsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)

# Model
"""
soft_model = CrnnLm(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = soft_args.emb_sz,
    x_emb_sz = soft_args.emb_sz,
    rnn_sz = soft_args.rnn_sz,
    nlayers = soft_args.nlayers,
    dropout = soft_args.dp,
    tieweights = soft_args.tieweights,
    inputfeed = soft_args.inputfeed,
)
split_model = CrnnLmA(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = split_args.emb_sz,
    x_emb_sz = split_args.emb_sz,
    rnn_sz = split_args.rnn_sz,
    nlayers = split_args.nlayers,
    dropout = split_args.dp,
    tieweights = split_args.tieweights,
    inputfeed = split_args.inputfeed,
)
ca_model = CrnnLmCa(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = ca_args.emb_sz,
    x_emb_sz = ca_args.emb_sz,
    rnn_sz = ca_args.rnn_sz,
    nlayers = ca_args.nlayers,
    dropout = ca_args.dp,
    tieweights = ca_args.tieweights,
    inputfeed = ca_args.inputfeed,
    qc = ca_args.qc,
    qcrnn = ca_args.qcrnn,
)
canov_model = CrnnLmCa(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = canov_args.emb_sz,
    x_emb_sz = canov_args.emb_sz,
    rnn_sz = canov_args.rnn_sz,
    nlayers = canov_args.nlayers,
    dropout = canov_args.dp,
    tieweights = canov_args.tieweights,
    inputfeed = canov_args.inputfeed,
    noattnvalues = False,
    qc = canov_args.qc if hasattr(canov_args, "qc") else False,
    qcrnn = canov_args.qcrnn if hasattr(canov_args, "qcrnn") else False,
)
b_model = CrnnLmB(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = b_args.emb_sz,
    x_emb_sz = b_args.emb_sz,
    rnn_sz = b_args.rnn_sz,
    nlayers = b_args.nlayers,
    dropout = b_args.dp,
    tieweights = b_args.tieweights,
    inputfeed = b_args.inputfeed,
    noattnvalues = b_args.noattnvalues,
    sigbias = b_args.sigbias,
)
rie_model = RnnIe(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    x_emb_sz = rie_args.emb_sz,
    rnn_sz = rie_args.rnn_sz,
    nlayers = rie_args.nlayers,
    dropout = rie_args.dp,
    joint   = rie_args.joint,
)
"""
sa_model = CrnnLmSa(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = sa_args.emb_sz,
    x_emb_sz = sa_args.emb_sz,
    rnn_sz = sa_args.rnn_sz,
    nlayers = sa_args.nlayers,
    dropout = sa_args.dp,
    tieweights = sa_args.tieweights,
    inputfeed = sa_args.inputfeed,
    noattnvalues = sa_args.noattnvalues,
    initu = sa_args.initu,
    initg = sa_args.initg,
    v2d = sa_args.v2d,
    initvy = sa_args.initvy,
    mlp = sa_args.mlp,
    hardc = sa_args.hardc if hasattr(sa_args, "hardc") else False,
    fixedc = sa_args.fixedc if hasattr(sa_args, "fixedc") else False,
    maskedc = sa_args.maskedc if hasattr(sa_args, "maskedc") else False,
)
"""
qca_model = CrnnLmEqca(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = qca_args.emb_sz,
    x_emb_sz = qca_args.emb_sz,
    rnn_sz = qca_args.rnn_sz,
    nlayers = qca_args.nlayers,
    dropout = qca_args.dp,
    tieweights = qca_args.tieweights,
    inputfeed = qca_args.inputfeed,
    noattnvalues = False,
    qc = qca_args.qc if hasattr(qca_args, "qc") else False,
    qcrnn = qca_args.qcrnn if hasattr(qca_args, "qcrnn") else False,
    qconly = qca_args.qconly if hasattr(qca_args, "qconly") else False,
    jointcopy = True,
    v2d = qca_args.v2d if hasattr(qca_args, "v2d") else False,
)
"""
qca_model = CrnnLmEqca(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = qca_args.emb_sz,
    x_emb_sz = qca_args.emb_sz,
    rnn_sz = qca_args.rnn_sz,
    nlayers = qca_args.nlayers,
    dropout = qca_args.dp,
    tieweights = qca_args.tieweights,
    inputfeed = qca_args.inputfeed,
    noattnvalues = qca_args.noattnvalues,
    jointcopy = qca_args.jointcopy,
    qc = qca_args.qc,
    qcrnn = qca_args.qcrnn,
    temp = qca_args.temp,
    qconly = qca_args.qconly,
    v2d = qca_args.v2d,
    initvy = qca_args.initvy,
    glove = qca_args.glove if hasattr(qca_args, "glove") else False,
    nuisance = qca_args.nuisance if hasattr(qca_args, "nuisance") else False,
    initu = qca_args.initu if hasattr(qca_args, "initu") else False,
    tanh = qca_args.tanh if hasattr(qca_args, "tanh") else True,
    vctxt = qca_args.vctxt if hasattr(qca_args, "vctxt") else False,
    wcv = qca_args.wcv if hasattr(qca_args, "wcv") else False,
    bil = qca_args.bil if hasattr(qca_args, "bil") else False,
)
"""
cqca_model = CrnnLmCqca(
    Ve = ENT.vocab,
    Vt = TYPE.vocab,
    Vv = VALUE.vocab,
    Vx = TEXT.vocab,
    r_emb_sz = qca_args.emb_sz,
    x_emb_sz = qca_args.emb_sz,
    rnn_sz = qca_args.rnn_sz,
    nlayers = qca_args.nlayers,
    dropout = qca_args.dp,
    tieweights = qca_args.tieweights,
    inputfeed = qca_args.inputfeed,
    noattnvalues = False,
    qc = qca_args.qc if hasattr(qca_args, "qc") else False,
    qcrnn = qca_args.qcrnn if hasattr(qca_args, "qcrnn") else False,
    v2d = args.v2d if hasattr(args, "v2d") else False,
    nuisance = args.nuisance if hasattr(args, "nuisance") else False,
)
"""
#soft_model.to(device)
#split_model.to(device)
#ca_model.to(device)
#canov_model.to(device)
#b_model.to(device)
#rie_model.to(device)
sa_model.to(device)
qca_model.to(device)
#cqca_model.to(device)
print(qca_model)

#ca_model.K = 2
#canov_model.K = 2
qca_model.K = 16
qca_model.Ke = 8
qca_model.Kl = 0
#cqca_model.K = 1

#cqca_model.tao = cqca_args.tao
#cqca_model.qtao = cqca_args.qtao

#model.load_state_dict(model_state)
#old_model.load_state_dict(model_state)
#soft_model.load_state_dict(soft_model_state)
#b_model.load_state_dict(b_model_state)
#split_model.load_state_dict(split_model_state)
#ca_model.load_state_dict(ca_model_state)
#canov_model.load_state_dict(canov_model_state)
#rie_model.load_state_dict(rie_model_state)
sa_model.load_state_dict(sa_model_state, strict=False)
qca_model.load_state_dict(qca_model_state, strict=False)
#cqca_model.load_state_dict(cqca_model_state)


def check_sa(
    x, y, r, ue, ut, v2d, vt2d, d,
    sa_py, sa_pe, sa_pt, sa_pc,
):
    def top5(t, query, p):
        if query == "e":
            p5, e5 = p.get("time", t).topk("e", 5)
            print(" | ".join(
                f"{qca_model.Ve.itos[ue.get('els', x).item()]} ({p:.2f})"
                for p,x in zip(p5.tolist(), e5.tolist())
            ))
        elif query == "t":
            p5, t5 = p.get("time", t).topk("t", 5)
            print(" | ".join(
                f"{qca_model.Vt.itos[ut.get('els', x).item()]} ({p:.2f})"
                for p,x in zip(p5.tolist(), t5.tolist())
            ))
        else:
            raise NotImplementedError

    # check if alphabetical reps are close to numerical
    words = ["2", "4", "6", "8"]
    for word in words:
        input = (
            sa_model.lutx.weight[sa_model.Vx.stoi[word]]
            if not sa_model.v2d
            else sa_model.lutv.weight[sa_model.Vv.stoi[word]]
        )
        if sa_model.mlp:
            probs, idx = (sa_model.lutx.weight @ sa_model.Wvy1(sa_model.Wvy0(NamedTensor(
                input, names=("ctxt",)
            )).tanh()).tanh().values).softmax(0).topk(5)
        else:
            probs, idx = (sa_model.lutx.weight @ input).softmax(0).topk(5)
        print(f"{word} probs "+ " || ".join(f"{sa_model.Vx.itos[x]}: {p:.2f}" for p,x in zip(probs.tolist(), idx.tolist())))


    # check if alphabetical and numerical words are aligned correctly and high prob under model
    ytext = [TEXT.vocab.itos[y] for y in y.tolist()]

    t = 185
    print(ytext[t-30:t+10])
    print(ytext[t])
    print()
    print("blake griffin assists alphabetical")
    print(f"py: {sa_py.get('time', t).item()}")
    print(f"pc: {sa_pc.get('time', t).get('copy', 1).item()}")
    top5(t, "e", sa_pe)
    top5(t, "t", sa_pt)

    t = 182
    print()
    print("blake griffin rebounds numerical")
    print(f"py: {sa_py.get('time', t).item()}")
    print(f"pc: {sa_pc.get('time', t).get('copy', 1).item()}")
    top5(t, "e", sa_pe)
    top5(t, "t", sa_pt)

    # check top pc
    print()
    print("Checking top copy prob words")
    probs, time = sa_pc.get("copy", 1).topk("time", 10)
    print(" || ".join(f"{ytext[t]}: {p:.2f}" for p,t in zip(probs.tolist(), time.tolist())))

    print()
    print("Checking for garbage alignments => the")
    [(top5(i, "e", sa_pe), top5(i, "t", sa_pt)) for i,x in enumerate(ytext[:50]) if x == "the"]
    import pdb; pdb.set_trace()

def examine(
    x, y, r, ue, ut, v2d, vt2d, d,
    #split_py, split_pe, split_pt,
    #ca_pc, ca_pe, ca_pt, ca_qc, ca_qe, ca_qt,
    #ca_py_a, ca_py_c0, ca_py_ac1,
    #canov_pc, canov_pe, canov_pt, canov_qc, canov_qe, canov_qt,
    #canov_py_a, canov_py_c0, canov_py_ac1,
    #rie_pe, rie_pt,
    sa_py, sa_pe, sa_pt, sa_pc,
    qca_pc, qca_pe, qca_pt, qca_qc, qca_qe, qca_qt,
    qca_py_c0, qca_py_ac1,
    #cqca_pc, cqca_pe, cqca_pt, cqca_qc, cqca_qe, cqca_qt,
    #cqca_py_c0, cqca_py_ac1,
):
    # everything should be unbatched
    xtext = [TEXT.vocab.itos[x] for x in x.tolist()]
    ytext = [TEXT.vocab.itos[y] for y in y.tolist()]
    r = list(zip(*[x.tolist() for x in r]))

    """
    Hsplitpe = torch.distributions.Categorical(split_pe.values).entropy()
    split_Hpe = torch.distributions.Categorical(split_pe.values).entropy()
    split_Hpt = torch.distributions.Categorical(split_pt.values).entropy()
    ca_Hpe = torch.distributions.Categorical(ca_pe.values).entropy()
    ca_Hpt = torch.distributions.Categorical(ca_pt.values).entropy()
    ca_Hqe = torch.distributions.Categorical(ca_qe.values).entropy()
    ca_Hqt = torch.distributions.Categorical(ca_qt.values).entropy()
    canov_Hpe = torch.distributions.Categorical(canov_pe.values).entropy()
    canov_Hpt = torch.distributions.Categorical(canov_pt.values).entropy()
    canov_Hqe = torch.distributions.Categorical(canov_qe.values).entropy()
    canov_Hqt = torch.distributions.Categorical(canov_qt.values).entropy()
    """
    qca_Hpe = torch.distributions.Categorical(qca_pe.values).entropy()
    qca_Hpt = torch.distributions.Categorical(qca_pt.values).entropy()
    qca_Hqe = torch.distributions.Categorical(qca_qe.values).entropy()
    qca_Hqt = torch.distributions.Categorical(qca_qt.values).entropy()


    # words where noncontent is worse than content
    w0 = [ytext[i] for i, x in enumerate((qca_py_c0.get("k", 0) < qca_py_ac1.get("k", 0)).tolist()) if x]
    w1 = [ytext[i] for i, x in enumerate((qca_py_c0.get("k", 0) < qca_py_ac1.get("k", 1)).tolist()) if x]
    # plus indices
    wi0 = [(i,ytext[i]) for i, x in enumerate((qca_py_c0.get("k", 0) < qca_py_ac1.get("k", 0)).tolist()) if x]
    wi1 = [(i,ytext[i]) for i, x in enumerate((qca_py_c0.get("k", 0) < qca_py_ac1.get("k", 1)).tolist()) if x]
    # Where is soft more sure than hard prior?
    #print([(x[0], ytext[x[0]]) for x in (split_Hpt < ca_Hqt).nonzero().tolist()])

    def snippets(binary):
        return [
            f"{ytext[x]}: {' '.join(ytext[x-3:x+3])}"
            for x in binary.nonzero().squeeze().tolist()
        ]

    #canov_qc_snippets = snippets(canov_qc.get("copy", 1).values > 0.6)

    def top5(t, query, p):
        if query == "e":
            p5, e5 = p.get("time", t).topk("e", 5)
            print(" | ".join(
                f"{qca_model.Ve.itos[ue.get('els', x).item()]} ({p:.2f})"
                for p,x in zip(p5.tolist(), e5.tolist())
            ))
        elif query == "t":
            p5, t5 = p.get("time", t).topk("t", 5)
            print(" | ".join(
                f"{qca_model.Vt.itos[ut.get('els', x).item()]} ({p:.2f})"
                for p,x in zip(p5.tolist(), t5.tolist())
            ))
        else:
            raise NotImplementedError

    def qcat5(t):
        top5(t, "e", qca_qe)
        top5(t, "t", qca_qt)

    def top5ie(t, query, p):
        if query == "e":
            p5, e5 = p.get("time", t).topk("e", 5)
            print(" | ".join(
                f"{rie_model.Ve.itos[x]} ({p:.2f})"
                for p,x in zip(p5.tolist(), e5.tolist())
            ))
        elif query == "t":
            p5, t5 = p.get("time", t).topk("t", 5)
            print(" | ".join(
                f"{rie_model.Vt.itos[x]} ({p:.2f})"
                for p,x in zip(p5.tolist(), t5.tolist())
            ))
        else:
            raise NotImplementedError

    def lookup(e, t, v2d):
        ex = ue.tolist().index(qca_model.Ve.stoi[e])
        tx = ut.tolist().index(qca_model.Vt.stoi[t])
        print(qca_model.Vx.itos[v2d[{"e": ex, "t": tx}].item()])

    # calculate accuracy
    for t, (es, ts) in d.items():
        #t = t + 1
        """
        pe_split = split_pe.get("time", t)
        pt_split = split_pt.get("time", t)
        _, e_split_max = pe_split.max("e")
        _, t_split_max = pt_split.max("t")
        e_split_preds = ue.get("els", e_split_max.item())
        t_split_preds = ut.get("els", t_split_max.item())
        split_correct = (es.eq(e_split_preds) * ts.eq(t_split_preds)).any().float().item()
        e_split_correct = es.eq(e_split_preds).any().float().item()
        t_split_correct = ts.eq(t_split_preds).any().float().item()

        qe_canov = canov_qe.get("time", t)
        qt_canov = canov_qt.get("time", t)
        _, e_canov_max = qe_canov.max("e")
        _, t_canov_max = qt_canov.max("t")
        e_canov_preds = ue.get("els", e_canov_max.item())
        t_canov_preds = ut.get("els", t_canov_max.item())
        canov_correct = (es.eq(e_canov_preds) * ts.eq(t_canov_preds)).any().float().item()
        e_canov_correct = es.eq(e_canov_preds).any().float().item()
        t_canov_correct = ts.eq(t_canov_preds).any().float().item()

        qe_ca = ca_qe.get("time", t)
        qt_ca = ca_qt.get("time", t)
        _, e_ca_max = qe_ca.max("e")
        _, t_ca_max = qt_ca.max("t")
        e_ca_preds = ue.get("els", e_ca_max.item())
        t_ca_preds = ut.get("els", t_ca_max.item())
        ca_correct = (es.eq(e_ca_preds) * ts.eq(t_ca_preds)).any().float().item()
        e_ca_correct = es.eq(e_ca_preds).any().float().item()
        t_ca_correct = ts.eq(t_ca_preds).any().float().item()

        _, e_rie_max = rie_pe.get("time", t).max("e")
        _, t_rie_max = rie_pt.get("time", t).max("t")
        rie_correct = (es.eq(e_rie_max) * ts.eq(t_rie_max)).any().float().item()
        """

        ue = ue.cpu()
        ut = ut.cpu()

        qe_qca = qca_qe.get("time", t)
        qt_qca = qca_qt.get("time", t)
        _, e_qca_max = qe_qca.max("e")
        _, t_qca_max = qt_qca.max("t")
        e_qca_preds = ue.get("els", e_qca_max.item())
        t_qca_preds = ut.get("els", t_qca_max.item())
        qca_correct = (es.eq(e_qca_preds) * ts.eq(t_qca_preds)).any().float().item()
        e_qca_correct = es.eq(e_qca_preds).any().float().item()
        t_qca_correct = ts.eq(t_qca_preds).any().float().item()

        """
        qe_cqca = cqca_qe.get("time", t)
        qt_cqca = cqca_qt.get("time", t)
        _, e_cqca_max = qe_cqca.max("e")
        _, t_cqca_max = qt_cqca.max("t")
        e_cqca_preds = ue.get("els", e_cqca_max.item())
        t_cqca_preds = ut.get("els", t_cqca_max.item())
        cqca_correct = (es.eq(e_cqca_preds) * ts.eq(t_cqca_preds)).any().float().item()
        e_cqca_correct = es.eq(e_cqca_preds).any().float().item()
        t_cqca_correct = ts.eq(t_cqca_preds).any().float().item()
        """

        #if qca_correct < 1 or split_correct < 1 or canov_correct < 1 or ca_correct < 1 or rie_correct < 1:
        #if qca_correct < 1 and  ca_correct == 1:
        if True:
            print("=========")
            print(f"time: {t} | {ytext[t]}")
            print(" ".join(ytext[max(0,t-15):t+15]))
            print(
                #f"pa: {split_correct} | qca: {qca_correct} | cqca: {cqca_correct} | ca: {ca_correct} | canov: {canov_correct} | rie: {rie_correct}"
                f"qca: {qca_correct}"
            )
            print()

            """
            print("split top5")
            top5(t, "e", split_pe)
            top5(t, "t", split_pt)
            print()


            print("ca top5")
            top5(t, "e", ca_qe)
            top5(t, "t", ca_qt)
            print()
            """

            print("qca top5")
            print("q")
            top5(t, "e", qca_qe)
            top5(t, "t", qca_qt)
            print("p")
            top5(t, "e", qca_pe)
            top5(t, "t", qca_pt)
            print()

            """
            print("cqca top5")
            print("q")
            top5(t, "e", cqca_qe)
            top5(t, "t", cqca_qt)
            print("p")
            top5(t, "e", cqca_pe)
            top5(t, "t", cqca_pt)
            print()
            """

            """
            print("canov top5")
            top5(t, "e", canov_qe)
            top5(t, "t", canov_qt)
            print()

            print("rie top5")
            top5ie(t, "e", rie_pe)
            top5ie(t, "t", rie_pt)
            print()
            """

            #import pdb; pdb.set_trace()

    ### hard attention fails here, but p(y | a=cell_idx) is the best...?
    #t = 22
    #ent = ENT.vocab.stoi["wilson chandler"]
    #type = TYPE.vocab.stoi["pts"]
    def find(ent, type):
        return [(t, r) for t, r in enumerate(r) if r[0] == ent and r[1] == type]
    #cell_idx = results[0][0]
    """
    pa_t = pa.get("time", t)
    pa_y_t = pa_y.get("time", t)
    soft_pa_t = soft_pa.get("time", t)
    print(pa_t.get("els", cell_idx))
    print(pa_y_t.get("els", cell_idx))
    print(soft_pa_t.get("els", cell_idx))
    print(soft_pa.get("time", t-1).topk("els", 10))

    print(soft_pa.get("time", 150).topk("els", 10))
    """

    #Hp = torch.distributions.Categorical(split_pa.values).entropy()
    #sumwords = [" ".join(ytext[x-5:x+5]) for x,y in enumerate(Hp.tolist())]
    import pdb; pdb.set_trace()

#iterlol = iter(train_iter)
#batch = next(iterlol)
#batch = next(iterlol)

iterlol = train_iter if test_args.train else valid_iter
for batch in iterlol:

    text, x_info = batch.text
    mask = x_info.mask
    lens = x_info.lengths

    ie_x_info = deepcopy(x_info)

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

    # values text
    vt, vt_info = batch.values_text
    # DBG
    couldve_copied = (vt == y).sum().item()

    vt2d = batch.vt2d
    v2d = batch.v2d

    # should i include <eos> in ppl?
    nwords = y.ne(1).sum()
    # assert nwords == lens.sum()
    #T = y.shape["time"]
    N = y.shape["batch"]
    #if states is None:
    states = qca_model.init_state(N)

    # should i include <eos> in ppl? no, should not.
    mask = y.ne(1) #* y.ne(3)
    nwords = mask.sum()

    """
    split_model.eval()
    split_model.copied = 0
    ca_model.eval()
    ca_model.copied = 0
    """
    # SAD, for now.
    with torch.no_grad():
        # small sanity checks...?
        """

        # soft
        logits, _ = soft_model(x, states, x_info, r, r_info, ue, ue_info, ut, ut_info, v2d)
        soft_pa = soft_model.ea
        soft_nll = soft_model.loss(logits, y)
        soft_log_py = logits.log_softmax("vocab").gather("vocab", y.chop("batch", ("lol", "batch"), lol=1), "lol").get("lol", 0)
        """

        # soft
        """
        logits, _ = split_model(x, states, x_info, r, r_info, vt, ue, ue_info, ut, ut_info, v2d, vt2d)
        split_pe = split_model.ea
        split_pt = split_model.ta
        split_nll = split_model.loss(logits, y)
        split_log_py = logits.log_softmax("vocab").gather("vocab", y.chop("batch", ("lol", "batch"), lol=1), "lol").get("lol", 0)

        rvinfo, _, ca_nll, ca_kl = ca_model(
            x, states, x_info, r, r_info, vt,
            ue, ue_info, ut, ut_info, v2d,
            y, x_info,
            T = 256,
        )
        #import pdb; pdb.set_trace()
        canovrvinfo, _, canov_nll, canov_kl = canov_model(
            x, states, x_info, r, r_info, vt,
            ue, ue_info, ut, ut_info, v2d,
            y, x_info,
            T = 256,
        )

        b_py, _ = logits, _ = b_model(
            x, states, x_info, r, r_info, vt, ue, ue_info, ut, ut_info, v2d)

        nll, rie_log_pe, rie_log_pt = rie_model(text, ie_x_info, batch.ie_d)
        # log_pe is a cat distribution over all of rie_model.Ve
        """
        logits, _ = sa_model(
            x, states, x_info, r, r_info, vt,
            ue, ue_info, ut, ut_info, v2d, vt2d,
            y,
        )
        sa_pe = sa_model.ea
        sa_pt = sa_model.ta
        sa_pc = sa_model.pc
        sa_nll = sa_model.loss(logits, y)
        sa_log_py = logits.log_softmax("vocab").gather("vocab", y.chop("batch", ("lol", "batch"), lol=1), "lol").get("lol", 0)

         
        qcarvinfo, _, qca_nll, qca_kl = qca_model(
            x, states, x_info, r, r_info, vt,
            ue, ue_info, ut, ut_info, v2d, vt2d,
            y, x_info,
            T = 256,
        )

        """
        cqcarvinfo, _, cqca_nll, cqca_kl = cqca_model(
            x, states, x_info, r, r_info, vt,
            ue, ue_info, ut, ut_info, v2d, vt2d,
            y, x_info,
            T = 256,
        )
        """

        i = 0
        check_sa(
            x.get("batch", i),
            y.get("batch", i),
            [x.get("batch", i) for x in r],
            ue.get("batch", i),
            ut.get("batch", i),
            v2d.get("batch", i),
            vt2d.get("batch", i),
            batch.ie_et_d[i],
            sa_log_py.exp().get("batch", i),
            sa_pe.exp().get("batch", i),
            sa_pt.exp().get("batch", i),
            sa_pc.get("batch", i),
        )
        def ip(i):
            examine(
                x.get("batch", i),
                y.get("batch", i),
                [x.get("batch", i) for x in r],
                ue.get("batch", i),
                ut.get("batch", i),
                v2d.get("batch", i),
                vt2d.get("batch", i),
                batch.ie_et_d[i],
                #split_log_py.exp().get("batch", i),
                #split_pe.exp().get("batch", i),
                #split_pt.exp().get("batch", i),
                #rvinfo.log_pc.exp().get("batch", i),
                #rvinfo.log_pe.exp().get("batch", i),
                #rvinfo.log_pt.exp().get("batch", i),
                #rvinfo.log_qc_y.exp().get("batch", i),
                #rvinfo.log_qe_y.exp().get("batch", i),
                #rvinfo.log_qt_y.exp().get("batch", i),
                #rvinfo.log_py_a.exp().get("batch", i),
                #rvinfo.log_py_c0.exp().get("batch", i),
                #rvinfo.log_py_ac1.exp().get("batch", i),
                #canovrvinfo.log_pc.exp().get("batch", i),
                #canovrvinfo.log_pe.exp().get("batch", i),
                #canovrvinfo.log_pt.exp().get("batch", i),
                #canovrvinfo.log_qc_y.exp().get("batch", i),
                #canovrvinfo.log_qe_y.exp().get("batch", i),
                #canovrvinfo.log_qt_y.exp().get("batch", i),
                #canovrvinfo.log_py_a.exp().get("batch", i),
                #canovrvinfo.log_py_c0.exp().get("batch", i),
                ##canovrvinfo.log_py_ac1.exp().get("batch", i),
                #rie_log_pe.exp().get("batch", i),
                #rie_log_pt.exp().get("batch", i),
                sa_log_py.exp().get("batch", i),
                sa_pe.exp().get("batch", i),
                sa_pt.exp().get("batch", i),
                sa_pc.get("batch", i),
                qcarvinfo.log_pc.softmax("copy").get("batch", i),
                qcarvinfo.log_pe.exp().get("batch", i),
                qcarvinfo.log_pt.exp().get("batch", i),
                qcarvinfo.log_qc_y.softmax("copy").get("batch", i),
                qcarvinfo.log_qe_y.exp().get("batch", i),
                qcarvinfo.log_qt_y.exp().get("batch", i),
                qcarvinfo.log_py_c0.exp().get("batch", i),
                qcarvinfo.log_py_ac1.exp().get("batch", i),
                #cqcarvinfo.log_pc.exp().get("batch", i),
                #cqcarvinfo.log_pe.softmax("e").get("batch", i),
                #cqcarvinfo.log_pt.softmax("t").get("batch", i),
                #cqcarvinfo.log_qc_y.exp().get("batch", i),
                #cqcarvinfo.log_qe_y.softmax("e").get("batch", i),
                #cqcarvinfo.log_qt_y.softmax("t").get("batch", i),
                #cqcarvinfo.log_py_c0.exp().get("batch", i),
                #cqcarvinfo.log_py_ac1.exp().get("batch", i),
            )
        #ip(1)
        for i in range(6):
            print(f"Example {i}")
            ip(i)

        """
        py_diff = (log_py_ref - log_py).values.abs().max()
        pa_diff = (log_pa_ref - log_pa)[log_pa != float("-inf")].values.abs().max()
        pa_y_diff = (log_pa_y_ref - log_pa_y)[log_pa_y != float("-inf")].values.abs().max()
        print(py_diff)
        print(pa_diff)
        print(pa_y_diff)
        """
