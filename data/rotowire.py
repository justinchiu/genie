
from collections import defaultdict

import torch

import torchtext
from torchtext import data
from torchtext.data import Batch, Dataset, Example, Iterator, TabularDataset, BucketIterator

from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import io
import os
import json
import numpy as np

PAD = "<pad>"
NONE = "<none>" # just use unk for bucket e,t,v?

def make_fields(maxlen=-1):
    ENT   = NamedField(names=("els",), lower=True, include_lengths=True)
    TYPE  = NamedField(names=("els",), lower=True, include_lengths=True)
    VALUE = NamedField(names=("els",), lower=True, include_lengths=True)
    VALUE_TEXT = NamedField(
        names = ("els",),
        lower=True, include_lengths=True, init_token=None, eos_token=None, is_target=True)
    TEXT  = NamedField(
        names = ("time",),
        lower=True, include_lengths=True,
        init_token="<bos>", eos_token="<eos>", is_target=True,
        fix_length = maxlen if maxlen > 0 else None,
    )
    return ENT, TYPE, VALUE, VALUE_TEXT, TEXT

def build_vocab(a,b,c,text, data, min_freq=1):
    a.build_vocab(data, specials=[PAD, NONE])
    b.build_vocab(data, specials=[PAD, NONE])
    c.build_vocab(data, specials=[PAD, NONE])
    text.build_vocab(data, specials=[PAD, NONE], min_freq=min_freq)


def nested_items(name, x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from nested_items(f"{name}_{k}", v)
    else:
        yield (name, x)


class RotoExample(Example):
    @classmethod
    def fromJson(
        cls, data, ent_field, type_field, value_field, text_field,
        sentences=False,
        numericvalues=False,
        supex=-1,
    ):
        exs = []
        jsonlist = json.load(data)

        for x_idx, x in enumerate(jsonlist):
            entities = []
            types = []
            values = []

            # Need to flatten all the tables into aligned lists of
            # entities, types, and values.
            # team stuff
            home_name = x["home_name"]
            vis_name = x["vis_name"]

            def add(entity, type, value):
                entities.append(entity)
                types.append(type)
                values.append(value)

            # didn't have this in...
            add(home_name, "home-name", home_name)
            add(vis_name, "vis-name", vis_name)
            add(home_name, "team-name", home_name)
            add(vis_name, "team-name", vis_name)

            # flat team stats
            add(home_name, "home-city", x["home_city"])
            add(vis_name, "vis-city", x["vis_city"])
            add(home_name, "team-city", x["home_city"])
            add(vis_name, "team-city", x["vis_city"])
            add("day", "day", x["day"])

            # team lines
            for k, v in x["home_line"].items():
                add(home_name, k, v)
            for k, v in x["vis_line"].items():
                add(vis_name, k, v)

            # flatten box_score: {key: {ID: value}}
            box_score = x["box_score"]
            id2name = box_score["PLAYER_NAME"]

            for k, d in box_score.items():
                if k != "TEAM_CITY":
                    for id, v in d.items():
                        add(id2name[id], k, v)

            ents = ent_field.preprocess(entities)
            types = type_field.preprocess(types)
            values = value_field.preprocess(values)
            values_text = text_field.preprocess(values)
            text = text_field.preprocess(x["summary"])

            uents = list(set(ents)) + [NONE]
            utypes = list(set(types)) + [NONE]

            # ie stuff
            def get_ie(text):
                ie_etv = [
                    [
                        (e,t,v)
                        for e,t,v in zip(ents, types, values)
                        if w == v and (not numericvalues or v.isnumeric())
                    ]
                    for i, w in enumerate(text)
                ]
                # add (NONE, NONE) if empty list
                ie_etv = [x if x else [(NONE, NONE, NONE)] for x in ie_etv]
                ie_et_d = {
                    i: [
                        (e,t)
                        for e,t,v in zip(ents, types, values)
                        if w == v and (not numericvalues or v.isnumeric())
                    ]
                    for i, w in enumerate(text)
                }
                ie_et_d = {
                    k: v
                    for k,v in ie_et_d.items()
                    if v
                }
                return ie_etv, ie_et_d

            # flattened sentences
            if sentences:
                def sens(ws):
                    idx = 0
                    next = 0
                    N = len(ws)
                    while next < N:
                        if ws[next] in [".", "?", "!"]:
                            sen = ws[idx:next+1]
                            next += 1
                            idx = next
                            yield sen
                        next += 1
                    if next > idx + 1:
                        yield ws[idx:next+1]

                for sen in sens(text):
                    ex = cls()
                    # copypasta
                    setattr(ex, "entities",    ents)
                    setattr(ex, "types",       types)
                    setattr(ex, "values",      values)
                    setattr(ex, "values_text", values_text)
                    setattr(ex, "text",        sen)
                    setattr(ex, "uentities",   uents)
                    setattr(ex, "utypes",      utypes)
                    # ie
                    ie_etv, ie_et_d = get_ie(sen)
                    setattr(ex, "ie_etv",      ie_etv)
                    setattr(ex, "ie_et_d",     ie_et_d)
                    setattr(ex, "idx",         x_idx)
                    exs.append(ex)
            else:
                ex = cls()
                # entities, types, values, summary
                setattr(ex, "entities",    ents)
                setattr(ex, "types",       types)
                setattr(ex, "values",      values)
                setattr(ex, "values_text", values_text)
                setattr(ex, "text",        text)
                setattr(ex, "uentities",   uents)
                setattr(ex, "utypes",      utypes)
                # ie
                ie_etv, ie_et_d = get_ie(text)
                setattr(ex, "ie_etv",      ie_etv)
                setattr(ex, "ie_et_d",     ie_et_d)
                setattr(ex, "idx",         x_idx)
                exs.append(ex)

        if supex >= 0:
            N = len(exs)
            perm = np.random.permutation(N)
            supexs = [exs[idx] for idx in perm[:supex]]
            return exs, supexs
        return exs


class RotoDataset(Dataset):

    @staticmethod
    def make_fields(
        entity_field, type_field, value_field,
        value_text_field, text_field,
    ):
        return [
            ("entities",    entity_field),
            ("types",       type_field),
            ("values",      value_field),
            ("values_text", value_text_field),
            ("text",        text_field),
            ("uentities",   entity_field),
            ("utypes",      type_field),
        ]


    def __init__(
        self, path,
        entity_field, type_field, value_field, value_text_field, text_field,
        reset=False,
        sentences=False, numericvalues=False, supex=-1,
        **kwargs
    ):

        # Sort by length of the text
        self.sort_key = lambda x: len(x.text)

        fields = self.make_fields(entity_field, type_field, value_field, value_text_field, text_field)

        print(path)
        suffix = ".pth"
        if sentences:
            suffix = ".sens" + suffix
        if numericvalues:
            suffix = ".numeric" + suffix
        if supex >= 0:
            suffix = suffix + f".{supex}"
        save_path = path + suffix
        if reset or not os.path.exists(save_path):
            with io.open(os.path.expanduser(path), encoding="utf8") as f:
                examples = RotoExample.fromJson(
                    f,
                    entity_field, type_field, value_field, text_field,
                    sentences = sentences,
                    numericvalues = numericvalues,
                    supex = supex,
                )
            torch.save(examples, save_path)
        else:
            examples = torch.load(save_path)

        if supex >= 0:
            examples, sup_examples = examples

        # unused
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(RotoDataset, self).__init__(examples, fields, **kwargs)
        self.sup_examples = sup_examples if supex >= 0 else self.examples
        self.unsup_examples = self.examples
        # Switch between sup and unsup by setting self.examples = self.*_examples
        # and create two iterators, one for each.


    # extra kwargs
    # reset = False : for resetting saved examples
    # sentences = False : for using only sentences
    # numericvalues = False : only numeric values for IE? Copy can just ignore
    @classmethod
    def splits(
        cls,
        entity_field, type_field, value_field, value_text_field, text_field,
        path = None,
        root='rotowire',
        train='train.json', validation='valid.json', test='test.json',
        supex = -1,
        reset = False,
        **kwargs
    ):
        Dtrain, Dvalid, Dtest = super(RotoDataset, cls).splits(
            path = path,
            root = root,
            train = train,
            validation = validation,
            test = test,
            entity_field = entity_field,
            type_field = type_field,
            value_field = value_field,
            value_text_field = value_text_field,
            text_field = text_field,
            supex = supex,
            reset = reset,
            **kwargs
        )
        if supex >= 0:
            Dtrain_sup, Dvalid_sup, Dtest_sup = super(RotoDataset, cls).splits(
                path = path,
                root = root,
                train = train,
                validation = validation,
                test = test,
                entity_field = entity_field,
                type_field = type_field,
                value_field = value_field,
                value_text_field = value_text_field,
                text_field = text_field,
                supex = supex,
                reset = False,
                **kwargs
            )
            Dtrain_sup.examples = Dtrain_sup.sup_examples
            Dvalid_sup.examples = Dvalid_sup.sup_examples
            Dtest_sup.examples = Dtest_sup.sup_examples
            return Dtrain, Dvalid, Dtest, Dtrain_sup, Dvalid_sup, Dtest_sup

        return Dtrain, Dvalid, Dtest


    @classmethod
    def iters(
        cls,
        batch_size=32, device=0,
        root=".data", vectors=None,
        **kwargs
    ):
        pass


class RotowireBatch(Batch):
    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))

            # 2d attn
            maxe = max(len(x.uentities) for x in data)
            maxt = max(len(x.utypes) for x in data)
            padded = []
            tostack = []
            tostack_v = []
            for x in data:
                ue = x.uentities
                ut = x.utypes
                lene = len(ue)
                lent = len(ut)
                etmap = {(e,t): v for e,t,v in zip(x.entities, x.types, x.values)}
                array = [
                    [etmap[(e, t)] if (e,t) in etmap else NONE for t in ut]
                        + [PAD] * (maxt - lent)
                    for e in ue
                ] + [[PAD] * maxt] * (maxe - lene)
                tensor = dataset.fields["values_text"].numericalize(
                    (array, [lent] * lene), device=device)
                tensor_v = dataset.fields["values"].numericalize(
                    (array, [lent] * lene), device=device)

                ue, ue_info = self.uentities
                ut, ut_info = self.utypes
                padded.append(array)
                tostack.append(tensor[0].rename("els", "t").rename("batch", "e"))
                tostack_v.append(tensor_v[0].rename("els", "t").rename("batch", "e"))
            setattr(self, "vt2d", ntorch.stack(tostack, "batch"))
            setattr(self, "v2d", ntorch.stack(tostack_v, "batch"))

            # ie stuff
            ie_etv = []
            ie_d = []
            num_cells = 0
            for x in data:
                etvx = x.ie_etv
                etvs = []
                for etv in etvx:
                    T = len(etv)
                    e = [x[0] for x in etv]
                    t = [x[1] for x in etv]
                    v0 = [x[2] for x in etv]
                    e, _ = dataset.fields["entities"].numericalize(
                        ([e], [T]), device="cpu")
                    t, _ = dataset.fields["types"].numericalize(
                        ([t], [T]), device="cpu")
                    v, _ = dataset.fields["values"].numericalize(
                        ([v0], [T]), device="cpu")
                    vt, _ = dataset.fields["values_text"].numericalize(
                        ([v0], [T]), device="cpu")
                    etvs.append((
                        e.get("batch", 0).rename("els", "e"),
                        t.get("batch", 0).rename("els", "t"),
                        v.get("batch", 0).rename("els", "v"),
                        vt.get("batch", 0).rename("els", "x"),
                    ))
                    num_cells += T
                ie_etv.append(etvs)

                ie_et_d = x.ie_et_d
                d = {}
                for k, v in ie_et_d.items():
                    T = len(v)
                    e = [x[0] for x in v]
                    t = [x[1] for x in v]
                    e, _ = dataset.fields["entities"].numericalize(
                        ([e], [T]), device="cpu")
                    t, _ = dataset.fields["types"].numericalize(
                        ([t], [T]), device="cpu")
                    d[k] = (
                        e.get("batch", 0).rename("els", "e"),
                        t.get("batch", 0).rename("els", "t"),
                    )
                    num_cells += T
                ie_d.append(d)

            setattr(self, "ie_etv", ie_etv)
            setattr(self, "ie_et_d", ie_d)
            setattr(self, "num_cells", num_cells)

            # indices
            idxs = [x.idx for x in data]
            setattr(self, "idxs", idxs)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(torch.typename(self))

        fields_to_index = filter(lambda field: field is not None, self.fields)
        var_strs = '\n'.join(['\t[.' + name + ']' + ":" + _short_str(getattr(self, name))
                              for name in fields_to_index if hasattr(self, name)])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name')
                    and isinstance(self.dataset.name, str) else '')

        strt = '[{} of size {}{}]\n{}'.format(torch.typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt

def _short_str(tensor):
    if isinstance(tensor, NamedTensor):
        tensor = tensor.values
    # unwrap variable to tensor
    if not torch.is_tensor(tensor):
        # (1) unpack variable
        if hasattr(tensor, 'data'):
            tensor = getattr(tensor, 'data')
        # (2) handle include_lengths
        elif isinstance(tensor, tuple):
            return str(tuple(_short_str(t) for t in tensor))
        # (3) fallback to default str
        else:
            return str(tensor)

    # copied from torch _tensor_str
    size_str = 'x'.join(str(size) for size in tensor.size())
    device_str = '' if not tensor.is_cuda else \
        ' (GPU {})'.format(tensor.get_device())
    strt = '[{} of size {}{}]'.format(torch.typename(tensor),
                                      size_str, device_str)
    return strt

class RotowireIterator(BucketIterator):
    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield RotowireBatch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return



if __name__ == "__main__":
    filepath = "rotowire/"
    reset = True
    ENT, TYPE, VALUE, VALUE_TEXT, TEXT = make_fields()

    train, valid, test = RotoDataset.splits(
        ENT, TYPE, VALUE, VALUE_TEXT, TEXT, path=filepath,
        reset=reset,
    )
    build_vocab(ENT, TYPE, VALUE, TEXT, train)
    TEXT.vocab.extend(VALUE.vocab)
    VALUE_TEXT.vocab = TEXT.vocab

    train_iter, valid_iter, test_iter = RotowireIterator.splits(
        (train, valid, test), batch_size=6, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    import pdb; pdb.set_trace()

    train, valid, test = RotoDataset.splits(
        ENT, TYPE, VALUE, VALUE_TEXT, TEXT, path=filepath,
        sentences=True,
        reset=reset,
    )
    build_vocab(ENT, TYPE, VALUE, TEXT, train)
    TEXT.vocab.extend(VALUE.vocab)
    VALUE_TEXT.vocab = TEXT.vocab

    train_iter, valid_iter, test_iter = RotowireIterator.splits(
        (train, valid, test), batch_size=6, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    #import pdb; pdb.set_trace()
