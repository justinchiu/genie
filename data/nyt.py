
from collections import defaultdict, Counter

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


class NytExample(Example):
    @classmethod
    def fromJson(
        cls, data, ent_field, type_field, value_field, text_field,
        sentences=False,
        numericvalues=False,
        supex=-1,
    ):
        exs = []
        jsonlist = [json.loads(line) for line in data]

        ents = []
        types = []
        kb = defaultdict(set)
        etv = []
        # values == ents
        for x in jsonlist:
            #text = x["sentext"]
            ents += x["entities"]
            for r in x["relations"]:
                types.append(r["rtext"])
                kb[(r["em1"], r["rtext"])].add(r["em2"])
                etv.append((r["em1"], r["rtext"], r["em2"]))
        # 16155???
        # do i let the model see the entities in the sentence? that seems like cheating
        # entities follow zipfian distribution, heavy tail
        cents = Counter(ents)
        # filter ents by # occurrences?
        g_ents = set(ents)
        # 29
        g_types = set(types)
        g_etv = set(etv)

        nrelations = [len(x["relations"]) for x in jsonlist]

        for x in jsonlist:
            # Need to flatten all the tables into aligned lists of
            # entities, types, and values.
            entities = []
            types = []
            values = []

            relations = x["relations"]
             
            def add(entity, type, value):
                entities.append(entity)
                types.append(type)
                values.append(value)

            for r in relations:
                add(r["em1"], r["rtext"], r["em2"])

            text = x["sentext"]

            ents = ent_field.preprocess(entities)
            types = type_field.preprocess(types)
            values = value_field.preprocess(values)
            values_text = text_field.preprocess(values)
            text = text_field.preprocess(text)

            uents = x["entities"] + [NONE]
            utypes = list(g_types) + [NONE]

            # predict entity and type pairs for each index
            # issue: are relations expressed left to right?
            # left-to-right prior will penalize
            #
            # left to right version
            # we use the entities and types of a relation as the
            # labels for the values
            labels = []
            for i, w in enumerate(text):
                label_i = []
                for r in relations:
                    tag = r["tags"][i]
                    if tag == 2 or tag == 5:
                        # entity 2, target / value
                        e_i = r["em1"]
                        t_i = r["rtext"]
                        v_i = r["em2"]
                        label_i.append((e_i, t_i, v_i))
                    elif tag == 1 or tag == 4:
                        # entity 1, source / argument
                        pass
                    else:
                        # non entity or not-relevant entity
                        pass
                if not label_i:
                    # If not a value
                    label_i.append((NONE, NONE, NONE))
                labels.append(label_i)
            ie_etv = labels

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
            setattr(ex, "ie_etv",       ie_etv)
            exs.append(ex)

        if supex >= 0:
            N = len(exs)
            perm = np.random.permutation(N)
            supexs = [exs[idx] for idx in perm[:supex]]
            return exs, supexs
        return exs


class NytDataset(Dataset):

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
                examples = NytExample.fromJson(
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
        super(NytDataset, self).__init__(examples, fields, **kwargs)
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
        train='train.json', validation='train.json', test='test.json',
        supex = -1,
        reset = False,
        sentences = False, # ignore
        numericvalues = False,
        **kwargs
    ):
        Dtrain, Dvalid, Dtest = super(NytDataset, cls).splits(
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
            numericvalues = numericvalues,
            **kwargs
        )
        dev_frac = 0.05
        N = len(Dtrain)
        perm = np.random.permutation(N)
        exs = np.array(Dtrain.examples)

        train_exs = list(exs[perm[int(N*dev_frac):]])
        valid_exs = list(exs[perm[:int(N*dev_frac)]])

        super(NytDataset, Dtrain).__init__(train_exs, Dtrain.fields, **kwargs)
        super(NytDataset, Dvalid).__init__(valid_exs, Dtrain.fields, **kwargs)
        if supex >= 0:
            Dtrain_sup, Dvalid_sup, Dtest_sup = super(NytDataset, cls).splits(
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
                numericvalues = numericvalues,
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


class NytBatch(Batch):
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
            import pdb; pdb.set_trace()

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

            setattr(self, "ie_etv", ie_etv)
            setattr(self, "num_cells", num_cells)


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

class NytIterator(BucketIterator):
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
                yield NytBatch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return



if __name__ == "__main__":
    filepath = "nyt/nyt10/"
    reset = True
    ENT, TYPE, VALUE, VALUE_TEXT, TEXT = make_fields()

    train, valid, test = NytDataset.splits(
        ENT, TYPE, VALUE, VALUE_TEXT, TEXT, path=filepath,
        reset=reset,
    )
    build_vocab(ENT, TYPE, VALUE, TEXT, train)
    TEXT.vocab.extend(VALUE.vocab)
    VALUE_TEXT.vocab = TEXT.vocab

    train_iter, valid_iter, test_iter = NytIterator.splits(
        (train, valid, test), batch_size=6, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    import pdb; pdb.set_trace()
