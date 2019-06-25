# genie

## Todo

### Ideas
* Buffer (v,y) that result in best likelihoods p(y|v) and enumerate over top K

### Experiments
* evaluate rotowire on full database PR
* try: gumbel c + hard attn, soft c + hard attn. Reason: avoid soft pretraining, as it biases the model.
  want to see: inference network gets team names and names with higher prob using content model
* LOO
* Buffer top likelihoods for words

### Model
* translation model CRF, and figure out induction

### Coding

#### p(v|y) and eval
* Maximize likelihood of value given words and alignment
* Approximate q(c|y) (at end of training)
* Compute precision and recall from greedily extracted alignments and translation
  versus ground truth data table 

#### Replay Buffer
* Key on time and example
* Need to have unique key for each example (idx)
 
#### Refactor
* pull out code into (conditional) distributions
* refactor out debugging code and evaluation code into hooks
* write concrete sampling code + KL + log prob
* pull out cat vs concrete inference code
* maybe pull out inference code? this one is difficult

#### Misc
* LOO baseline
* Buffer likelihoods


## commands
* LM + Attn: Soft c soft a
```
python main.py --filepath ../rotowIrE/rotowire --devid 3 --model crnnlmsa --tieweights --lr 0.001 --bsz 6 --K 64 --Ke 16 --dp 0 --T 64 --mode elbo --re 10 --klannealsteps 0 --jointcopy --seed 123 --prefix dbg-soft --save
```
* LM + Attn: Hard c hard a, trained from soft
```
python main.py --filepath ../rotowIrE/rotowire --devid 1 --model crnnlmeqca --tieweights --lr 0.001 --bsz 6 --K 64 --Ke 32 --dp 0 --T 64 --mode elbo --re 10 --klannealsteps 0 --jointcopy --qtao 1 --prefix lolhuh --seed 123 --cwarmupsteps 1000 --cannealsteps 1000 --Kl 32 --nuisance --train-from ../rotowIrE/dbg-crnnlmsa-a-crnnlmsa-v2dFalse-mlpFalse-hardFalse-fixFalse-maskFalse/dbg-crnnlmsa-a-crnnlmsa-v2dFalse-mlpFalse-hardFalse-fixFalse-maskFalse-e31-vloss2.53.pt --qwarmupsteps 512 --qsteps 0
```
```
python main.py --filepath ../rotowIrE/rotowire --model crnnlmeqca --tieweights --lr 0.001 --bsz 6 --dp 0 --T 64 --mode elbo --re 10 --klannealsteps 0 --jointcopy --qtao 1 --prefix lolhuh --seed 123 --cwarmupsteps 1000 --cannealsteps 1000 --nuisance --train-from ../rotowIrE/dbg-crnnlmsa-a-crnnlmsa-v2dFalse-mlpFalse-hardFalse-fixFalse-maskFalse/dbg-crnnlmsa-a-crnnlmsa-v2dFalse-mlpFalse-hardFalse-fixFalse-maskFalse-e31-vloss2.53.pt
--qwarmupsteps 512 --qsteps 0 --devid 3 --K 64 --Kq 16 --Kb 16 --Kl 0 --epochs 10
```
