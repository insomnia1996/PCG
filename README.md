# PCG

Code for *Few-Shot Table-to-Text Generation with Prefix-Controlled Generator*.

*The content plan part still need refactoring for the generation pipeline.*

## Requirements
Simply install packages in requirements.txt, don't remember to modify *configuration.py* in AdapterTransformer package:
For example, change adapter in MAMConfig to:
`adapter = adapter or ParallelConfig(mh_adapter=True, leave_out=[12,13,14,15,16,17,18,19,20,21,22,23])` for bart

## Data
We use Data from [Chen et al](https://arxiv.org/abs/1904.09521). First fetch the data from [Dropbox](https://www.dropbox.com/sh/u3t8yhcctqczpo0/AAAZV7S-qoIyaQW99r_88nUra?dl=0), then use `python split_train.py` to sample data under low-resource constraints. Then run ``bash process.sh`` to preprocess data.

## Training
Simply run `bash run.sh`. There're some configs to be clarified later.
