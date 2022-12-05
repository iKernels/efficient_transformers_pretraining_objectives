# Effective Pre-Training objectives for Transformers

This repository contains the code for the paper [Effective Pre-Training Objectives for Transformer-based Autoencoders](https://arxiv.org/pdf/2210.13536.pdf), accepted as in the Findings track at EMNLP 2022. To cite our work, click [here](#citation).

# Getting started

## Prepare env

```bash
pip install -r requirements.txt
```

## Prepare data

Prepare the pre-training datasets with:

```bash
python -m process_datasets \
    --loader DatasetLoader \
    --name lucadiliello/english_wikipedia \
    --strategy SplitPretrainingStrategy \
    --output_folder data/wikipedia_pretraining_512 \
    --num_proc 32 \
    --batch_size 1000 \
    --tokenizer bert-base-cased \
    --max_sequence_length 512 \
    --field maintext
```

and

```bash
python -m process_datasets \
    --loader DatasetLoader \
    --name lucadiliello/bookcorpusopen \
    --strategy SplitPretrainingStrategy \
    --output_folder data/bookcorpusopen_pretraining_512 \
    --num_proc 32 \
    --batch_size 100 \
    --tokenizer bert-base-cased \
    --max_sequence_length 512 \
    --field text
```

Now, concatenate created datasets with:

```bash
python -m process_datasets.concatenate --inputs data/wikipedia_pretraining_512 data/bookcorpusopen_pretraining_512 --output data/wikipedia_bookcorpusopen_pretraining_512
```

## Training the models

This framework in this repository is studied to easily train different kind of models on pipelines like:
- Masked Language Modeling
- Token Detection
- Answers Sentence Selection

Moreover, you can train ELECTRA like models with both a generator and a discriminator and many others various combinations of pipelines.
Since this library is based on `pytorch-lightning`, this framework will automatically work on many device types and many machines. Checkpoints will be saved both for `pytorch-lightning` (to be able to restore training) and for `transformers` (to easily share them). This library is also backed by [`transformers-lightning`](https://github.com/iKernels/transformers-lightning), which defines a series of interfaces and provides utilities for easier transformers training and testing.

Run an experiment with:

```bash
python -m transformers_framework ...
```

### Main command line arguments

The first important argument is `--pipeline`, which selects which pipeline will be used to train the transformer. For example, setting

```bash
--model masked_language_modeling
```
will activate all the arguments to define the pre-training of some model using MLM.

Setting
```bash
--model answer_sentence_selection
```

instead, will activate instead all the arguments to define the fine-tuning of some model for Answer Sentence Selection.

After chosing the pipeline, you should choose which transformer architecture to use. For example:

```bash
--model roberta
```

will fine-tune some `RoBERTa` model.

Based on which pipeline and model you selected, many other CLI arguments will be available, like

```bash
--pre_trained_model roberta-base  # both models on the huggingface hub and local checkpoints are accepted
```

to select a starting checkpoint or

```
--pre_trained_config roberta-base
```

if you want to start from scratch with a model defined by that configuration.
If you provide only a `--pre_trained_model`, then `--pre_trained_config` and `--pre_trained_tokenizer` will be set equal to it.


## Pre-Training

Pre-Train `BERT-base-uncased` of the paper with the following scripts. You may need to change `--devices`, `--accelerator`, `--train_batch_size` and `--accumulate_grad_batches` based on your machine.

**Masked Language Modeling**

```bash
python -m transformers_framework \
    --pipeline masked_language_modeling \
    --model bert \
    --devices 8 \
    --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_config bert-base-uncased \
    --pre_trained_tokenizer bert-base-uncased \
    --name bert-base-uncased-mlm \
    --output_dir outputs/pretraining \
    \
    --train_batch_size 32 \
    --train_filepath data/wikipedia_bookcorpusopen_pretraining_512 \
    --train_split train \
    --post_processors parser tokenizer masked_language_modeling \
    --data_columns text \
    --max_sequence_length 512 \
    \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-04 \
    --max_steps 1000000 \
    --checkpoint_interval 50000 \
    --weight_decay 0.01 \
    --gradient_clip_val 1.0 \
    --num_warmup_steps 10000 \
    --num_workers 16 \
    --num_sanity_val_steps 0 \
    \
    --log_every_n_steps 1000 \
    --probability 0.15 \
    --probability_masked 0.8 \
    --probability_replaced 0.1 \
    --probability_unchanged 0.1 \
```

**Random Token Substitution**

```bash
python -m transformers_framework \
    --pipeline random_token_detection \
    --model bert \
    --devices 8 \
    --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_config bert-base-uncased \
    --pre_trained_tokenizer bert-base-uncased \
    --name bert-base-uncased-rts \
    --output_dir outputs/pretraining \
    \
    --train_batch_size 32 \
    --train_filepath data/wikipedia_bookcorpusopen_pretraining_512 \
    --train_split train \
    --post_processors parser tokenizer random_token_detection \
    --data_columns text \
    --max_sequence_length 512 \
    \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-04 \
    --max_steps 1000000 \
    --checkpoint_interval 50000 \
    --weight_decay 0.01 \
    --gradient_clip_val 1.0 \
    --num_warmup_steps 10000 \
    --num_workers 16 \
    --num_sanity_val_steps 0 \
    \
    --log_every_n_steps 1000 \
    --probability 0.15 \
    --td_weight 50 \
```

**Cluster-based Random Token Substitution**

```bash
python -m transformers_framework \
    --pipeline cluster_random_token_detection \
    --model bert \
    --devices 8 \
    --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_config bert-base-uncased \
    --pre_trained_tokenizer bert-base-uncased \
    --name bert-base-uncased-crts \
    --output_dir outputs/pretraining \
    \
    --train_batch_size 16 \
    --train_filepath data/bookcorpusopen_wikipedia_pretraining_512 \
    --train_split train \
    --post_processors parser tokenizer random_token_detection \
    --data_columns text \
    --max_sequence_length 512 \
    \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-04 \
    --max_steps 1000000 \
    --checkpoint_interval 50000 \
    --weight_decay 0.01 \
    --gradient_clip_val 1.0 \
    --num_warmup_steps 10000 \
    --num_workers 16 \
    --num_sanity_val_steps 0 \
    \
    --log_every_n_steps 1000 \
    --probability 0.15 \
    --beta 2.0 \
    --clusters_filename clusters/cluster_bert_base_uncased_100.txt
```


**Swapped Language Modeling**

Run script is the same of Masked Language Modeling, but change name and MLM probabilities:

```bash
    --name bert-base-uncased-slm \
    --probability_masked 0.0 \
    --probability_replaced 1.0 \
    --probability_unchanged 0.0 \
```

### Small models

For small models, just change config and tokenizer:

```bash
    --pre_trained_config configurations/bert-small-uncased \
    --pre_trained_tokenizer configurations/bert-small-uncased \
```

and adjust training parameters like `max_sequence_length` and `learning_rate`.



## Fine-Tuning

### Answer Sentence Selection

First, download the required datasets:

```python
from datasets import load_dataset
load_dataset('lucadiliello/asnq').save_to_disk('data/asnq')
load_dataset('lucadiliello/trecqa').save_to_disk('data/trecqa')
load_dataset('lucadiliello/wikiqa').save_to_disk('data/wikiqa')
```

**ASNQ**

```bash
python -m transformers_framework \
    --pipeline answer_sentence_selection \
    --model bert \
    --devices 8 \
    --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_model <path_to_pretrained_model> \
    --name bert-base-uncased-<mlm|rts|crts|slm>-asnq \
    --output_dir outputs/finetuning \
    \
    --train_batch_size 256 \
    --train_filepath data/asnq --train_split train \
    --valid_filepath data/asnq --valid_split dev \
    --test_filepath data/asnq --test_split test \
    --post_processors parser tokenizer \
    --data_columns question answer \
    --label_column label \
    --key_column key \
    --max_sequence_length 128 \
    \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-05 \
    --max_epochs 6 \
    --weight_decay 0.01 \
    --gradient_clip_val 1.0 \
    --num_warmup_steps 1000 \
    --num_workers 16 \
    --early_stopping \
    --monitor valid/map \
    --patience 5 \
    --val_check_interval 0.5 \
    --log_every_n_steps 100 \
```

**WikiQA** and **TREC-QA**

```bash
python -m transformers_framework \
    --pipeline answer_sentence_selection \
    --model bert \
    --devices 1 \
    --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_model <path_to_pretrained_model> \
    --name bert-base-uncased-<mlm|rts|crts|slm>-<wikiqa|trecqa> \
    --output_dir outputs/finetuning \
    \
    --train_batch_size 128 \
    --train_filepath data/<wikiqa|trecqa> --train_split train \
    --valid_filepath data/<wikiqa|trecqa> --valid_split dev_clean \
    --test_filepath data/<wikiqa|trecqa> --test_split test_clean \
    --post_processors parser tokenizer \
    --data_columns question answer \
    --label_column label \
    --key_column key \
    --max_sequence_length 128 \
    \
    --accumulate_grad_batches 1 \
    --learning_rate 1e-05 \
    --max_epochs 40 \
    --weight_decay 0.01 \
    --gradient_clip_val 1.0 \
    --num_warmup_steps 1000 \
    --num_workers 16 \
    --early_stopping \
    --monitor valid/map \
    --patience 5 \
    --val_check_interval 0.5 \
    --log_every_n_steps 100 \
```


## (Advanced) General CLI arguments

All the arguments of `pytorch-lightning` trainer are integrated in this framework. So if you want for example to train over 8 GPUs in `fp16` with `deepspeed`, just pass the followings:

```bash
--devices 8 \
--accelerator gpu \
--strategy deepspeed_stage_2 \  # or deepspeed_stage_1
--precision 16 \
```

Is it also very important (and mandatory) to assign a unique name to every run through:

```
--name my-first-run-with-transformers-framework
```

Moreover, remember to set the output directory to some disk with low latency because otherwise the script will spend most of the time writing logs and checkpoints. If you cannot, consider increasing the interval between disk writes by setting `--log_every_n_steps` to a larger value.

You can set the output directory with:

```
--output_dir /path/to/outputs
```

After training, the `output_dir` will contain 4 subdirectories:
- `tensorboard`: the tensorboards of the actual run;
- [`jsonboard`](https://github.com/lucadiliello/jsonboard): an alternative to tensorboard based only on `jsonl` files (DPES friendly);
- `pre_trained_models`: folder with all the `transformers` checkpoints, saved with `model.save_pretrained()`;
- `checkpoints`: full training checkpoint in `pytorch-lightning` format with also optimizer and scheduler states to restore training;

Experiments will be differentiated by the name you set before.

Other general arguments comprehend `--accumulate_grad_batches X` for gradient accumulation, `--learning_rate 1e-04`, `--max_steps X` or `--max_epochs` and many others. To change optimizer or scheduler, just set `--optimizer` or `--scheduler`.


### Data

In order to use this library, your data should be `jsonlines` files or `DatasetDict` (or `Dataset`) instances, which are the formats loadable with the Huggingface `datasets` library. You can potentially load very large datasets without consuming lot of RAM because the `datasets` library is based on` Apache Arrow`, which uses memory mapping between RAM and disk. If your dataset is in `csv` or `tsv`, you can easily convert it to be a `Dataset` instance with [`load_dataset`](https://huggingface.co/docs/datasets/loading) method. Loading directly from the HF hub will be added in the future.

The dataset used for pre-training and fine-tuning are already compatible with the this framework.

The loaded dataset(s) will be subject to two processing steps: one before the training (which allows also to split examples in many others or to do filtering) and one while training (for preparation of examples like tokenization). The first step is called `pre_processing` while the second is called `post_processing`.

Pre-processors and post-processors are classes that have a `__call__` method. No pre-processors are defined yet. You can create your own if necessary. Regarding post-processors, we made available classes for advanced tokenization and other simple tasks. You can add your custom pre- or post-processor in the folder `transformers_framework/processing`.

#### Tokenization

The `tokenizer` post-processor is what you will use most of the times (without pre-processors). Setting the CLI argument `--post_processors parser tokenizer` will make available some additional CLI parameters:

```bash
--data_columns question answer \
--label_column label \
--key_column key \
```

The `--key_column` argument is very important to group examples together. For example, in AS2 you have to group all the QA pairs with the same question together to compute the metrics like MAP. So make sure that your dataset has some column where a unique integer is assigned to every line corresponding to the same group. If you don't need grouping like in AS2, just use `--auto_generate_keys`.

`--label_column` is the name of the column that will be used as labels in your task. You can avoid this arguments in pipelines like `masked_language_modeling` where labels are automatically generated while training in a self-supervised way.

`--data_columns` is the most important argument of the `tokenizer` post-processor. It says which columns should be encoded to create the inputs for the model. For example, if your dataset has two columns called `question` and `candidate`, you can set field names to `--data_columns question candidate` to encode them together. You can specify up to 2 entities to be encoded together.

If you want to concatenate some fields together, you can use `:`, for example `--data_columns question previous:answer:successive`. Notice that `previous:answer:successive` will be concatenated with `--concat_char` **before** the tokenization.

If you want to concatenate all strings in a list, do `--data_columns question *candidates`. Otherwise, if you want to unpack all the candidates do `--data_columns +text`.


### Datasets

Finally, pass the path to the `dataset` instance(s) or the `jsonl` folder(s) with

```bash
--train_filepaths /path/to/the/data ... \
--valid_filepaths /path/to/the/data ... \
--test_filepaths /path/to/the/data ... \
```

If you provide more than 1 path, datasets will be concatenated. Make sure they have the same schema (features).

Notice that if the path points to a `DatasetDict` instance, you should also specify which split to use. By default, the split will be `train` for training, `valid` for validation and `test` for testing. You can define the split with `--train_split train`, `--valid_split dev` and so on.

If you skip one of the arguments before, the corresponding step will be skipped. If you want to `test` on the best checkpoint found while training, use early stopping over some metric. For example:

```bash
--early_stopping \
--patience 5 \
--monitor valid/map \
--val_check_interval 0.25 \
```

will do 4 validation per each training epoch and check the validation MAP (from the AS2 pipeline). If for `5` times in a row the MAP will not improve, training will be stopped and the best checkpoints (selected with `valid/map`) will be automagically used for testing.


## FAQ
- Multi node training: just run the same script on all the machines (just once per machine) and set `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK` and `WORLD_SIZE` on all machines. For `MASTER_ADDR` and `MASTER_PORT`, just select one of the machines that will behave as master and a free open port (possibly > 1024). Assign `NODE_RANK=0` to the master.


### Mac with Apple Silicon

You can train on Macs using Apple Silicon processors with the env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` and by setting `--accelerator mps` and `--num_workers 0`. Using many workers results in errors which we need still to debug.


<a name="citation"></a>
# Citation

If you found our work interesting or you plan to use the code in this repository, please cite us:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.13536,
  doi = {10.48550/ARXIV.2210.13536},
  url = {https://arxiv.org/abs/2210.13536},
  author = {Di Liello, Luca and Gabburo, Matteo and Moschitti, Alessandro},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Effective Pre-Training Objectives for Transformer-based Autoencoders},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```