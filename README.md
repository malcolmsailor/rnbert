# Roadmap

We cobbled the initial version of this repo together quickly to meet the ISMIR deadline. Here are some next steps we intend to do in the short to medium term future:

1. Include the dataset creation code (i.e., the code that creates `dataset.zip`).
2. Port the model from `fairseq` to `huggingface`. Using a more actively maintained library should simplify the environment creation, allow others to use the model more easily, and make it easier to try more recent fine-tuning techniques like LORA.
3. Include tools for visualizing the predictions.

# 1. Set up environment

We use two different environments, the first, `write_seqs` to write the dataset in the OctupleMIDI format, and the second, `rnbert` for the fine-tuning. We found setting up a `fairseq` environment capable of running the `MidiBERT` checkpoint to be quite finicky and were only able to get it working with Python 3.8, whereas the code used to write the dataset requires Python >= 3.11.

## Create `write_seqs` environment

First create the `write_seqs` environment with conda or pip according to your preference, then do

```bash
pip install -r write_seqs_requirements.txt
```

## Create `rnbert` environment

```bash
conda create --name rnbert --file rnbert_environment.yaml
conda activate rnbert
pip install -r rnbert_extra_requirements.txt
```

# 2. Build the data

To specify where the following commands put the dataset, set the `RNDATA_ROOT` environment variable. The default location is `${HOME}/datasets`.

In the `write_seqs` environment, make the raw dataset:

```bash
bash scripts/make_raw_sequences.sh
```

In the `rnbert` environment, binarize the dataset:

```bash
bash scripts/binarize_sequences.sh
```

The above command first binarizes an "abstract" dataset containing all the features we might wish to predict, and then instantiates specific versions of it with symlinks for the key prediction, conditioned roman numeral prediction, and unconditioned roman numeral prediction tasks.

# 3. Download checkpoint

Download the `musicbert_base` checkpoint from [https://1drv.ms/u/s!Av1IXjAYTqPsuBaM9pZB47xjX_b0?e=wg2D5O](https://1drv.ms/u/s!Av1IXjAYTqPsuBaM9pZB47xjX_b0?e=wg2D5O). Save it wherever you like and then assign the MUSICBERT_DEFAULT_CHECKPOINT environment variable to its path:

```bash
export MUSICBERT_DEFAULT_CHECKPOINT=/path/to/checkpoint
```

# 4. Fine-tune RNBert

<!-- TODO 2024-04-11 update paths -->

Run the following commands inside the `rnbert` environment. Optionally, you can add a `-W/--wandb-project [project name]` argument to any of the below commands to log the training metrics to a wandb project.

## Train key prediction model

```bash
python musicbert_fork/training_scripts/train_chord_tones.py \
    -d TODO \
    -a base \
    --validate-interval-updates 2500 \
    --lr 0.00025 \
    --freeze-layers 9 \
    --total-updates 25000 \
    --warmup-updates 2500 \
    --fp16
```

## Train unconditioned roman numeral model


```bash
python training_scripts/train_chord_tones.py \
    -a base \
    -d TODO \
    --multitask \
    --validate-interval-updates 2500 \
    --lr 0.00025 \
    --fp16 \
    --freeze-layers 9 \
    --total-updates 50000 \
    --warmup-updates 2500
```

## Train conditioned roman numeral model

```bash
python training_scripts/train_chord_tones.py \
    -a dual_encoder_base \
    -d TODO \ 
    --conditioning key_pc_mode \
    --multitask \
    --validate-interval-updates 2500 \
    --lr 0.00025 \
    --fp16 \
    --freeze-layers 9 \
    --total-updates 50000 \
    --warmup-updates 2500 \
    --z-encoder mlp \
    --z-embed-dim 256
```

# 5. Get evaluation metrics
