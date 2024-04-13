# Roadmap

We cobbled the initial version of this repo together quickly to meet the ISMIR deadline. Here are some next steps we intend to do in the short to medium term future:

1. Include the dataset creation code (i.e., the code that creates the `dataset.zip` currently included in this repository).
2. Port the model from `fairseq` to `huggingface`. Using a more actively maintained library should simplify the environment creation, allow others to use the model more easily, and make it easier to try more recent fine-tuning techniques like LORA.
3. Include our tools for visualizing the predictions on scores.

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

## Environment variables

There are a few environment variables that control the behavior of the scripts. You can leave them with their default values or set them as you prefer:

- `RNDATA_ROOT`: where the data is saved. Default: `${HOME}/datasets`.
- `RN_CKPTS`: where checkpoints are saved. Default: `${HOME}/saved_checkpoints/rnbert`.
- `RN_PREDS`: where predictions are saved. Default: `${HOME}/saved_predictions/rnbert`.


# 2. Build the data

To specify where the following commands put the dataset, set the `RNDATA_ROOT` environment variable. The default location is `${HOME}/datasets`.

In the `write_seqs` environment, make the raw dataset (sadly, quite slow):

```bash
bash scripts/make_raw_sequences.sh
```

In the `rnbert` environment, binarize the dataset:

```bash
bash scripts/binarize_sequences.sh
```

The above command first binarizes an "abstract" dataset containing all the features we might wish to predict, and then instantiates specific versions of it with symlinks for the key prediction, conditioned roman numeral prediction, and unconditioned roman numeral prediction tasks.

## Make key-conditioned test dataset

To get the metrics for the key-conditioned model, using predicted keys, run the following command in the `rnbert` environment. First, you'll need to [train a key prediction model](#train-key-prediction-model) and note the associated run id.

```bash
bash scripts/make_key_cond_data.sh [KEY_RUN_ID]
```

# 3. Download checkpoint

Download the `musicbert_base` checkpoint from [https://1drv.ms/u/s!Av1IXjAYTqPsuBaM9pZB47xjX_b0?e=wg2D5O](https://1drv.ms/u/s!Av1IXjAYTqPsuBaM9pZB47xjX_b0?e=wg2D5O). Save it wherever you like and then assign the MUSICBERT_DEFAULT_CHECKPOINT environment variable to its path:

```bash
export MUSICBERT_DEFAULT_CHECKPOINT=/path/to/checkpoint
```

# 4. Fine-tune RNBert

Run the following commands inside the `rnbert` environment. Optionally, you can add a `-W/--wandb-project [project name]` argument to any of the below commands to log the training metrics to a wandb project. 

These commands fine-tune a model, saving checkpoints to the ${RN_CKPTS} directory and saving the logits on the test set to the ${RN_PREDS} directory.

## Train key prediction model

```bash
python musicbert_fork/training_scripts/train_chord_tones.py \
    -a base \
    -d ${RNDATA_ROOT-${HOME}/datasets}/rnbert_key_data_bin \
    --validate-interval-updates 2500 \
    --lr 0.00025 \
    --freeze-layers 9 \
    --total-updates 25000 \
    --warmup-updates 2500 \
    --fp16
```

## Train unconditioned roman numeral model


```bash
python musicbert_fork/training_scripts/train_chord_tones.py \
    -a base \
    -d ${RNDATA_ROOT-${HOME}/datasets}/rnbert_rn_uncond_data_bin \
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
python musicbert_fork/training_scripts/train_chord_tones.py \
    -a dual_encoder_base \
    -d ${RNDATA_ROOT-${HOME}/datasets}/rnbert_rn_cond_data_bin \
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

## Save roman numeral predictions conditioned on predicted keys (for testing)

First, [train a key prediction model](#train-key-prediction-model) and [train a conditioned roman numeral model](#train-conditioned-roman-numeral-model), noting the run ids associated with each. Then [make the key-conditioned test set](#make-key-conditioned-test-dataset). 

Now assign the following variables:

```bash
RN_RUN_ID=# Run id of the conditioned roman numeral model checkpoint you want to use
KEY_RUN_ID=# Run id of the key model whose predictions you are using
```

Then run the following command (ideally with CUDA):

```bash
python musicbert_fork/eval_scripts/save_multi_task_predictions.py \
    --dataset test \
    --data-dir "${RNDATA_ROOT-${HOME}/datasets}/rnbert_rn_cond_test_data_bin" \
    --checkpoint "${RN_CKPTS}/${RN_RUN_ID}/checkpoint_best.pt" \
    --output-folder "${RN_PREDS}"/${RN_RUN_ID}_predicted_keys_from_${KEY_RUN_ID} \
    --task musicbert_conditioned_multitarget_sequence_tagging
```

# 5. Get evaluation metrics

These commands should be run in the `write_seqs` environment. You'll need to note the "RUN_ID", which is a numeric string under which the logits will have been saved in `${RN_PREDS}`. If you're running on SLURM, it'll be the ID of the job. Otherwise, it'll be taken from the system clock.

## Key metrics

```bash
bash scripts/rnbert_key_metrics.sh [RUN_ID]
```

## Unconditioned roman numeral metrics

```bash
bash scripts/rnbert_unconditioned_metrics.sh [RUN_ID]
```

## Conditioned roman numeral metrics (teacher forcing)

```bash
bash scripts/rnbert_conditioned_metrics.sh [RUN_ID]
```

## Conditioned roman numeral metrics (with predicted keys)

```bash
bash scripts/rnbert_conditioned_on_preds_metrics.sh [RN_RUN_ID] [KEY_RUN_ID]
```


# 6. Run existing checkpoints

TODO
