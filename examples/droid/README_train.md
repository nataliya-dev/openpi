# Training on DROID

Here we describe how to fine-tune the pi0-FAST model on the DROID dataset. This is an approximate open-source reproduction of the pi0-FAST-DROID training pipeline.
(small differences in data loading and the used action space)

In contrast to the rest of openpi, which uses LeRobot for data loading, we need to use RLDS as the data format for DROID training (since atm LeRobot isn't scalable enough 
for larger datasets like DROID -- they are working on improving it though). Below, we provide instructions for updating your openpi environment for RLDS data loading and where to download the DROID dataset.

## Install

After you followed the openpi installation instructions from the main README, you need to install the following data loading dependencies for DROID RLDS training:
```bash
uv pip install tensorflow_datasets==4.9.9
uv pip install tensorflow==2.15.0
uv pip install "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
uv pip install ml-dtypes==0.5.1  # somehow the previous installations mess up this package version, so we install it again
```

## Download DROID dataset

You can download a (slightly outdated) version of DROID with the following command (after installing the `gsutil` google cloud CLI):
```
gsutil -m cp -r gs://gresearch/robotics/droid <your_download_path>
```

Note that this version of DROID is slightly outdated: it only contains a partial set of language annotations (~30k episodes).
Please email (mailto:karl.pertsch@gmail.com)[karl.pertsch@gmail.com] to get access to the most up-to-date version of the DROID RLDS dataset (with language annotations on 75k episodes)!
(sorry, we are working on updating the version on the official bucket).

You will need 1.8TB of disk storage to download the DROID RLDS dataset.

## Run

Compute normalization statistics (this will take ~10 minutes):
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid_finetune --max-frames 10_000_000
```

Run training:
```bash
uv run scripts/train.py pi0_fast_droid_finetune --exp-name=my_experiment --overwrite
```

**Note**: The original pi0-FAST-DROID model was trained with joint velocity actions.
Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend training with joint velocity actions and instead use joint position actions here.


## Compute Requirements

Our DROID training config requires approximately 2 days on 8x H100 GPUs for convergence (100k iterations, approx. 1 epoch).
If you start from PaliGemma instead of pi0 initialization, plan with ~5 days on 8x H100s (240k iterations, i.e. 3 epochs).

We have experimented with LoRA for cheaper finetuning, but haven't found the policies to perform well so far.









