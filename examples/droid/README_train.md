# Training on DROID

## Install

Data loading dependencies:
```bash
uv pip install tensorflow_datasets
uv pip install tensorflow
uv pip install "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
uv pip install ml-dtypes==0.5.1  # somehow the previous installations mess up this package version, so we install it again
```

## Download DROID dataset

## Run

Compute normalization statistics (this will take ~5 minutes):
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid_finetune --max-frames 500_000
```

Run training:
```bash
uv run scripts/train.py pi0_fast_droid_finetune --exp-name=my_experiment --overwrite
```


## Compute Requirements

The table below shows approximate GPU memory requirements for different batch sizes and LoRA configurations when training on DROID data:

| Configuration | Batch Size | GPUs Required | Iteration Time (sec/batch)|
|--------------|------------|---------------------|----------------------------|
| No LoRA      | 256        | 8xH100              | 1.7 |
| No LoRA      | 128        | 4xH100              | 1.7 |
| LoRA r=16    | 128        | 1xH100              | 5.2 |
| LoRA r=128   | 128        | 1xH100              | 5.4 |

Notes:
- Using LoRA reduces memory usage by freezing base model weights
- Larger LoRA ranks (r) provide more model capacity but require more memory
- Batch size has the largest impact on memory usage


## Joint Velocity Training

The original pi0-fast-droid checkpoint was trained using joint velocity actions. We provide a config for joint velocity training below:

```bash
uv run scripts/train.py pi0_fast_droid_finetune_joint_velocity --exp-name=my_experiment --overwrite
```

**Note**: Joint velocity actions are not compatible with simulated evaluation environments (much harder to simulate). 
Thus, we do not recommend using this config for training going forward.






