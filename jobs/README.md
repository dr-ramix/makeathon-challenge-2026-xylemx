# SLURM Jobs

This folder contains lightweight SLURM job templates for the XylemX baseline pipeline.

## Files

- `preprocess.slurm`: run preprocessing and cache artifacts under `output/preprocessing/`
- `train_baseline.slurm`: train the baseline segmentation model
- `train_all_models.slurm`: train all registered models sequentially, one after another
- `submit_all_models.sh`: submit one SLURM training job per model with `afterok` dependencies
- `run_all_models_local.sh`: run all registered models sequentially on the current machine without SLURM
- `run_all_models_gpu_local.sh`: run all registered models sequentially on the current machine, but fail fast unless PyTorch can see a GPU
- `predict_val.slurm`: run prediction from a trained checkpoint
- `predict_all_models.slurm`: generate prediction rasters for every trained model run
- `submit_all_and_predict.sh`: submit the full dependency chain, then automatically submit all-model prediction after training

## Before You Submit

Edit the `#SBATCH` lines to match your cluster:

- account if required
- partition
- wall time
- GPU type or count
- memory

The templates use:

- `./.venv/bin/python`
- `./.venv/bin/pip`
- repo-relative output folders

That matches the rest of this repository.

## Typical Usage

Preprocess:

```bash
sbatch jobs/preprocess.slurm
```

Train:

```bash
sbatch jobs/train_baseline.slurm
```

Train all models sequentially for 20 epochs each:

```bash
sbatch jobs/train_all_models.slurm
```

Submit one dependent SLURM job per model:

```bash
bash jobs/submit_all_models.sh
```

Submit dependent training jobs and automatically chain all-model prediction:

```bash
bash jobs/submit_all_and_predict.sh
```

Run all models locally without SLURM:

```bash
bash jobs/run_all_models_local.sh
```

Run all models locally on GPU only:

```bash
bash jobs/run_all_models_gpu_local.sh
```

Predict:

```bash
sbatch jobs/predict_val.slurm
```

Predict for all trained models:

```bash
sbatch jobs/predict_all_models.slurm
```

## Notes

- The training code already falls back to `num_workers=0` if PyTorch multiprocessing is blocked by the environment.
- In offline environments, editable install may need:

```bash
./.venv/bin/pip install -e . --no-build-isolation
```

- Job logs are written under `jobs/logs/`.
- `train_all_models.slurm` defaults to `EPOCHS=20`.
- `submit_all_models.sh` also defaults to `EPOCHS=20`.
- `run_all_models_local.sh` also defaults to `EPOCHS=20`.
