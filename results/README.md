# Results

The benchmark writes these files:

## evaluations.csv

One row per objective evaluation.

`run_id`, `feature`, `model`, `optimizer`, `seed`, `eval_index`, `params`, `val_macro_f1`, `val_accuracy`, `fit_time_s`, `gpu_mem_mb`, `ram_mb`.

## runs.csv

One row per optimizer/seed run.

`run_id`, `feature`, `model`, `optimizer`, `seed`, `best_val_macro_f1`, `best_val_accuracy`, `best_params`, `total_time_s`, `evals_to_best`, `evals_to_95`, `model_size_mb`, `param_count`, `inference_ms_batch1`, `inference_ms_batch32`, `gpu_mem_mb`, `backend`, `history`.

## final.csv

One row per final feature/model.

`feature`, `model`, `optimizer`, `seed`, `best_params`, `train_size`, `test_accuracy`, `test_macro_f1`, `test_precision`, `test_recall`, `per_class`, `confusion_matrix`, `train_time_s`, `inference_ms_batch1`, `inference_ms_batch32`, `model_size_mb`, `param_count`, `gpu_mem_mb`, `backend`.

## cnn.csv

One row per CNN seed.

`seed`, `epochs`, `batch_size`, `lr`, `weight_decay`, `train_size`, `val_macro_f1`, `test_accuracy`, `test_macro_f1`, `test_precision`, `test_recall`, `per_class`, `confusion_matrix`, `train_time_s`, `inference_ms_batch1`, `inference_ms_batch32`, `model_size_mb`, `param_count`, `gpu_mem_mb`, `backend`, `history`.

## ablation.csv

One row per ablation variant/model/seed.

`variant`, `model`, `seed`, `val_macro_f1`, `test_accuracy`, `test_macro_f1`, `per_class`, `confusion_matrix`, `train_time_s`, `inference_ms_batch1`, `inference_ms_batch32`, `model_size_mb`, `param_count`, `gpu_mem_mb`, `backend`.

## metadata.json

Config, Python version, package versions, CUDA availability, and GPU name.

## plots/

Generated figures referenced by `docs/benchmark.md`.
