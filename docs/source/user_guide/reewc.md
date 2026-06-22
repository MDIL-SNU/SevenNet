# Forgetting-prevented (Continual-learning) fine-tuning (reEWC)

Fine-tuning a pretrained model on a target system improves accuracy there, but the
model can lose accuracy on the original training domain (catastrophic forgetting).
reEWC mitigates this with two complementary mechanisms that can be used together or
separately:

- **Experience replay (rehearsal)** -- replay an old-task "memory" set each training
  step so the model keeps fitting it while learning the target data.
- **Elastic Weight Consolidation (EWC)** -- add a penalty
  `lambda/2 * sum_i F_i (theta_i - theta*_i)^2` that anchors parameters to their
  pre-fine-tuning values `theta*`, weighted by a precomputed Fisher matrix `F`.

reEWC is for **single-modal** models (e.g. SevenNet-0, SevenNet-Nano). Multi-fidelity
(modal) models are not supported yet.

## Getting started

A ready-to-edit input with both mechanisms is available as a preset:

```bash
sevenn preset reewc > input.yaml
```

The preset documents every key inline. Replay lives in the `data:` block
(`rehearsal`, `load_memory_path`, `mem_batch_size`, `mem_ratio`) and EWC in the
`train.continue:` block (`fisher_information`, `opt_params`, `ewc_lambda`). Every
reEWC key is optional; when none are set, training is unchanged. Remove the replay
block or the EWC keys to run only one mechanism. Run training as usual:

```bash
sevenn train input.yaml -s
```

## Fisher information and reference parameters

`fisher_information` and `opt_params` are **precomputed and consumed** -- SevenNet
does not estimate the Fisher matrix. Both are `torch.save`d dictionaries keyed by
parameter name; `opt_params` is the parameter set of the checkpoint before
fine-tuning. They must satisfy:

- `fisher_information` and `opt_params` cover the **same parameter names** with the
  **same shapes** (they are a matched pair).
- Names that overlap with the model's trainable parameters must have **matching
  shapes**; a mismatch is an error (usually an incompatible checkpoint or SevenNet
  version).
- At least one name must overlap with the model; no overlap is an error.
- A trainable parameter without a Fisher entry is **left unconstrained** and a
  warning is emitted, so partial-coverage Fisher matrices are allowed but visible.

`ewc_lambda` must be `> 0`, and EWC requires both `fisher_information` and
`opt_params` to be set.

## Notes

- Replay supports `dataset_type: 'graph'` (the default) only.
- reEWC does not support distributed (DDP) training.
- `load_memory_path` is reserved for replay: setting it without `rehearsal: True`
  raises an error.
- When replay is enabled, the memory set is evaluated each epoch and logged as a
  `memoryset` column group in `lc.csv`, alongside `trainset` and `validset`.
- A `cosineannealingwarmuplr` scheduler (cosine annealing with warm-up restarts,
  used for the reEWC paper work) is also available for fine-tuning.
