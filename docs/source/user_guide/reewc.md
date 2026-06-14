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

## Configuration

reEWC adds keys to the `data:` and `train.continue:` blocks of `input.yaml`. Every
key is optional; when none are set, training is unchanged.

Replay (`data:`):

```yaml
data:
    rehearsal: True                          # enable experience replay
    load_memory_path: ['./memory.extxyz']    # old-task (memory) set; reserved for replay
    mem_batch_size: 8                        # batch size for the memory set
    mem_ratio: 1                             # fraction (0, 1] of the memory set to use
```

EWC (`train.continue:`):

```yaml
train:
    continue:
        checkpoint: '7net-0'                 # model to fine-tune
        fisher_information: './fisher.pt'    # dict {param_name: tensor} of Fisher information
        opt_params: './opt_params.pt'        # dict {param_name: tensor} of reference parameters
        ewc_lambda: 100000                   # EWC penalty weight (> 0)
```

`fisher_information` and `opt_params` are **precomputed and consumed** -- SevenNet does
not estimate the Fisher matrix. Both are `torch.save`d dictionaries keyed by parameter
name, matching the model's trainable parameters; `opt_params` is the parameter set of
the checkpoint before fine-tuning. `ewc_lambda` requires both to be set.

## Examples

Replay only:

```yaml
train:
    continue: {checkpoint: '7net-0'}
data:
    load_trainset_path: ['./target_train.extxyz']
    rehearsal: True
    load_memory_path: ['./memory.extxyz']
    mem_batch_size: 8
```

EWC only:

```yaml
train:
    continue:
        checkpoint: '7net-0'
        fisher_information: './fisher.pt'
        opt_params: './opt_params.pt'
        ewc_lambda: 100000
data:
    load_trainset_path: ['./target_train.extxyz']
```

reEWC (replay + EWC):

```yaml
train:
    continue:
        checkpoint: '7net-0'
        fisher_information: './fisher.pt'
        opt_params: './opt_params.pt'
        ewc_lambda: 100000
data:
    load_trainset_path: ['./target_train.extxyz']
    rehearsal: True
    load_memory_path: ['./memory.extxyz']
    mem_batch_size: 8
```

Run any of them with the standard training command:

```bash
sevenn train input.yaml -s
```

When replay is enabled, the memory set is evaluated each epoch and logged as a
`memoryset` column group in `lc.csv`, alongside `trainset` and `validset`, so the
preserved old-task accuracy can be tracked during training.

## Notes

- Replay supports `dataset_type: 'graph'` (the default) only.
- reEWC does not support distributed (DDP) training.
- `load_memory_path` is reserved for replay: setting it without `rehearsal: True`
  raises an error.
- A `cosineannealingwarmuplr` scheduler (cosine annealing with warm-up restarts,
  used for the reEWC paper work) is also available for fine-tuning.
