## SevenNet-0 (22May2024)
SevenNet-0 (22May2024) is an interatomic potential pre-trained on the [M3GNet dataset](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599).

We released a new pre-trained model on July 11. Please check 'SevenNet/pretrained_potentials/SevenNet_0__11July2024' for details.

This is the model referred as SevenNet-0 in [our paper](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00190). We distinguish SevenNet-0 models by its release date. The number of parameters and model architecture are the same, but the training process and dataset differ.

**Warning:** Please update the potential with this version if you used SevenNet-0 before May 01, 2024. We deprecated the previous version, which has a bug related to the cutoff function.

It can be directly applied to any system without training and fine-tuned with another dataset if accuracy is unsatisfactory.

- checkpoint_sevennet_0.pth: checkpoint of SevenNet-0 model.
- fine_tune.yaml: example input.yaml for fine-tuning.

### Accuracy
MAE (Mean absolute error)
|                |Energy (eV/atom)|Force (eV/Å)|Stress (GPa)|
|----------------|--------|-------|-------|
|Train|0.016|0.037|0.30|
|Valid|0.032|0.064|0.58|
|Test|0.025|0.070|0.68|

### Note
You can obtain models for LAMMPS by running the following command (-p for parallel)
```bash
$ sevenn_get_model {-p} checkpoint_sevennet_0.pth
```
Refer to example_inputs/md_{serial/parallel}_example/ for their usage in LAMMPS.
Both serial and parallel model gives the same results but the parallel model enables multi-GPU MD simulation.
