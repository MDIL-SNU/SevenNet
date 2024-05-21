## SevenNet-0
Updated: 2024-05-23
SevenNet-0 is an interatomic potential pre-trained on the [M3GNet dataset](https://figshare.com/articles/dataset/MPF_2021_2_8/19470599).

**Warning:** Please update the potential with this version if you used SevenNet-0 before May 01, 2024. We deprecated the previous version, which has a bug related to the cutoff function.

It can be directly applied to any system without training and fine-tuned with another dataset if accuracy is unsatisfactory.

- checkpoint_sevennet_0.pth: checkpoint of SevenNet-0 model.
- serial_model/deployed_serial.pt: LAMMPS potential for single GPU runs.
- parallel_model/deployed_parallel_{0-4}.pt: LAMMPS potential for multi-GPU runs.
- fine_tune.yaml: example input.yaml for fine-tuning.

### Accuracy

|                |Energy (eV/atom)|Force (eV/Å)|Stress (GPa)|
|----------------|--------|-------|-------|
|Train|0.018|0.041|0.33|
|Valid|0.033|0.063|0.58|
|Test|0.025|0.067|0.67|


### Note
You can obtain the same deployed models by running the following command (-p for parallel)
```bash
$ sevenn_get_model {-p} checkpoint_sevennet_0.pth
```
Refer to example_inputs/md_{serial/parallel}_example/ for their usage in LAMMPS.
Both serial and parallel model gives the same results but the parallel model enable multi-GPU MD simulation.

