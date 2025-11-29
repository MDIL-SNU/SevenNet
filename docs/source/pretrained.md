# Pretrained models

So far, we have released multiple pretrained SevenNet models. Each model has various hyperparameters and training sets, leading to different levels of accuracy and speed. Please read the descriptions below carefully and choose the model that best suits your purpose.
We provide the F1 score, and RMSD for the WBM dataset, along with $\kappa_{\mathrm{SRME}}$ from phononDB and CPS (Combined Performance Score). For details on these metrics and performance comparisons with other pretrained models, please visit [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

These models can be used as interatomic potentials in LAMMPS and loaded through the ASE calculator using each model’s keywords. Please refer to the [ASE calculator](#ase_calculator) section for instructions on loading a model via the ASE calculator.
Additionally, `keywords` can be used in other parts of SevenNet, such as `sevenn inference`, `sevenn get_model`, and the `checkpoint` section in `input.yaml` for fine-tuning.

**Acknowledgments**: The models trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) were supported by the Neural Processing Research Center program at Samsung Advanced Institute of Technology, part of Samsung Electronics Co., Ltd. The computations for training models were carried out using the Samsung SSC-21 cluster.

```{note}
Multiple inference channels are available for multi-fidelity architecture models, SevenNet-Omni and SevenNet-MF-ompa. Each channel is designed to produce results that are consistent with the DFT settings used in the corresponding training datasets. For example, `mpa` is trained on the combined [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and [sAlex](https://huggingface.co/datasets/fairchem/OMAT24) datasets and is used for evaluating Matbench Discovery, while `omat24` is trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset. For detailed information on the DFT settings, please refer to the original publications of each dataset.
```

```{note}
We supports cuEquivariance and FlashTP kernels for tensor-product acceleration. Pass these options to `SevenNetCalculator` using `enable_cueq=True` or `enable_flash=True` when available.
```

---
## SevenNet-Omni
> Model keywords: `7net-Omni` | `SevenNet-Omni`

**This is our recommended pretrained model**
[Download link for fully detailed checkpoint](https://figshare.com/articles/software/SevenNet-Omni_checkpoint/30399814?file=58886557)

This model exploits [cross-domain knowledge transfer strategies](https://arxiv.org/abs/2510.11241) within a [multi-task training framework](https://pubs.acs.org/doi/10.1021/jacs.4c14455) to train simultaneously on the 15 open ab initio datasets, covering a wide material space including crystals, molecules, and surfaces.

It is currently our best pretrained model, achieving state-of-the-art accuracy across diverse material domains at the PBE level, while also providing high-fidelity channels such as r²SCAN and ωB97M-V.  For detailed information on the training datasets, knowledge-transfer strategies and comprehensive benchmark results, please refer to the [paper](https://arxiv.org/abs/2510.11241).

```{table} Representative channels for each fidelity
:widths: 30 70
| Fidelity       | Channel name    |
|----------------|-----------------|
| PBE(+U)        | `mpa`           |
| r²SCAN         | `matpes_r2scan` |
| ωB97M-V        | `omol25_low`    |
```

Also consider `omat24` or `matpes_pbe` channels for more accurate PBE-level descriptions of high-force configurations. Note that `matpes_pbe` is trained on PBE level of theory, without incorporating the Hubbard U correction.

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(
    model="/path/to/7net-omni",
    modal='mpa',
    enable_cueq=False,
    enable_flash=False
)
```
When using the command-line interface of SevenNet, include the channel as `--modal ${channel}` option to select the desired modality. Run `sevenn_cp SevenNet-Omni.pth` to see an overview all the available channels.

### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|**0.849**|0.889|0.265|0.0639|


---
## SevenNet-MF-ompa
> Model keywords: `7net-mf-ompa` | `SevenNet-mf-ompa`

This model leverages [multi-fidelity learning](https://pubs.acs.org/doi/10.1021/jacs.4c14455) to train simultaneously on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842), [sAlex](https://huggingface.co/datasets/fairchem/OMAT24), and [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) datasets. This model achieves a high ranking on the [Matbench Discovery]((https://matbench-discovery.materialsproject.org/)) leaderboard. Our evaluations show that it outperforms other models on most tasks, except for the isolated molecule energy task, where it performs slightly worse than `SevenNet-l3i5`.

```python
from sevenn.calculator import SevenNetCalculator
# "mpa" refers to the MPtrj + sAlex channel, used for evaluating Matbench Discovery.
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')  # Use modal='omat24' for OMat24-trained channel weights.
```

When using the command-line interface of SevenNet, include the `--modal mpa` or `--modal omat24` option to select the desired modality.

### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|**0.845**|**0.901**|0.317| **0.064** |

[Detailed instructions for multi-fidelity learning](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md)

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/7net_MF_ompa/28590722?file=53029859)

---
## SevenNet-omat
> Model keywords: `7net-omat` | `SevenNet-omat`

 This model was trained exclusively on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset. It achieves high performance in $\kappa_{\mathrm{SRME}}$ on [Matbench Discovery](https://matbench-discovery.materialsproject.org/), but its F1 score is unavailable due to a difference in the POTCAR version. Like `SevenNet-MF-ompa`, this model outperforms `SevenNet-l3i5` on most tasks, except for the isolated molecule energy.

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/SevenNet_omat/28593938).

### **Matbench Discovery**
* $\kappa_{\mathrm{SRME}}$: **0.221**

---
## SevenNet-l3i5
> Model keywords: `7net-l3i5` | `SevenNet-l3i5`

This model increases the maximum spherical harmonic degree ($l_{\mathrm{max}}$) to 3, compared to `SevenNet-0`, which has an $l_{\mathrm{max}}$ of 2. While **l3i5** offers improved accuracy for various systems, it is approximately four times slower than `SevenNet-0`.

### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|0.714 |0.760|0.550|0.085|

---

## SevenNet-0
> Model keywords:: `7net-0` | `SevenNet-0` | `7net-0_11Jul2024` | `SevenNet-0_11Jul2024`

This model is one of our earliest pretrained models. Although we recommend using newer and more accurate models, it can still be useful in certain cases due to its shortest inference time. The model was trained on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and is loaded as the default pretrained model in the ASE calculator.
For more information, click [here](sevenn/pretrained_potentials/SevenNet_0__11Jul2024).

### **Matbench Discovery**
| F1 | $\kappa_{\mathrm{SRME}}$ |
|:---:|:---:|
|0.67|0.767|

---
You can find our legacy models in [pretrained_potentials](./sevenn/pretrained_potentials).
