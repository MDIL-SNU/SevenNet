# Pretrained models

So far, we have released multiple pretrained SevenNet models. Each model has various hyperparameters and training sets, leading to different levels of accuracy and speed. Please read the descriptions below carefully and choose the model that best suits your purpose.
We provide the F1 score, and RMSD for the WBM dataset, along with $\kappa_{\mathrm{SRME}}$ from phononDB and CPS (Combined Performance Score). For details on these metrics and performance comparisons with other pretrained models, please visit [Matbench Discovery](https://matbench-discovery.materialsproject.org/).

These models can be used as interatomic potentials in LAMMPS and loaded through the ASE calculator using each model’s keywords. Please refer to the [ASE calculator](#ase_calculator) section for instructions on loading a model via the ASE calculator.
Additionally, `keywords` can be used in other parts of SevenNet, such as `sevenn_inference`, `sevenn_get_model`, and the `checkpoint` section in `input.yaml` for fine-tuning.

**Acknowledgments**: The models trained on [`MPtrj`](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) were supported by the Neural Processing Research Center program at Samsung Advanced Institute of Technology, part of Samsung Electronics Co., Ltd. The computations for training models were carried out using the Samsung SSC-21 cluster.

---

### SevenNet-MF-ompa
> Model keywords: `7net-mf-ompa` | `SevenNet-mf-ompa`

**This is our recommended pretrained model**

This model leverages [multi-fidelity learning](https://pubs.acs.org/doi/10.1021/jacs.4c14455) to train simultaneously on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842), [sAlex](https://huggingface.co/datasets/fairchem/OMAT24), and [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) datasets. This model is the best among our pretrained models and achieves a high ranking on the [Matbench Discovery]((https://matbench-discovery.materialsproject.org/)) leaderboard. Our evaluations show that it outperforms other models on most tasks, except for the isolated molecule energy task, where it performs slightly worse than `SevenNet-l3i5`.

```python
from sevenn.calculator import SevenNetCalculator
# "mpa" refers to the MPtrj + sAlex modal, used for evaluating Matbench Discovery.
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')  # Use modal='omat24' for OMat24-trained modal weights.
```
> [!NOTE]
> Each modal is expected to produce results that are more consistent with the DFT settings in the training datasets (e.g., `mpa`, trained on the combined [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and [sAlex](https://huggingface.co/datasets/fairchem/OMAT24) datasets; `omat24`, trained on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset). For detailed DFT settings, please refer to their papers.

When using the command-line interface of SevenNet, include the `--modal mpa` or `--modal omat24` option to select the desired modality.

#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|**0.845**|**0.901**|0.317| **0.064** |

[Detailed instructions for multi-fidelity learning](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_MF_0/README.md)

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/7net_MF_ompa/28590722?file=53029859)

---
### SevenNet-omat
> Model keywords: `7net-omat` | `SevenNet-omat`

 This model was trained exclusively on the [OMat24](https://huggingface.co/datasets/fairchem/OMAT24) dataset. It achieves high performance in $\kappa_{\mathrm{SRME}}$ on [Matbench Discovery](https://matbench-discovery.materialsproject.org/), but its F1 score is unavailable due to a difference in the POTCAR version. Like `SevenNet-MF-ompa`, this model outperforms `SevenNet-l3i5` on most tasks, except for the isolated molecule energy.

[Download link for fully detailed checkpoint](https://figshare.com/articles/software/SevenNet_omat/28593938).

#### **Matbench Discovery**
* $\kappa_{\mathrm{SRME}}$: **0.221**

---
### SevenNet-l3i5
> Model keywords: `7net-l3i5` | `SevenNet-l3i5`

This model increases the maximum spherical harmonic degree ($l_{\mathrm{max}}$) to 3, compared to `SevenNet-0`, which has an $l_{\mathrm{max}}$ of 2. While **l3i5** offers improved accuracy for various systems, it is approximately four times slower than `SevenNet-0`.

#### **Matbench Discovery**
| CPS  | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD |
|:---:|:---:|:---:|:---:|
|0.714 |0.760|0.550|0.085|

---

### SevenNet-0
> Model keywords:: `7net-0` | `SevenNet-0` | `7net-0_11Jul2024` | `SevenNet-0_11Jul2024`

This model is one of our earliest pretrained models. Although we recommend using newer and more accurate models, it can still be useful in certain cases due to its shortest inference time. The model was trained on the [MPtrj](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842) and is loaded as the default pretrained model in the ASE calculator.
For more information, click [here](sevenn/pretrained_potentials/SevenNet_0__11Jul2024).

#### **Matbench Discovery**
| F1 | $\kappa_{\mathrm{SRME}}$ |
|:---:|:---:|
|0.67|0.767|

---
You can find our legacy models in [pretrained_potentials](./sevenn/pretrained_potentials).


