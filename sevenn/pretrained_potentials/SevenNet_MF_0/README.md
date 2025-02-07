### **SevenNet-MF-0 (16Dec2024)**
> Keywords in ASE: `7net-MF-0` and `SevenNet-MF-0`

The model is trained on PBE (+U) and $\mathrm{r}^{2}$ SCAN database provided in Materials Project.
It has the same architecture with **SevenNet-0 (11Jul2024)**, except this model contains additional 'fidelity-dependent' parameters utilized for multi-fidelity training.
However, overhead of calculations regarding fidelity-dependent parameters are negligible, which results in almost the same inference speed with **SevenNet-0 (11Jul2024)**.

Details in using this model as well as choosing level-of-theory for inference can be found in [here](./sevenn/pretrained_potentials/SevenNet_MF_0).

* Training set MAE ($\mathrm{r}^{2}$ SCAN): 10.8 meV/atom (energy), 0.018 eV/Ang. (force), and 0.58 kbar (stress)

SevenNet-MF-0 is a pretrained model trained on PBE (+U) and $\mathrm{r}^{2}$ SCAN crystal relaxation trajectories provided in Materials Project.
For more detailed explanation of training set curation, please refer to [ref](https://arxiv.org/abs/2409.07947).

To select the level-of-theory between PBE (+U) and $\mathrm{r}^{2}$ SCAN when using this model, following instructions can be used.
1. Using ASE calculator

Additional argument `modal` can be passed to `SevenNetCalculator` to specify the target level-of-theory.
```python
from sevenn.sevennet_calculator import SevenNetCalculator
r2scan_calculator = SevenNetCalculator(model='7net-MF-0', device='cpu', modal='R2SCAN')
pbe_calculator = SevenNetCalculator(model='7net-MF-0', device='cpu', modal='PBE')
```

2. Deploy model for LAMMPS

Additional argument `--modal` would specify the target level-of-theory for deployment.
```bash
sevenn_get_model 7net-MF-0 --modal R2SCAN
sevenn_get_model 7net-MF-0 --modal PBE
```

3. Inference mode from the checkpoint file

Additional argument `--modal` would specify the target level-of-theory for inference.
```bash
sevenn_inference 7net-MF-0 r2scan_graph.pt --modal R2SCAN
sevenn_inference 7net-MF-0 pbe_graph.pt --modal PBE
```

If you utilize the multi-fidelity feature of this code or the pretrained model SevenNet-MF-0, please cite the following paper:
```txt
@article{kim_sevennet_mf_2024,
	title = {Data-Efficient Multifidelity Training for High-Fidelity Machine Learning Interatomic Potentials},
	volume = {xx},
	doi = {10.1021/jacs.4c14455},
	number = {xx},
	journal = {J. Am. Chem. Soc.},
	author = {Kim, Jaesun and Kim, Jisu and Kim, Jaehoon and Lee, Jiho and Park, Yutack and Kang, Youngho and Han, Seungwu},
	year = {2024},
	pages = {xx--xx},
```
