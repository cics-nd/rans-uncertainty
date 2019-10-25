# Training Flows

[Quantifying model form uncertainty in Reynolds-averaged turbulence models with Bayesian deep neural networks](https://www.sciencedirect.com/science/article/pii/S0021999119300464)

[Nicholas Geneva](http://nicholasgeneva.com/), [Nicholas Zabaras](https://www.zabaras.com)

We have provided the set of training flow data. Additionally for each a pre-processing script is included for spacial averaging which helps reduce noise as well as elminating any artificial secondary flows. The LES data has been sub-sampled onto the RANS mesh. OpenFOAM files have also been included for viewing each flow in ParaView.

**We highly suggest generating your own training data as *consistency* with models, boundary conditions and other flow parameters is essential! For this reason we have provided meshes that can be used as training/testing flows.**

## Citation
Find this useful? Cite us with:
```latex
@article{geneva2019quantifying,
  title = {Quantifying model form uncertainty in {Reynolds-averaged} turbulence models with {Bayesian} deep neural networks},
  journal = {Journal of Computational Physics},
  volume = {383},
  pages = {125 - 147},
  year = {2019},
  issn = {0021-9991},
  doi = {10.1016/j.jcp.2019.01.021},
  url = {http://www.sciencedirect.com/science/article/pii/S0021999119300464},
  author = {Nicholas Geneva and Nicholas Zabaras}
}
```
