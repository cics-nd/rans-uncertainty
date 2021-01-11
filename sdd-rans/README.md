# Deep Learning in OpenFOAM

[Quantifying model form uncertainty in Reynolds-averaged turbulence models with Bayesian deep neural networks](https://www.sciencedirect.com/science/article/pii/S0021999119300464)

[Nicholas Geneva](http://nicholasgeneva.com/), [Nicholas Zabaras](https://www.zabaras.com)

This repository contains programs related to the implementation of Stochastic Data-Diven RANS (SDD-RANS). Due to the constant changes with needed libraries, the following does not aim to be a functioning package but rather a reference for others to understand our approach for incorporating deep learning into OpenFOAM. In this application we use Caffe2's C++ backend to directly integrate deep neural networks into OpenFOAM. This requires the user to have sufficient knowledge on linking C++ compiler libraries, wmake and OpenFOAM's general structure.

**Note**: Due to Caffe 2 updates (now being obsolete completely), compiling the Reynolds Net library may be difficult. Please see some of the closed issues for some additional information regarding [compatibility updates](https://github.com/cics-nd/rans-uncertainty/issues/11), [potential dependencies](https://github.com/cics-nd/rans-uncertainty/issues/10) and [compiling order](https://github.com/cics-nd/rans-uncertainty/issues/12). But such updates will never be official supported.

## Contents
* `solvers/simpleGradFoam` - A slightly modified version of the incompressible solver simpleFoam used for obtaining the baseline RANS data. This solver additionally outputs the scaled rate-of-strain and symmetric tensors needed for the neural network as well as a few additional flow features.
* `pytorchToCaffe2.py` -  A script for converting PyTorch neural networks into Caffe2's protobuf format to be read in C++.
* `TurbulenceModels/turbulenceModels/ReynoldsNet` - An OpenFOAM library used for loading and executing neural networks using Caffe2's mysterious C++ API. This approach avoids explicitly programming neural network architectures in C++, allowing for rapid testing and modification of neural networks used.
* `TurbulenceModels/incompressible` - The ReynoldsNet library is then called by a modified incompressible turbulence model library with an additional *divDevRhoReff(U, S, R)* function that does the data-driven prediction. This function can be found in linearViscousStress class.
* `solvers/simpleNNFoam` -  A slightly modified version of the incompressible solver simpleFoam used to solve the constrained R-S RANS equations yielding flow field predictions. Note that this solver is ran after an initial baseline simulation using *simpleGradFoam* for a particular test flow since converged RANS flow features are needed for model predictions.

Stochastic Data-Driven RANS Framework |
| ------------- |
![](../images/sdd-rans/sdd-rans-framework.gif)|

> The OpenFOAM files included are only the modified ones. For compiling one needs the remainder of turbulent model files required for the incompressible turbulence model.

## Dependencies
* Python 3.6.5
* [PyTorch](https://pytorch.org/) 1.0.0
* [ONNX](https://onnx.ai/)
* [Caffe 2](https://caffe2.ai/)
* [OpenFOAM](https://www.openfoam.com/) 4.1

*Listed versions were used for development. Compatability with newer versions is likely but not guaranteed.*

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
