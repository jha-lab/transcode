# TransCODE: Co-design of Transformers and Accelerators for Efficient Training and Inference

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

This repository contains the simulation code for the paper "TransCODE: Co-design of Transformers and Accelerators for Efficient Training and Inference" published at the IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository and initialize sub-modules](#clone-this-repository-and-initialize-sub-modules)
  - [Setup python environment](#setup-python-environment)
- [Run DynaProp](#run-dynaprop)
- [Run Co-design](#run-co-design)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

### Clone this repository and initialize sub-modules

```shell
git clone https://github.com/JHA-Lab/transcode.git
cd ./transcode/
git submodule init
git submodule update
```

### Setup python environment  

To setup python environment, please look at the instruction in the [txf_design-space](https://github.com/JHA-Lab/txf_design-space) and the [acceltran](https://github.com/JHA-Lab/acceltran) repositories.

## Run DynaProp

To run evaluation of DynaProp when training transformer models, run the following command:
```shell
cd ./dynaprop/
python run_evaluation.py --max_evaluation_threshold <tau_I> --max_train_threshold <tau_T>
cd ..
```
Here, `<tau_I` and `<tau_T>` are the evaluation and training pruning thresholds. For more information on the possible inputs to the simulation script, use:
```shell
cd ./dynaprop/
python3 run_evaluation.py --help
cd ..
```

## Run Co-design

To run hardware-software co-design over the AccelTran and FlexiBERT 2.0 design spaces, use the following command:
```shell
cd ./co-design/
python run_co-design.py
cd ..
```
For more information on the possible inputs to the co-design script, use:
```shell
cd ./co-design/
python3 run_co-design.py --help
cd ..
```

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our previous works that define the hardware (AccelTran) and software (FlexiBERT) design spaces, using the following bitex entry:
```bibtex
@article{tuli2023acceltran,
  author={Tuli, Shikhar and Jha, Niraj K.},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={AccelTran: A Sparsity-Aware Accelerator for Dynamic Inference with Transformers}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCAD.2023.3273992}}
```
```bibtex
@article{tuli2023flexibert, 
  author = {Tuli, Shikhar and Dedhia, Bhishma and Tuli, Shreshth and Jha, Niraj K.}, 
  title = {{FlexiBERT}: Are Current Transformer Architectures Too Homogeneous and Rigid?}, 
  year = {2023},
  volume = {77}, 
  doi = {10.1613/jair.1.13942}, 
  journal = {Journal of Artificial Intelligence Reseasrch},
  numpages = {32}
}
```
If you use the provided co-design scripts, please cite our paper:
```bibtex
@article{tuli2023transcode,
  title={{TransCODE}: Co-design of Transformers and Accelerators for Efficient Training and Inference},
  author={Tuli, Shikhar and Jha, Niraj K},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  year={2023}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
