# Efficient-Privacy-Preserving-Federated-Learning-With-Improved-Compressed-Sensing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8347455.svg)](https://doi.org/10.5281/zenodo.8347455)

This is the code of paper [Efficient-Privacy-Preserving-Federated-Learning-With-Improved-Compressed-Sensing](https://ieeexplore.ieee.org/document/10235260).


## Environments
With the installed conda, you can run this project in a conda virtual environment.
* **Python**: 3.7
* **Pytorch**: 1.13.1
* **[seal](https://github.com/Huelse/SEAL-Python)**: 4.0.0

## Usage

Note that the dataset needs to be prepared before running main.py
in folder "dataset"
```
  cd ./dataset
  python generate_mnist.py iid - - # for iid and unbalanced setting
  # python generate_mnist.py noniid - - # for pathological noniid setting
  # python generate_mnist.py noniid - dir # for practical noniid setting
  # python generate_mnist.py noniid - noise # for feature skew noniid setting
```

## Comparison Schemes

You can find the comparative studies in the repository below.

* **[PFL-Non-IID(baseline)](https://github.com/TsingZ0/PFL-Non-IID/tree/0af30fc8665e04ea9200b041f0c457c2260cbc99)**
* **[HeteroFL](https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients)**
* **[CEFL](https://github.com/AshwinRJ/Federated-Learning-PyTorch)**






## Citation
If you find this repository useful, please cite our paper:

```
@ARTICLE{10235260,
  author={Zhang, Yifan and Miao, Yinbin and Li, Xinghua and Wei, Linfeng and Liu, Zhiquan and Choo, Kim-Kwang Raymond and Deng, Robert H.},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Efficient Privacy-Preserving Federated Learning With Improved Compressed Sensing}, 
  year={2024},
  volume={20},
  number={3},
  pages={3316-3326},
  keywords={Computational modeling;Data models;Servers;Training;Privacy;Federated learning;Data privacy;CKKS;communication costs;compression sensing (CS);federated learning (FL);homomorphic encryption},
  doi={10.1109/TII.2023.3297596}}

```
