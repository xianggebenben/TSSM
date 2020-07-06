

# TSSM: Distributed Deep Learning  via Tunable Subnetwork Splitting Method
This is a  simple demo of Tunable Subnetwork Splitting Method (TSSM) as described in our paper:

Junxiang Wang, Zheng Chai, Yue Cheng, and Liang Zhao. [Tunable Subnetwork Splitting for Model-parallelism of Neural Network Training.](https://www.researchgate.net/publication/342542966_Tunable_Subnetwork_Splitting_for_Model-parallelism_of_Neural_Network_Training) 
(ICML 2020 Workshop: Beyond First Order Methods in Machine Learning)

## Installation

python setup.py install

## Requirements

tensorflow 2.1.0

numpy 1.18.1

scipy 1.4.1

keras 2.3.1

tensorflow_datasets 1.3.2

## Run the Demo

python main.py

## Data

Three benchmark datasets MNIST, Fashion-MNIST, kMNIST are included in this package.

## Cite

Please cite our paper if you use this code in your own work:

@inproceedings{wang2020,

author = {Wang, Junxiang and Zheng Chai and Yue Chen and Zhao, Liang},

title = {Tunable Subnetwork Splitting for Model-parallelism of Neural Network Training},

year = {2020},

booktitle = {ICML 2020 Workshop: Beyond First Order Methods in Machine Learning},

}
