# Local Steps in Differentially Private Federated Learning: Middle is More

Differential Privacy (DP) enhance the privacy performance of Federated Learning (FL). In Federated Learning with Differential Privacy (DPFL), a primary challenge is balancing the number of local training steps with the consumption of the privacy budget. On one hand, an insufficient number of local steps may result in aggregating locally trained models that are not fully optimized, thereby consuming the privacy budget and potentially impeding the convergence of the global model before the budget is exhausted. On the other hand, an excessive number of local steps increases the computational burden on clients and heightens susceptibility to client heterogeneity, which can hinder the convergence of the global model. Previous studies typically treat the number of local steps as a hyperparameter, usually assigning a fixed value (such as 1 or 5), without thoroughly investigating its impact on privacy budget consumption. In this study, we systematically analyze the relationship between the number of local steps and the consumption of the privacy budget. Through rigorous convergence analysis, we identify the existence of an optimal number of local steps that minimizes privacy budget consumption while ensuring model convergence. Our theoretical findings are extensively validated through comprehensive experiments conducted on three datasets.

## Anonymous

Due to submission restrictions, the father-repository authors' information and paper authors' information will be added to this repository after the paper is accepted.

## Contribution

This is a repository focus on Client-Level Differential Privacy.

## Environment

Our experiments are performed on a server equipped with an Intel I9 13900ks CPU (24 cores), 64GB RAM, and an NVIDIA 4090 GPU, using PyTorch-1.8 on Windows 10. 

We advise use the CUDA to recurrent our experiment.

`requirement.txt` had freeze in project which can quickly equip environment by using

```
pip install -r requirements.txt
```

## Run

First check the dataset had been processed in `./dataset/generate_DATESETNAME.py`

then run the `run.sh` in `./system/run.sh`

