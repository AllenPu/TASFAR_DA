# TASFAR Domain Adaptation
This repo implements the paper [Target-agnostic Source-free Domain Adaptation for Regression Tasks](https://arxiv.org/abs/2312.00540) which is accepted by ICDE 2024. 

The following example generates pseudo label for [housing-price prediction](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
```
# Environment setup
pip install scipy
pip install statsmodels
# Generate pseudo labels for target data
python gen_pseudo_label.py
```
Evaluation
