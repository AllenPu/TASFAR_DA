# TASFAR Domain Adaptation
This Repo implements the paper [Target-agnostic Source-free Domain Adaptation for Regression Tasks](https://arxiv.org/abs/2312.00540) which is accepted by ICDE 2024. 

The following example generates pseudo label for [housing-price prediction](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
### Pseudo-label Generation
```
# Environment setup
pip install scipy
pip install statsmodels
# Prepare data for pseudo-label generating
python col_ys.py  # For housing-price prediction, data have been given in ./data/
# Generate pseudo labels for target data
python gen_pseudo_label.py
```
Evaluation
```
------------------------------------------------------------
Prediction MSE: 0.2421
Pseudo-label MSE: 0.2085
------------------------------------------------------------
```
### Adaptation Training 
```
python train.py # Training using pseudo labels, weighted by pseudo-label credibility
python test.py
```
Testing result
```
------------------------------------------------------------
Price MSE before adaptation: 0.2421
Price MSE after adaptation: 0.1734
MSE reduction rate: 28.38%
------------------------------------------------------------
