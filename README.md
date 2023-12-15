# TASFAR Domain Adaptation
This repo implements the paper [Target-agnostic Source-free Domain Adaptation for Regression Tasks](https://arxiv.org/abs/2312.00540) which is accepted by ICDE 2024. 

The following example generates pseudo label for [housing-price prediction](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
### Pseudo-label testing
```
# Environment setup
pip install scipy
pip install statsmodels
# Prepare data for pseudo-label generating
python col_ys.py  # You can ignore this step as we have provided the data.
# Generate pseudo labels for target data
python gen_pseudo_label.py
```
Evaluation
```
------------------------------------------------------------
Prediction error: 0.2421
Pseudo-label error: 0.2085
------------------------------------------------------------
```
### Training 
Training will further improve the model performance because of the credibility of the pseudo-label.
```
python train.py
python test.py
```
Evaluation
```
------------------------------------------------------------
Price MSE before adaptation: 0.2421
Price MSE after adaptation: 0.1734
MSE reduction rate: 28.38%
------------------------------------------------------------
