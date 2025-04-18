# Employee Attrition Prediction

This repository contains the implementation and dataset used in a undergraduate thesis project focused on modeling employee attrition using both survival analysis and classification techniques.
**Author**: Mengxiao Ma 
**Institution**: [Brunel University of London]  
**Brunel Number**: 2160910
**Year**: 2025

## Files

- `main.py` — Main script for data preprocessing, model training, evaluation, and visualization.
- `V1datasetAttrition.csv` — Input dataset containing employee demographic, job-related, and attrition information.

## Modeling Approach

The analysis combines traditional survival models with modern machine learning classifiers to predict both long-term risk and short-term attrition outcomes.

### 🔹 Survival Models
- Kaplan-Meier estimation
- Cox Proportional Hazards model (with stratification and interaction terms)
- Random Survival Forest (RSF)

### 🔹 Classification Models
- XGBoost
- LightGBM
- CatBoost
- Weighted ensemble model (LightGBM + CatBoost)

### 🔹 Evaluation Metrics
- **Survival models:** C-index, Integrated Brier Score (IBS)
- **Classification models:** AUC, F1-score, Accuracy
- **Feature importance:** Permutation-based and model-specific methods

## Key Techniques

- Automated encoding, feature engineering, and interaction term construction
- Hyperparameter tuning with early stopping
- Log-rank tests for categorical survival comparison
- Parallelized visualization using `joblib` and `seaborn`

## Environment

Developed with:
- Python 3.9
- pandas
- scikit-learn
- lifelines
- sksurv
- lightgbm
- catboost
- seaborn

## Note

This dataset is used for academic research purposes only.

## Citation

If you use this codebase or dataset for your own research or learning, please cite or refer to:

> Mengxiao Ma, "Hybrid Modeling of Employee Attrition using Survival Analysis and Machine Learning", Undergraduate Thesis, 2025.
