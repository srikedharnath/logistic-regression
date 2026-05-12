# logistic-regression
# Insurance Charges Prediction using SVR

## Project Overview

This project predicts medical insurance charges using Support Vector Regression (SVR).

The model is trained using features like:
- age
- sex
- bmi
- children
- smoker
- region

and predicts:
- insurance charges

---

# Machine Learning Algorithm Used

- Support Vector Regression (SVR)

---

# Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit

---

# Dataset Information

Dataset contains the following columns:

| Column | Description |
|---|---|
| age | Age of person |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Residential area |
| charges | Medical insurance charges |

---

# Project Workflow

1. Load Dataset
2. Understand Dataset
3. Check Null Values
4. Encode Categorical Data
5. Feature Scaling
6. Train-Test Split
7. Train SVR Model
8. Prediction
9. Model Evaluation

---

# Hyperparameters Used in SVR

```python
SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)
