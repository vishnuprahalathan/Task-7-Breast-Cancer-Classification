# Task-7-Breast-Cancer-Classification

This project implements Support Vector Machines (SVM) to classify breast cancer tumors as malignant or benign using the popular Breast Cancer dataset.  
It includes data preprocessing, normalization, PCA visualization, model training with linear and RBF kernels, hyperparameter tuning using GridSearchCV, and performance evaluation.


 Dataset
- Source: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- Target: `diagnosis` column (`M` = Malignant, `B` = Benign)

---

 Tools & Libraries
- Python
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn

Objective

To build and evaluate robust SVM models (Linear and RBF) to classify breast cancer data and visualize decision boundaries using PCA.

---

Libraries Used

- `pandas`, `numpy`: Data handling
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: ML models, preprocessing, PCA, GridSearchCV

---

Dataset

- File: `breast-cancer.csv`
- Target Column: `diagnosis`
  - `M` → Malignant (1)
  - `B` → Benign (0)
- Note: Any `'id'` and `'?'` values are dropped or cleaned.

---

Workflow Summary

1. Data Cleaning
- Dropped `'id'` column.
- Replaced `'?'` with `NaN` and dropped missing rows.
- Converted target labels `M` and `B` into binary (1 and 0).

2. Feature Scaling
- Normalized all features using `StandardScaler`.

3. Dimensionality Reduction (PCA)
- Reduced features to 2 dimensions using `PCA` for visualization of decision boundaries.

4. Model Training
- Linear SVM: `kernel='linear'`
- RBF SVM: `kernel='rbf'`
- GridSearchCV for hyperparameter tuning:
  ```python
  param_grid = {
      'C': [0.1, 1, 10],
      'gamma': ['scale', 0.01, 0.1, 1],
      'kernel': ['rbf']
  }
5. Model Evaluation
Confusion matrix and classification report for:

Linear SVM

RBF SVM

Tuned RBF SVM

5-fold cross-validation accuracy for each model.

6. Visualization
Used PCA-reduced data to plot decision boundaries for each model.
