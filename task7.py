

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv('C:\\Users\\Vishnu Prahalathan\\Desktop\\breast-cancer.csv')
df.drop(['id'], axis=1, inplace=True, errors='ignore')
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)


df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)


svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters (RBF):", grid.best_params_)
print("Best Estimator:\n", grid.best_estimator_)


models = {'Linear SVM': svm_linear, 'RBF SVM': svm_rbf, 'Tuned RBF SVM': grid.best_estimator_}
for name, model in models.items():
    print(f"\n{name} Evaluation:")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation Accuracy: {np.mean(scores):.4f}")

def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

for name, model in models.items():
    model.fit(X_pca, y)
    plot_decision_boundary(model, X_pca, y, f"{name} Decision Boundary (PCA Reduced)")

