import matplotlib
matplotlib.use('Agg')  # Suppress GUI backend for server-side plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# Import custom dependencies
from model.linear_model import LinearModel
from model.metrics import logloss
from model.gradients import logloss_sigmoid_grad
from model.activations import sigmoid

# =====================================================================
# 1. Configuration & Data Preprocessing
# =====================================================================
seed = 40
np.random.seed(seed)

df = pd.read_csv('data/NYCU_Iris.csv')
feature_cols = [col for col in df.columns if col != 'Species']

# Apply Min-Max Normalization
for col in feature_cols:
    col_min, col_max = df[col].min(), df[col].max()
    df[col] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0
        
X = df[feature_cols].values.astype(float)
y = df['Species'].values

# Perform train-test split (CV to be applied exclusively to training set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# =====================================================================
# 2. Scikit-learn API Wrapper
# =====================================================================
class CustomModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, dim, lr=0.1, reg_lambda=0.0):
        self.dim = dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.model = LinearModel(
            dim=self.dim, is_reg=False,
            loss_fn=logloss, act_fn=sigmoid, grad_fn=logloss_sigmoid_grad
        )

    def fit(self, X, y):
        # Implementation utilizes L2 regularization per specification
        self.model.fit(X, y, lr=self.lr, n_iteration=10000, val_ratio=0.0, 
                        reg_type='l2', reg_lambda=self.reg_lambda)
        return self

    def predict(self, X):
        return self.model.predict(X)

# =====================================================================
# 3. Hyperparameter Tuning via 5-Fold Cross-Validation
# =====================================================================
lrs = [0.005, 0.01, 0.1, 0.5]
lambdas = [1.0, 2.0, 4.0, 8.0]
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=seed)
results_matrix = np.zeros((len(lrs), len(lambdas)))

print("Executing 5-fold cross-validation for 16 hyperparameter combinations...")

for i, lr in enumerate(lrs):
    for j, reg in enumerate(lambdas):
        model_wrapper = CustomModelWrapper(dim=X_train.shape[1], lr=lr, reg_lambda=reg)
        scores = cross_val_score(model_wrapper, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        results_matrix[i, j] = scores.mean()
        plt.close('all') # Cleanup resources from internal plotting

# =====================================================================
# 4. Results Aggregation
# =====================================================================
results_df = pd.DataFrame(results_matrix, 
                          index=[f'LR={lr}' for lr in lrs], 
                          columns=[f'Lambda={reg}' for reg in lambdas])

print("\n--- Summary: Mean 5-Fold Cross-Validation Accuracy ---")
print(results_df)