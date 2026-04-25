import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.linear_model import LinearModel
from model.metrics import logloss, evaluate_binary_classifier
from model.gradients import logloss_sigmoid_grad
from model.activations import sigmoid

# --- Configuration & Data Preprocessing ---
seed = 40
np.random.seed(seed)

df = pd.read_csv('data/NYCU_Iris.csv')
feature_cols = [col for col in df.columns if col != 'Species']

# Min-Max Normalization
for col in feature_cols:
    col_min, col_max = df[col].min(), df[col].max()
    df[col] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0
        
X = df[feature_cols].values.astype(float)
y = df['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# --- Optimized Hyperparameters ---
top_settings = [
    {'lr': 0.005, 'reg_lambda': 2.0, 'label': 'Top 1 (LR=0.005, L2=2.0)'},
    {'lr': 0.01,  'reg_lambda': 1.0, 'label': 'Top 2 (LR=0.01,  L2=1.0)'}
]

# --- Model Training & Test Set Evaluation ---
for config in top_settings:
    print(f"\n{'='*50}\nEvaluating: {config['label']}\n{'='*50}")
    
    model = LinearModel(
        dim=X_train.shape[1], 
        is_reg=False,
        loss_fn=logloss, 
        act_fn=sigmoid, 
        grad_fn=logloss_sigmoid_grad
    )
    
    # Train on full training set
    model.fit(
        X_train, y_train, 
        lr=config['lr'], 
        n_iteration=10000, 
        val_ratio=0.0, 
        reg_type='l2', 
        reg_lambda=config['reg_lambda']
    )
    
    # Inference and Performance Metrics
    y_pred = model.predict(X_test)
    evaluate_binary_classifier(
        y_test, y_pred, 
        title=f"Test Results - {config['label']}"
    )