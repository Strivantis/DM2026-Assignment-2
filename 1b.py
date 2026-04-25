import matplotlib
matplotlib.use('Agg')  # Suppress GUI backend for server-side plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 匯入助教寫好的自定義模組
from model.linear_model import LinearModel
from model.metrics import logloss, evaluate_binary_classifier
from model.gradients import logloss_sigmoid_grad
from model.activations import sigmoid

# =====================================================================
# 1. 資料前處理 (與 1a 及 TA 原始設定完全相同)
# =====================================================================
seed = 40
np.random.seed(seed)

print("載入資料與進行 Min-Max Normalization...")
df = pd.read_csv('data/NYCU_Iris.csv')
feature_cols = [col for col in df.columns if col != 'Species']

for col in feature_cols:
    col_min, col_max = df[col].min(), df[col].max()
    if col_max > col_min:
        df[col] = (df[col] - col_min) / (col_max - col_min)
    else:
        df[col] = 0.0
        
X = df[feature_cols].values.astype(float)
y = df['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# =====================================================================
# 2. 定義從 1(a) 找出的 Top 2 超參數
# =====================================================================
top_settings = [
    {'rank': 1, 'lr': 0.005, 'reg_lambda': 2.0, 'label': 'Top 1 (LR=0.005, Lambda=2.0)'},
    {'rank': 2, 'lr': 0.01,  'reg_lambda': 1.0, 'label': 'Top 2 (LR=0.01,  Lambda=1.0)'}
]

# =====================================================================
# 3. 訓練模型並使用 Testing Data 進行驗證
# =====================================================================
print("\n開始進行 1(b) Testing Data 評估...\n")

for setting in top_settings:
    print(f"==================================================")
    print(f" Evaluating: {setting['label']} ")
    print(f"==================================================")
    
    # 初始化助教的 LinearModel
    model = LinearModel(
        dim=X_train.shape[1], 
        is_reg=False,
        loss_fn=logloss, 
        act_fn=sigmoid, 
        grad_fn=logloss_sigmoid_grad
    )
    
    # 步驟一：使用訓練集 (Training Data) 訓練模型
    # (我們將 val_ratio 設為 0.0，因為現在要直接拿全部訓練集來 fit，然後去測 test_set)
    model.fit(
        X_train, y_train, 
        lr=setting['lr'], 
        n_iteration=10000, 
        val_ratio=0.0, 
        reg_type='l2', 
        reg_lambda=setting['reg_lambda']
    )
    
    # 步驟二：使用測試集 (Testing Data) 進行預測
    y_pred = model.predict(X_test)
    
    # 步驟三：呼叫助教的評估函數，印出四項指標
    evaluate_binary_classifier(
        y_test, y_pred, 
        title=f"Testing Results - {setting['label']}"
    )
    print("\n")