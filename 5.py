import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from mlxtend.frequent_patterns import apriori, association_rules

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['price_range'])
    y = df['price_range']
    return df, X, y

def map_clusters_to_labels(y_true, y_pred):
    """
    Solve label switching problem using the Hungarian algorithm.
    """
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col_ind[i]: i for i in range(4)}
    return np.array([mapping[p] for p in y_pred])

def evaluate_kmeans(X, y_true, seeds, title=""):
    metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    for seed in seeds:
        kmeans = KMeans(n_clusters=4, random_state=seed, n_init='auto')
        y_pred = kmeans.fit_predict(X)
        y_pred_mapped = map_clusters_to_labels(y_true, y_pred)
        
        metrics['acc'].append(accuracy_score(y_true, y_pred_mapped))
        metrics['prec'].append(precision_score(y_true, y_pred_mapped, average='macro', zero_division=0))
        metrics['rec'].append(recall_score(y_true, y_pred_mapped, average='macro', zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred_mapped, average='macro', zero_division=0))
        
    print(f"--- {title} ---")
    print(f"Avg Accuracy : {np.mean(metrics['acc']):.4f}")
    print(f"Avg Precision: {np.mean(metrics['prec']):.4f}")
    print(f"Avg Recall   : {np.mean(metrics['rec']):.4f}")
    print(f"Avg F1-score : {np.mean(metrics['f1']):.4f}\n")

def apply_car_feature_weighting(df, X, X_scaled):
    """
    CAR-GFW Framework: Guided Feature Weighting via Class Association Rules.
    """
    print("Executing CAR-GFW (Mining target-related rules)...")
    
    # 1. Feature discretization
    df_binned = pd.DataFrame()
    for col in X.columns:
        if X[col].nunique() <= 2:
            df_binned[col] = X[col].astype(str)
        else:
            bins = pd.qcut(X[col], q=3, duplicates='drop')
            df_binned[col] = bins.astype(str)
            
    # 2. Append target labels for CAR mining
    df_binned['target_price'] = 'class_' + df['price_range'].astype(str)
    df_dummy = pd.get_dummies(df_binned)
    
    # 3. Frequent itemset mining (Apriori)
    frequent_itemsets = apriori(df_dummy.astype(bool), min_support=0.05, use_colnames=True)
    
    # 4. Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    
    # 5. Filter for Class Association Rules (CARs)
    def is_price_target(consequents):
        return any(item.startswith('target_price_') for item in consequents)
        
    car_rules = rules[rules['consequents'].apply(is_price_target)]
    
    # 6. Score feature importance (Confidence * Support)
    feature_weights = {col: 0.0 for col in X.columns}
    for idx, row in car_rules.iterrows():
        antecedents = row['antecedents']
        weight_score = row['confidence'] * row['support'] 
        for item in antecedents:
            for col in X.columns:
                if item.startswith(col + '_'):
                    feature_weights[col] += weight_score
                    break

    weights_array = np.array([feature_weights[col] for col in X.columns])
    
    # Exception handling: fallback to uniform weights
    if weights_array.max() == weights_array.min():
        print("Warning: No valid CARs found. Returning original features.")
        return X_scaled
    
    # 7. Normalize weights (Scale: 1.0x to 4.0x)
    weights_scaled = 1.0 + 3.0 * (weights_array - weights_array.min()) / (weights_array.max() - weights_array.min())
    
    print("\nTop 5 CAR-derived feature weights:")
    sorted_weights = sorted(zip(X.columns, weights_scaled), key=lambda x: x[1], reverse=True)
    for col, w in sorted_weights[:5]:
        print(f"- {col}: {w:.2f}x")
    print("")
    
    # 8. Apply weights to standardized matrix
    return X_scaled * weights_scaled

if __name__ == "__main__":
    filepath = 'data/mobile_price.csv'
    seeds = [0, 10, 42, 100, 999]
    
    df, X, y = load_data(filepath)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    evaluate_kmeans(X_scaled, y, seeds, title="Baseline K-Means")
    
    # Apply CAR-GFW enhancement
    X_weighted = apply_car_feature_weighting(df, X, X_scaled)
    evaluate_kmeans(X_weighted, y, seeds, title="CAR-GFW Enhanced K-Means")