import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def main():
    # 0. Initialize output directory
    fig_dir = 'fig'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print(f"Created directory: {fig_dir}")

    # 1. Dataset ingestion
    try:
        df = pd.read_csv('data/mobile_price.csv')
    except FileNotFoundError:
        print("Error: 'data/mobile_price.csv' not found.")
        return

    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # 2. 60/20/20 Train-Validation-Test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # 3. Feature scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Hyperparameter sweep for Regularization Parameter (C)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    results = {
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'train_f1': [], 'val_f1': [], 'test_f1': []
    }

    print("Training SVM models with different C values...")
    
    for c in C_values:
        svm_clf = SVC(C=c, random_state=42)
        svm_clf.fit(X_train_scaled, y_train)

        y_train_pred = svm_clf.predict(X_train_scaled)
        y_val_pred = svm_clf.predict(X_val_scaled)
        y_test_pred = svm_clf.predict(X_test_scaled)

        # Performance tracking
        results['train_acc'].append(accuracy_score(y_train, y_train_pred))
        results['val_acc'].append(accuracy_score(y_val, y_val_pred))
        results['test_acc'].append(accuracy_score(y_test, y_test_pred))

        results['train_f1'].append(f1_score(y_train, y_train_pred, average='weighted'))
        results['val_f1'].append(f1_score(y_val, y_val_pred, average='weighted'))
        results['test_f1'].append(f1_score(y_test, y_test_pred, average='weighted'))

    # Identify optimal C via validation metrics
    max_val_acc = max(results['val_acc'])
    best_c_acc = C_values[results['val_acc'].index(max_val_acc)]

    max_val_f1 = max(results['val_f1'])
    best_c_f1 = C_values[results['val_f1'].index(max_val_f1)]

    # 5. Visualization
    plt.figure(figsize=(14, 6))
    plt.suptitle('Impact of Regularization Parameter (C) on SVM Performance', fontsize=16, fontweight='bold', y=1.02)

    # Accuracy analysis
    plt.subplot(1, 2, 1)
    plt.plot(C_values, results['train_acc'], marker='o', label='Training')
    plt.plot(C_values, results['val_acc'], marker='s', label='Validation')
    plt.plot(C_values, results['test_acc'], marker='^', label='Testing')
    
    # Highlight optimal C threshold
    plt.axvline(x=best_c_acc, color='red', linestyle='--', alpha=0.6, label=f'Best C = {best_c_acc}')
    plt.annotate(f'Max Val: {max_val_acc:.4f}', 
                 xy=(best_c_acc, max_val_acc), 
                 xytext=(15, -20), 
                 textcoords='offset points', 
                 color='red', 
                 fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='red', alpha=0.7))

    plt.xscale('log')
    plt.xlabel('C value (Log Scale)')
    plt.ylabel('Accuracy')
    plt.title('SVM Accuracy vs. C')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Weighted F1-score analysis
    plt.subplot(1, 2, 2)
    plt.plot(C_values, results['train_f1'], marker='o', label='Training')
    plt.plot(C_values, results['val_f1'], marker='s', label='Validation')
    plt.plot(C_values, results['test_f1'], marker='^', label='Testing')
    
    plt.axvline(x=best_c_f1, color='red', linestyle='--', alpha=0.6, label=f'Best C = {best_c_f1}')
    plt.annotate(f'Max Val: {max_val_f1:.4f}', 
                 xy=(best_c_f1, max_val_f1), 
                 xytext=(15, -20), 
                 textcoords='offset points', 
                 color='red', 
                 fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color='red', alpha=0.7))

    plt.xscale('log')
    plt.xlabel('C value (Log Scale)')
    plt.ylabel('F1-score (Weighted)')
    plt.title('SVM F1-score vs. C')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    save_path = os.path.join(fig_dir, 'svm_c_parameter_tuning.png')
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close()
    
    print(f"Results successfully saved to: {save_path}")

if __name__ == "__main__":
    main()