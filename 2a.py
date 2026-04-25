import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def main():
    # Load dataset
    try:
        df = pd.read_csv('data/mobile_price.csv')
    except FileNotFoundError:
        print("Error: 'data/mobile_price.csv' not found.")
        return

    # Feature and target separation
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # 60/20/20 Train-Validation-Test split
    # Split 20% for testing
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split remaining 80% into 75% training (60% total) and 25% validation (20% total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # SVM configuration (C=1.0) and training
    svm_clf = SVC(C=1.0, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # Model evaluation
    datasets = {
        "Training": (X_train_scaled, y_train),
        "Validation": (X_val_scaled, y_val),
        "Testing": (X_test_scaled, y_test)
    }

    print("--- SVM Model Evaluation (C=1.0) ---")
    for name, (X_data, y_true) in datasets.items():
        y_pred = svm_clf.predict(X_data)
        acc = accuracy_score(y_true, y_pred)
        # Weighted F1-score for multi-class classification
        f1 = f1_score(y_true, y_pred, average='weighted') 
        
        print(f"{name} Data:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()