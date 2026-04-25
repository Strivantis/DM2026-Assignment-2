import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def load_and_preprocess_data(filepath):
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None, None

    # Feature/Target separation
    X = df.drop(columns=['price_range'])
    y = df['price_range']
    
    print(f"Split complete. X: {X.shape}, y: {y.shape}")

    # Standardize features (μ=0, σ=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns

def apply_pca_and_visualize(X_scaled, y, output_dir):
    """
    Apply PCA to project features onto 2D and save the scatter plot to 'fig/' directory.
    """
    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA complete. New shape: {X_pca.shape}")
    
    # Plotting the 2D scatter plot
    plt.figure(figsize=(8, 6))
    
    classes = sorted(y.unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for cls, color in zip(classes, colors):
        idx = (y == cls)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                    c=color, label=f'Price Range {cls}', 
                    alpha=0.6, edgecolors='w', linewidth=0.5)
        
    plt.title('PCA of Mobile Price Dataset (First 2 Components)')
    plt.xlabel('First Principal Component (PC1)')
    plt.ylabel('Second Principal Component (PC2)')
    plt.legend(title="Class Labels")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    save_path = os.path.join(output_dir, 'pca_scatterplot.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {save_path}")
    
    return X_pca

def apply_kmeans_and_evaluate(X_train, y_true, X_viz, title, filename, output_dir):
    """
    Apply K-Means clustering, calculate ARI, and visualize on 2D PCA space.
    - X_train: Data used for clustering (could be all features or PCA features).
    - X_viz: 2D Data used for visualization (always X_pca).
    """
    # Initialize and fit K-Means (n_init='auto' suppresses sklearn warnings)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_train)
    
    # Calculate clustering performance
    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f"Adjusted Rand Score (ARI): {ari:.4f}")
    
    # Plotting the 2D scatter plot
    plt.figure(figsize=(8, 6))
    
    clusters = sorted(list(set(cluster_labels)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for cluster, color in zip(clusters, colors):
        idx = (cluster_labels == cluster)
        plt.scatter(X_viz[idx, 0], X_viz[idx, 1], 
                    c=color, label=f'Cluster {cluster}', 
                    alpha=0.6, edgecolors='w', linewidth=0.5)
        
    plt.title(f'{title}\nAdjusted Rand Score (ARI): {ari:.4f}')
    plt.xlabel('First Principal Component (PC1)')
    plt.ylabel('Second Principal Component (PC2)')
    plt.legend(title="K-Means Clusters")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {save_path}")


if __name__ == "__main__":
    data_path = 'data/mobile_price.csv'
    output_dir = 'fig'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # ==========================================
    print("\n" + "="*40)
    print("--- Task 4a: Data Preprocessing ---")
    print("="*40)
    X_scaled, y, feature_names = load_and_preprocess_data(data_path)
    
    if X_scaled is not None:
        # Preview first 3 standardized records
        print("\nPreview (X_scaled):")
        print(pd.DataFrame(X_scaled[:3], columns=feature_names))
        
        # ==========================================
        print("\n" + "="*40)
        print("--- Task 4b: PCA and Visualization ---")
        print("="*40)
        X_pca = apply_pca_and_visualize(X_scaled, y, output_dir)
        
        # ==========================================
        print("\n" + "="*40)
        print("--- Task 4c: K-Means on All Features ---")
        print("="*40)
        # Note: We cluster using X_scaled (all features), but visualize using X_pca (2D)
        apply_kmeans_and_evaluate(X_train=X_scaled, 
                                  y_true=y, 
                                  X_viz=X_pca, 
                                  title='K-Means on All Features (Projected to PCA Space)', 
                                  filename='kmeans_all_features.png', 
                                  output_dir=output_dir)
        
        # ==========================================
        print("\n" + "="*40)
        print("--- Task 4d: K-Means on PCA Features ---")
        print("="*40)
        # Note: We cluster using X_pca (2D features) and visualize using X_pca (2D)
        apply_kmeans_and_evaluate(X_train=X_pca, 
                                  y_true=y, 
                                  X_viz=X_pca, 
                                  title='K-Means on PCA Features', 
                                  filename='kmeans_pca_features.png', 
                                  output_dir=output_dir)
        print("\nAll tasks completed successfully!")