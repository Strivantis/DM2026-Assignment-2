# NYCU Data Mining (Spring 2026) Assignment 2

### Prerequisites
Ensure the following Python packages are installed before execution:
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `mlxtend` (for association rule mining)
* `scipy`

---

## File Structure

```text
.
├── 1a.py               # Task 1: Logistic Regression hyperparameter tuning (5-Fold CV)
├── 1b.py               # Task 1: Optimal Logistic Regression model evaluation
├── 2a.py               # Task 2: SVM basic classification and evaluation
├── 2b.py               # Task 2: Performance analysis and visualization of SVM regularization parameter C
├── 3.py                # Task 3: Frequent itemset mining and association rules via FP-growth
├── 4.py                # Task 4: PCA dimensionality reduction and K-Means clustering analysis
├── 5.py                # Task 5: Enhanced K-Means clustering with CAR-GFW
├── README.md
├── data/               # Dataset directory
│   ├── NYCU_Iris.csv            
│   └── mobile_price.csv         
├── fig/                # Generated visualizations and plots
└── model/              # Provided model implementations and modules
    ├── activations.py           
    ├── gradients.py             
    ├── linear_model.py          
    ├── metrics.py               
    └── utils.py                 
```