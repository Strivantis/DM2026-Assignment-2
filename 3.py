import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def main():
    # ==========================================
    # Task 3(a): 資料前處理與 FP-growth
    # ==========================================
    # 1. 讀取並過濾資料
    df = pd.read_csv('data/mobile_price.csv')
    df_filtered = df[df['price_range'] == 1].copy()

    # 2. 計算特徵邊界 (3:4:3)
    features = ['ram', 'int_memory', 'px_width', 'battery_power']
    boundaries = {}
    for col in features:
        min_val, max_val = df_filtered[col].min(), df_filtered[col].max()
        rng = max_val - min_val
        boundaries[col] = (min_val + rng * 0.3, min_val + rng * 0.7)

    # 3. 轉換為 Transaction 格式
    transactions = []
    for _, row in df_filtered.iterrows():
        transaction = []
        for col in features:
            val = row[col]
            t_low, t_high = boundaries[col]
            
            if val <= t_low:
                transaction.append(f"{col}_low")
            elif val <= t_high:
                transaction.append(f"{col}_medium")
            else:
                transaction.append(f"{col}_high")
        transactions.append(transaction)

    # 4. One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

    # 5. 執行 FP-growth (找出 Support >= 0.3)
    print("Executing FP-growth algorithm...")
    frequent_itemsets = fpgrowth(df_transactions, min_support=0.3, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    
    print("\n=== 3(a): Frequent Itemsets (Support >= 0.3) ===")
    print(frequent_itemsets.to_string(index=False))


    # ==========================================
    # Task 3(b): 關聯規則探勘 (Association Rules)
    # ==========================================
    print("\nExecuting Association Rules generation...")
    
    # 1. 根據 confidence >= 0.4 篩選規則
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    
    # 2. 進一步根據 lift >= 0.8 篩選
    rules = rules[rules['lift'] >= 0.8]
    
    # 3. 依照 confidence (主) 和 lift (副) 由高到低排序，方便觀察
    rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
    
    print("\n=== 3(b): Association Rules (Support >= 0.3, Conf >= 0.4, Lift >= 0.8) ===")
    # 為了版面簡潔，我們只挑選最重要的幾個指標欄位印出
    columns_to_show = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(rules[columns_to_show].to_string(index=False))

if __name__ == '__main__':
    main()