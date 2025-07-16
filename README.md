# credit-score-defi-aave

# 🏦 DeFi Wallet Credit Scoring System — Aave V2

This project assigns a **credit score (0–1000)** to DeFi wallets that interact with the **Aave V2 protocol**, using historical transaction behavior. The goal is to identify responsible, high-quality users versus risky or exploitative behavior, based on features like deposits, borrows, repayments, and more.

---

## 🎯 Objective

Given a dataset of ~100,000 DeFi wallet transactions, develop a machine learning model that:

- Extracts wallet-level behavioral features from raw transaction logs
- Clusters wallet behavior into risk categories
- Assigns a **credit score from 0 to 1000** to each wallet
- Produces a transparent scoring method that is explainable and reproducible

---

## 🏗️ Project Structure

```
credit-score-defi-aave/
│
├── data/                          # Place the raw JSON here
│   └── user-wallet-transactions.json
│
├──  credit-score-defi-aave.ipynb
│
├── score_generator.py            
├── wallet_scores.csv             # Output: wallet with assigned credit score
├── top_wallets.csv               # Top 10 highest scoring wallets
├── bottom_wallets.csv            # Top 10 lowest-scoring wallets
│
├── README.md                     # Project overview (you are here!)
├── requirements.txt              # Python libraries used
```

---

## ⚙️ Methodology

1. **Data Parsing**  
   Parse nested JSON logs and extract transaction metadata like wallet address, action type, amount, asset, price, and timestamp.

2. **Feature Engineering**  
   For each wallet, compute features such as:
   - Number of deposits, borrows, repayments, redeems, liquidations
   - Total, average, and std. deviation of amounts
   - Unique asset types
   - Average USD price of assets used

3. **Clustering + Scoring**  
   - Normalize features using `StandardScaler`
   - Reduce dimensions with PCA
   - Apply `KMeans (k=5)` clustering to segment wallets
   - Map each cluster to a score bucket:
     ```
     Cluster 0 → 900
     Cluster 1 → 700
     Cluster 2 → 500
     Cluster 3 → 300
     Cluster 4 → 100
     ```

4. **Export Results**  
   - Final scores are saved in `wallet_scores.csv`
   - Top and bottom wallets exported for inspection

---

## 💡 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place JSON file
Download and place `user-wallet-transactions.json` from:
https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing

Putting it inside the `data/` directory.

### Step 3: Run script
```bash
python score_generator.py
```

You’ll get the following files:
- `wallet_scores.csv`
- `top_wallets.csv`
- `bottom_wallets.csv`

Or run the Jupyter notebook:
```bash
notebooks/01_end_to_end_credit_scoring.ipynb
```

---

## 📊 Output Example

| Wallet                                | Credit Score |
|--------------------------------------|--------------|
| 0x00000000001accfa9cef68cf5371a23025 | 900          |
| 0xABCDEF1234567890ABCDEF1234567890AB | 300          |

---

## 📈 Features Used for Scoring

- Transaction frequency and diversity
- Asset volume and average price
- Behavior patterns (e.g., repayment vs liquidation)
- Cluster-based unsupervised learning

---

## 📎 Files

- `README.md`: Complete project documentation
- `score_generator.py`: Script for processing and scoring
- `wallet_scores.csv`: Final output

---

## 📦 Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them with:
```bash
pip install -r requirements.txt
```

---
