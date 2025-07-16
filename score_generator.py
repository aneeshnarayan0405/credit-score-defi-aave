import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def feature_engineering(df):
    grouped = df.groupby('wallet')
    feature_df = pd.DataFrame()
    feature_df['total_txns'] = grouped.size()
    feature_df['n_deposits'] = grouped.apply(lambda g: (g['action'] == 'deposit').sum())
    feature_df['n_borrows'] = grouped.apply(lambda g: (g['action'] == 'borrow').sum())
    feature_df['n_repays'] = grouped.apply(lambda g: (g['action'] == 'repay').sum())
    feature_df['n_redeems'] = grouped.apply(lambda g: (g['action'] == 'redeemUnderlying').sum())
    feature_df['n_liquidations'] = grouped.apply(lambda g: (g['action'] == 'liquidationCall').sum())
    feature_df['total_amount'] = grouped['amount'].sum()
    feature_df['avg_amount'] = grouped['amount'].mean()
    feature_df['std_amount'] = grouped['amount'].std()
    feature_df['avg_price_usd'] = grouped['price_usd'].mean()
    feature_df['unique_assets'] = grouped['asset'].nunique()
    return feature_df.fillna(0).reset_index()

def generate_scores(input_json):
    with open(input_json, 'r') as f:
        raw_data = json.load(f)

    records = []
    for tx in raw_data:
        record = {
            'wallet': tx.get('userWallet'),
            'action': tx.get('action'),
            'timestamp': tx.get('timestamp'),
        }
        ad = tx.get('actionData', {})
        record['amount'] = pd.to_numeric(ad.get('amount', 0), errors='coerce')
        record['asset'] = ad.get('assetSymbol')
        record['price_usd'] = pd.to_numeric(ad.get('assetPriceUSD', 0), errors='coerce')
        records.append(record)

    df = pd.DataFrame(records).dropna(subset=['wallet', 'amount'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    features = feature_engineering(df)

    X = features.drop('wallet', axis=1)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=5).fit_transform(X_scaled)
    clusters = KMeans(n_clusters=5, random_state=42).fit_predict(X_pca)

    cluster_score_map = {0: 900, 1: 700, 2: 500, 3: 300, 4: 100}
    features['cluster'] = clusters
    features['credit_score'] = features['cluster'].map(cluster_score_map)

    result = features[['wallet', 'credit_score']]
    result.to_csv("wallet_scores.csv", index=False)
    result.sort_values(by='credit_score', ascending=False).head(10).to_csv("top_wallets.csv", index=False)
    result.sort_values(by='credit_score').head(10).to_csv("bottom_wallets.csv", index=False)
    print("✅ Scores generated and saved to wallet_scores.csv")
