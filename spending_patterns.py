import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv('transactions.csv')
data.drop_duplicates(inplace=True)
data.fillna(0, inplace=True)

# 2. Feature engineering
user_stats = data.groupby('user_id').agg(
    total_spent=('amount', 'sum'),
    avg_spent=('amount', 'mean'),
    num_transactions=('amount', 'count')
).reset_index()

# 3. Clustering
features = user_stats[['total_spent', 'avg_spent', 'num_transactions']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
user_stats['cluster'] = kmeans.fit_predict(features_scaled)

# 4. Anomaly detection
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05)
user_stats['anomaly'] = iso.fit_predict(features_scaled)
user_stats['anomaly'] = user_stats['anomaly'].map({1: 0, -1: 1})  # 1 is anomaly, 0 is normal

print(user_stats[user_stats['anomaly'] == 1])


# 5. Plotting
plt.scatter(user_stats['total_spent'], user_stats['num_transactions'], c=user_stats['cluster'])
plt.xlabel('Total Spent')
plt.ylabel('Number of Transactions')
plt.title('User Spending Clusters')
plt.show()
