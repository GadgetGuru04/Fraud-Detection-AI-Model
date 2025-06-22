from surprise import Dataset, Reader, SVD
import pandas as pd

# Sample data
data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'product_id': [101, 102, 101, 103, 102, 103],
    'rating': [5, 4, 4, 2, 3, 5]
}
df = pd.DataFrame(data)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

# Define and train the model
algo = SVD()
algo.fit(trainset)


# Function to get top N recommendations for a user
def get_top_n_recommendations(algo, user_id, product_ids, n=3):
    predictions = [algo.predict(user_id, pid) for pid in product_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:n]

# Example usage
product_ids = [101, 102, 103]
top_n = get_top_n_recommendations(algo, 1, product_ids, n=2)
for pred in top_n:
    print(f"Product {pred.iid}: Predicted rating {pred.est}")
