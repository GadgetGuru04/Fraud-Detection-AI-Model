from surprise import Dataset, Reader, SVD
import pandas as pd

# Example data: user_id, product_id, rating
data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'product_id': [101, 102, 101, 103, 102, 103],
    'rating': [5, 4, 4, 2, 3, 5]
}
df = pd.DataFrame(data)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Example: Predict rating for user 1 and product 103
pred = algo.predict(1, 103)
print(f"Predicted rating for user 1 and product 103: {pred.est}")

def get_top_n_recommendations(algo, user_id, product_ids, n=3):
    predictions = [algo.predict(user_id, pid) for pid in product_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return predictions[:n]

