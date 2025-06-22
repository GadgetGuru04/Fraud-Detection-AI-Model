from flask import Flask, request, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

# Load and prepare your data (as above)
data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'product_id': [101, 102, 101, 103, 102, 103],
    'rating': [5, 4, 4, 2, 3, 5]
}
df = pd.DataFrame(data)
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
trainset = dataset.build_full_trainset()
algo = SVD()
algo.fit(trainset)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    product_ids = data.get('product_ids', [101, 102, 103])  # Default or from request
    predictions = [algo.predict(user_id, pid) for pid in product_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [{'product_id': pred.iid, 'predicted_rating': pred.est} for pred in predictions[:3]]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return "Flask server is running! Try POSTing to /recommend."

@app.errorhandler(404)
def not_found(e):
    return "Sorry, this page does not exist.", 404
