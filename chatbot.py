from flask import Flask, request, jsonify
from recomendation import get_top_n_recommendations, algo  # adjust import as needed

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_id = 1  # Or extract from session/input
    recommendations = get_top_n_recommendations(algo, user_id, [101, 102, 103])
    response = {
        "message": "Here are some recommended products for you:",
        "recommendations": [f"Product {pred.iid} (Rating: {pred.est:.1f})" for pred in recommendations]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

