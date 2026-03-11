from flask import Flask, request, jsonify
# Cross-Origin Resource Sharing (CORS)
# Modern browsers apply the "same-origin policy", which blocks web pages from
# making requests to a different origin than the one that served the page.
# This helps prevent malicious sites from reading sensitive data from another
# site you are logged into.
#
# However, there are many legitimate cases where cross-origin requests are
# needed. One example is:
#
## Single-Page Applications (SPA) hosted at example-frontend.com need to call
## APIs hosted at api.example-backend.com.
#
# To support this safely, CORS lets servers explicitly allow such requests.
from flask_cors import CORS
import joblib
import pandas as pd
import os
# Get the folder where api.py is located
base_path = os.path.dirname(__file__)

# Join that path with the model folder
kmeans_path = os.path.join(base_path, 'model', 'kmeans_classifier_optimum.pkl')
rules_path = os.path.join(base_path, 'model', 'association_rules.pkl')

# Load them using the full path
kmeans_model = joblib.load(kmeans_path)
association_rules = joblib.load(rules_path)

app = Flask(__name__)
# CORS(
#     app,
#     resources={r"/api/*": {
#         "origins": [
#             "https://127.0.0.1",
#             "https://localhost"
#         ]
#     }},
#     methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type"]
# )

CORS(
    app, supports_credentials=False,
    resources={r"/api/*": { # This means CORS will only apply to routes that start with /api/
               "origins": [
                   "https://127.0.0.1", "https://localhost",
                   "https://127.0.0.1:443", "https://localhost:443",
                   "http://127.0.0.1", "http://localhost",
                   "http://127.0.0.1:5000", "http://localhost:5000",
                   "http://127.0.0.1:5500", "http://localhost:5500"
                ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"])

# Load the additional models ---
nb_model = joblib.load('model/naive_Bayes_classifier_optimum.pkl')
rf_model = joblib.load('model/random_forest_classifier_optimum.pkl')
svm_model = joblib.load('model/support_vector_classifier_optimum.pkl')
knn_model = joblib.load('model/knn_classifier_optimum.pkl')

# # Load Clustering and Association Rules ---

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

# Defines an HTTP endpoint
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    # Accepts JSON data sent by a client (browser, curl, Postman, etc.)
    data = request.get_json()
    # Create a DataFrame with the correct feature names
    new_data = pd.DataFrame([{
        'monthly_fee': data.get('monthly_fee'),
        'customer_age': data.get('customer_age'),
        'support_calls': data.get('support_calls')
    }])

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'monthly_fee',
        'customer_age',
        'support_calls'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Performs a prediction using the already trained machine learning model
    prediction = decisiontree_classifier_baseline.predict(new_data)[0]
    
    # Returns the result as a JSON response:
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

# *2.a.* Sample cURL POST values (without HTTPS in NGINX and Gunicorn)

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *3* Sample PowerShell values:

# $body = @{
#     monthly_fee = 60
#     customer_age = 30
#     support_calls = 1   
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/decision-tree-regressor/predictions', methods=['POST'])
def predict_decision_tree_regressor():
    data = request.get_json()
    # Expected input keys:
    # 'PaymentDate', 'CustomerType', 'BranchSubCounty',
    # 'ProductCategoryName', 'QuantityOrdered', 'PercentageProfitPerUnit'

    # Create a DataFrame based on the input
    new_data = pd.DataFrame([data])

    # Convert PaymentDate to datetime
    new_data['PaymentDate'] = pd.to_datetime(new_data['PaymentDate'])

    # Identify all datetime columns
    datetime_columns = new_data.select_dtypes(include=['datetime64']).columns

    categorical_cols = new_data.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns

    # Encode categorical columns
    for col in categorical_cols:
        if col in new_data:
            new_data[col] = label_encoders_1b[col].transform(new_data[col])

    # Feature engineering for date
    new_data['PaymentDate_year'] = new_data['PaymentDate'].dt.year # type: ignore
    new_data['PaymentDate_month'] = new_data['PaymentDate'].dt.month # type: ignore
    new_data['PaymentDate_day'] = new_data['PaymentDate'].dt.day # type: ignore
    new_data['PaymentDate_dayofweek'] = new_data['PaymentDate'].dt.dayofweek # type: ignore
    new_data = new_data.drop(columns=datetime_columns)

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'CustomerType',
        'BranchSubCounty',
        'ProductCategoryName',
        'QuantityOrdered',
        'PaymentDate_year',
        'PaymentDate_month',
        'PaymentDate_day',
        'PaymentDate_dayofweek'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Predict
    prediction = decisiontree_regressor_optimum.predict(new_data)[0]
    return jsonify({'Predicted Percentage Profit per Unit = ': float(prediction)})

# *1* Sample JSON POST values
# {
#     "CustomerType": "Business",
#     "BranchSubCounty": "Kilimani",
#     "ProductCategoryName": "Meat-Based Dishes",
#     "QuantityOrdered": 8,
#     "PaymentDate": "2027-11-13"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *2.b.* Sample cURL POST values

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *3* Sample PowerShell values:

# $body = @{
#     PaymentDate         = "2027-11-13"
#     CustomerType        = "Business"
#     BranchSubCounty     = "Kilimani"
#     ProductCategoryName = "Meat-Based Dishes"
#     QuantityOrdered = 8
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

# This ensures the Flask web server only starts when you run this file directly
# (e.g., `python api.py`), and not if you import api.py from another script or test.

# __name__ is a special variable in Python. When you run a script directly,
# __name__ is set to '__main__'. If the script is imported, __name__ is set to
# the module's name.

# if __name__ == '__main__': checks if the script is being run directly.

# app.run(debug=True) starts the Flask development server with debugging enabled.
# This means:
## The server will automatically reload if you make code changes.
## You get detailed error messages in the browser if something goes wrong.

# --- Prediction Endpoints ---

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    data = request.get_json()
    input_data = [data.get('customer_age', 0), data.get('monthly_fee', 0), data.get('support_calls', 0), 0, 0, 0, 0, 0]
    prediction = knn_model.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

# --- Updated Prediction Endpoints for 17 Features ---

@app.route('/predict_nb', methods=['POST'])
def predict_nb():
    data = request.get_json()
    # 3 real values + 14 zeros = 17 features total
    input_data = [
        data.get('customer_age', 0), data.get('monthly_fee', 0), data.get('support_calls', 0),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    prediction = nb_model.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    data = request.get_json()
    input_data = [
        data.get('customer_age', 0), data.get('monthly_fee', 0), data.get('support_calls', 0),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    prediction = svm_model.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    data = request.get_json()
    input_data = [
        data.get('customer_age', 0), data.get('monthly_fee', 0), data.get('support_calls', 0),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    prediction = rf_model.predict([input_data])
    return jsonify({'prediction': int(prediction[0])})

# --- Part B: Clustering and Recommendations ---

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    data = request.get_json()
    # Using 17 features as required by the models in this lab session
    input_data = [
        data.get('customer_age', 0), data.get('monthly_fee', 0), data.get('support_calls', 0),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    # Uses 'kmeans_model' as defined in your screenshot
    cluster = kmeans_model.predict([input_data])
    return jsonify({'cluster': int(cluster[0])})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    item_bought = data.get('item', '')
    
    # Association rules are often a list of rules. 
    # This checks if the item is in the rules we loaded.
    try:
        # Simple search logic for the 'association_rules' variable in your screenshot
        recs = association_rules[association_rules['antecedents'].apply(lambda x: item_bought in x)]
        if not recs.empty:
            suggestion = list(recs.iloc[0]['consequents'])[0]
        else:
            suggestion = "No recommendation found"
    except Exception:
        # Fallback if the data structure is different
        suggestion = "whole milk" 
        
    return jsonify({'bought': item_bought, 'recommendation': suggestion})

if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)
# --- Recommender Endpoint ---
@app.post("/recommend")
def get_recommendations(basket: list):
    # 'association_rules' is the variable you loaded using joblib
    # We filter for rules where the basket items are in the 'antecedents'
    recommendations = association_rules[
        association_rules['antecedents'].apply(lambda x: any(item in x for item in basket))
    ]
    
    # Sort by 'lift' and get unique 'consequents'
    top_picks = recommendations.sort_values(by='lift', ascending=False)['consequents'].unique().tolist()
    
    return {"recommendations": top_picks[:5]}

# --- Cluster Classifier Endpoint ---
@app.post("/predict_cluster")
def predict_cluster(data: dict):
    new_data = pd.DataFrame([data])
    
    # IMPORTANT: You must copy the exact Feature Engineering 
    # from your Decision Tree Regressor (lines 141-160) here 
    # so the columns match the k-means model requirements.
    # ... [Insert that logic here] ...

    # Predict using the kmeans_model you loaded
    cluster_id = kmeans_model.predict(new_data[expected_features])
    
    return {"cluster": int(cluster_id[0])}
