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
import numpy as np
import pandas as pd
import ast

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

# CORS(app, supports_credentials=False,
#      origins=["*"])

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

# Load Naive Bayes classifier and its label encoders
naive_bayes_classifier_optimum = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')  # Note: capital B in Bayes
label_encoders_2 = joblib.load('./model/label_encoders_2.pkl')

# Load KNN classifier, its one-hot encoder, and scaler
knn_classifier_optimum = joblib.load('./model/knn_classifier_optimum.pkl')
onehot_encoder_3 = joblib.load('./model/onehot_encoder_3.pkl')
scaler_3 = joblib.load('./model/scaler_3.pkl')

# Load SVM classifier, its label encoders, and scaler
support_vector_classifier_optimum = joblib.load('./model/support_vector_classifier_optimum.pkl')
label_encoders_4 = joblib.load('./model/label_encoders_4.pkl')
scaler_4 = joblib.load('./model/scaler_4.pkl')

# Load Random Forest classifier, its label encoders, and scaler
random_forest_classifier_optimum = joblib.load('./model/random_forest_classifier_optimum.pkl')
label_encoders_5 = joblib.load('./model/label_encoders_5.pkl')
scaler_5 = joblib.load('./model/scaler_5.pkl')

# Load association rules for the product recommender
association_rules = pd.read_csv('./model/top_rules_7b.csv')

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

# ============================================================================
# Naive Bayes Classifier Endpoint
# Dataset: Online Shoppers Purchasing Intention
# Target: Revenue (0 = No Purchase, 1 = Purchase)
# ============================================================================
@app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'])
def predict_naive_bayes_classifier():
    data = request.get_json()

    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])

    # Encode categorical columns using label encoders
    for col in ['Month', 'VisitorType', 'Weekend']:
        if col in new_data.columns and col in label_encoders_2:
            new_data[col] = label_encoders_2[col].transform(new_data[col])

    expected_features = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
        'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    new_data = new_data[expected_features]

    prediction = naive_bayes_classifier_optimum.predict(new_data)[0]
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "Administrative": 0,
#     "Administrative_Duration": 0.0,
#     "Informational": 0,
#     "Informational_Duration": 0.0,
#     "ProductRelated": 1,
#     "ProductRelated_Duration": 0.0,
#     "BounceRates": 0.2,
#     "ExitRates": 0.2,
#     "PageValues": 0.0,
#     "SpecialDay": 0.0,
#     "Month": "Feb",
#     "OperatingSystems": 1,
#     "Browser": 1,
#     "Region": 1,
#     "TrafficType": 1,
#     "VisitorType": "Returning_Visitor",
#     "Weekend": false
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/naive-bayes-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Administrative\": 0, \"Administrative_Duration\": 0.0, \"Informational\": 0, \"Informational_Duration\": 0.0, \"ProductRelated\": 1, \"ProductRelated_Duration\": 0.0, \"BounceRates\": 0.2, \"ExitRates\": 0.2, \"PageValues\": 0.0, \"SpecialDay\": 0.0, \"Month\": \"Feb\", \"OperatingSystems\": 1, \"Browser\": 1, \"Region\": 1, \"TrafficType\": 1, \"VisitorType\": \"Returning_Visitor\", \"Weekend\": false}"


# ============================================================================
# KNN Classifier Endpoint
# Dataset: DataCo Smart Supply Chain
# Target: Late_delivery_risk (0 = On-time, 1 = Late)
# ============================================================================
@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn_classifier():
    data = request.get_json()

    new_data = pd.DataFrame([{
        'Days for shipping (real)': data.get('Days_for_shipping_real'),
        'Days for shipment (scheduled)': data.get('Days_for_shipment_scheduled'),
        'Order Item Quantity': data.get('Order_Item_Quantity'),
        'Sales': data.get('Sales'),
        'Order Profit Per Order': data.get('Order_Profit_Per_Order'),
        'Shipping Mode': data.get('Shipping_Mode')
    }])

    # One-hot encode 'Shipping Mode'
    encoded = onehot_encoder_3.transform(new_data[['Shipping Mode']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=onehot_encoder_3.get_feature_names_out(['Shipping Mode']),
        index=new_data.index
    )

    # Drop the original 'Shipping Mode' column and concatenate the encoded columns
    new_data = pd.concat(
        [new_data.drop('Shipping Mode', axis=1), encoded_df],
        axis=1
    )

    # Scale the data using the fitted scaler
    new_data_scaled = scaler_3.transform(new_data)

    prediction = knn_classifier_optimum.predict(new_data_scaled)[0]
    return jsonify({'Predicted Late Delivery Risk = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "Days_for_shipping_real": 3,
#     "Days_for_shipment_scheduled": 4,
#     "Order_Item_Quantity": 1,
#     "Sales": 250.0,
#     "Order_Profit_Per_Order": 64.17,
#     "Shipping_Mode": "Second Class"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/knn-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Days_for_shipping_real\": 3, \"Days_for_shipment_scheduled\": 4, \"Order_Item_Quantity\": 1, \"Sales\": 250.0, \"Order_Profit_Per_Order\": 64.17, \"Shipping_Mode\": \"Second Class\"}"


# ============================================================================
# SVM Classifier Endpoint
# Dataset: Online Shoppers Purchasing Intention
# Target: Revenue (0 = No Purchase, 1 = Purchase)
# ============================================================================
@app.route('/api/v1/models/svm-classifier/predictions', methods=['POST'])
def predict_svm_classifier():
    data = request.get_json()

    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])

    # Encode categorical columns using label encoders
    for col in ['Month', 'VisitorType', 'Weekend']:
        if col in new_data.columns and col in label_encoders_4:
            new_data[col] = label_encoders_4[col].transform(new_data[col])

    expected_features = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
        'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    new_data = new_data[expected_features]

    # Scale the data using the fitted scaler
    new_data_scaled = scaler_4.transform(new_data)

    prediction = support_vector_classifier_optimum.predict(new_data_scaled)[0]
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values (same input format as Naive Bayes)
# {
#     "Administrative": 0,
#     "Administrative_Duration": 0.0,
#     "Informational": 0,
#     "Informational_Duration": 0.0,
#     "ProductRelated": 1,
#     "ProductRelated_Duration": 0.0,
#     "BounceRates": 0.2,
#     "ExitRates": 0.2,
#     "PageValues": 0.0,
#     "SpecialDay": 0.0,
#     "Month": "Feb",
#     "OperatingSystems": 1,
#     "Browser": 1,
#     "Region": 1,
#     "TrafficType": 1,
#     "VisitorType": "Returning_Visitor",
#     "Weekend": false
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/svm-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Administrative\": 0, \"Administrative_Duration\": 0.0, \"Informational\": 0, \"Informational_Duration\": 0.0, \"ProductRelated\": 1, \"ProductRelated_Duration\": 0.0, \"BounceRates\": 0.2, \"ExitRates\": 0.2, \"PageValues\": 0.0, \"SpecialDay\": 0.0, \"Month\": \"Feb\", \"OperatingSystems\": 1, \"Browser\": 1, \"Region\": 1, \"TrafficType\": 1, \"VisitorType\": \"Returning_Visitor\", \"Weekend\": false}"


# ============================================================================
# Random Forest Classifier Endpoint
# Dataset: Online Shoppers Purchasing Intention
# Target: Revenue (0 = No Purchase, 1 = Purchase)
# ============================================================================
@app.route('/api/v1/models/random-forest-classifier/predictions', methods=['POST'])
def predict_random_forest_classifier():
    data = request.get_json()

    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])

    # Encode categorical columns using label encoders
    for col in ['Month', 'VisitorType', 'Weekend']:
        if col in new_data.columns and col in label_encoders_5:
            new_data[col] = label_encoders_5[col].transform(new_data[col])

    expected_features = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
        'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    new_data = new_data[expected_features]

    # Scale the data using the fitted scaler
    new_data_scaled = scaler_5.transform(new_data)

    prediction = random_forest_classifier_optimum.predict(new_data_scaled)[0]
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values (same input format as Naive Bayes and SVM)
# {
#     "Administrative": 0,
#     "Administrative_Duration": 0.0,
#     "Informational": 0,
#     "Informational_Duration": 0.0,
#     "ProductRelated": 1,
#     "ProductRelated_Duration": 0.0,
#     "BounceRates": 0.2,
#     "ExitRates": 0.2,
#     "PageValues": 0.0,
#     "SpecialDay": 0.0,
#     "Month": "Feb",
#     "OperatingSystems": 1,
#     "Browser": 1,
#     "Region": 1,
#     "TrafficType": 1,
#     "VisitorType": "Returning_Visitor",
#     "Weekend": false
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/random-forest-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"Administrative\": 0, \"Administrative_Duration\": 0.0, \"Informational\": 0, \"Informational_Duration\": 0.0, \"ProductRelated\": 1, \"ProductRelated_Duration\": 0.0, \"BounceRates\": 0.2, \"ExitRates\": 0.2, \"PageValues\": 0.0, \"SpecialDay\": 0.0, \"Month\": \"Feb\", \"OperatingSystems\": 1, \"Browser\": 1, \"Region\": 1, \"TrafficType\": 1, \"VisitorType\": \"Returning_Visitor\", \"Weekend\": false}"


# ============================================================================
# Product Recommender Endpoint
# Based on association rules from the Groceries dataset (Hahsler et al., 2011)
# ============================================================================
@app.route('/api/v1/recommender/association-rules', methods=['POST'])
def recommend_products():
    data = request.get_json()
    # Expects: {"products": ["whole milk", "other vegetables"]}
    input_products = set(data.get('products', []))

    if not input_products:
        return jsonify({'error': 'Please provide a list of products'}), 400

    recommendations = []
    for _, rule in association_rules.iterrows():
        # Parse the frozenset strings back to sets
        antecedents = ast.literal_eval(rule['antecedents'].replace('frozenset(', '').rstrip(')'))
        consequents = ast.literal_eval(rule['consequents'].replace('frozenset(', '').rstrip(')'))
        antecedents = set(antecedents)
        consequents = set(consequents)

        # If the input products contain all antecedents, recommend the consequents
        if antecedents.issubset(input_products):
            for product in consequents:
                if product not in input_products:
                    recommendations.append({
                        'recommended_product': product,
                        'confidence': round(rule['confidence'], 4),
                        'lift': round(rule['lift'], 4),
                        'based_on': list(antecedents)
                    })

    # Remove duplicates and sort by lift (best recommendations first)
    seen = set()
    unique_recommendations = []
    for rec in sorted(recommendations, key=lambda x: x['lift'], reverse=True):
        if rec['recommended_product'] not in seen:
            seen.add(rec['recommended_product'])
            unique_recommendations.append(rec)

    return jsonify({
        'input_products': list(input_products),
        'recommendations': unique_recommendations
    })

# *1* Sample JSON POST values
# {
#     "products": ["whole milk", "other vegetables"]
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/recommender/association-rules \
#   -H "Content-Type: application/json" \
#   -d "{\"products\": [\"whole milk\", \"other vegetables\"]}"


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
if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)
