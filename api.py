from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import ast
import warnings
from functools import wraps

# Suppress scikit-learn version mismatch warning (temporary - better to retrain models)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

app = Flask(__name__)

# CORS - generous for local development; tighten in production
CORS(
    app,
    supports_credentials=False,
    resources={r"/api/*": {
        "origins": ["*"]  # ← for dev convenience; restrict later
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Global storage for models and preprocessors
models = {}
preprocessors = {}
association_rules = None

# ───────────────────────────────────────────────
# Load models with graceful error handling
# ───────────────────────────────────────────────
try:
    # Decision Tree models
    models['dt_classifier'] = joblib.load('./model/decisiontree_classifier_baseline.pkl')
    models['dt_regressor'] = joblib.load('./model/decisiontree_regressor_optimum.pkl')
    preprocessors['label_encoders_1b'] = joblib.load('./model/label_encoders_1b.pkl')

    # Naive Bayes
    models['nb_classifier'] = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')
    preprocessors['label_encoders_2'] = joblib.load('./model/label_encoders_2.pkl')

    # KNN
    models['knn_classifier'] = joblib.load('./model/knn_classifier_optimum.pkl')
    preprocessors['onehot_encoder_3'] = joblib.load('./model/onehot_encoder_3.pkl')
    preprocessors['scaler_3'] = joblib.load('./model/scaler_3.pkl')

    # SVM
    models['svm_classifier'] = joblib.load('./model/support_vector_classifier_optimum.pkl')
    preprocessors['label_encoders_4'] = joblib.load('./model/label_encoders_4.pkl')
    preprocessors['scaler_4'] = joblib.load('./model/scaler_4.pkl')

    # Random Forest
    models['rf_classifier'] = joblib.load('./model/random_forest_classifier_optimum.pkl')
    preprocessors['label_encoders_5'] = joblib.load('./model/label_encoders_5.pkl')
    preprocessors['scaler_5'] = joblib.load('./model/scaler_5.pkl')

    # Association rules
    association_rules = pd.read_csv('./model/top_rules_7b.csv')

    print("Core models and association rules loaded successfully.")

except Exception as e:
    print(f"Error loading core models: {e}")

# Optional: K-Means (advanced) - won't crash if missing
try:
    models['kmeans'] = joblib.load('./model/kmeans_model.pkl')
    preprocessors['scaler_cluster'] = joblib.load('./model/scaler_cluster.pkl')  # adjust name if different
    print("K-Means model loaded.")
except FileNotFoundError:
    print("K-Means model files not found → /predict/cluster endpoint disabled")
    models['kmeans'] = None
except Exception as e:
    print(f"Error loading K-Means: {e}")
    models['kmeans'] = None

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    loaded = list(models.keys())
    has_rules = association_rules is not None
    return jsonify({
        "status": "healthy",
        "loaded_models": loaded,
        "association_rules_loaded": has_rules
    })

# ───────────────────────────────────────────────
# Decision Tree Classifier
# ───────────────────────────────────────────────
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    try:
        data = request.get_json()
        required = {'monthly_fee', 'customer_age', 'support_calls'}
        if not data or not required.issubset(data.keys()):
            missing = required - set(data or {})
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        new_data = pd.DataFrame([{
            'monthly_fee': float(data['monthly_fee']),
            'customer_age': float(data['customer_age']),
            'support_calls': float(data['support_calls'])
        }])

        pred = models['dt_classifier'].predict(new_data)[0]
        return jsonify({"predicted_class": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

# ───────────────────────────────────────────────
# Decision Tree Regressor
# ───────────────────────────────────────────────
@app.route('/api/v1/models/decision-tree-regressor/predictions', methods=['POST'])
def predict_decision_tree_regressor():
    try:
        data = request.get_json()
        required = {'PaymentDate', 'CustomerType', 'BranchSubCounty', 'ProductCategoryName', 'QuantityOrdered'}
        if not data or not required.issubset(data.keys()):
            missing = required - set(data or {})
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame([data])
        df['PaymentDate'] = pd.to_datetime(df['PaymentDate'])

        df['PaymentDate_year'] = df['PaymentDate'].dt.year
        df['PaymentDate_month'] = df['PaymentDate'].dt.month
        df['PaymentDate_day'] = df['PaymentDate'].dt.day
        df['PaymentDate_dayofweek'] = df['PaymentDate'].dt.dayofweek

        cat_cols = ['CustomerType', 'BranchSubCounty', 'ProductCategoryName']
        for col in cat_cols:
            if col in df.columns:
                df[col] = preprocessors['label_encoders_1b'][col].transform(df[col])

        df = df.drop(columns=['PaymentDate'])

        expected = [
            'CustomerType', 'BranchSubCounty', 'ProductCategoryName', 'QuantityOrdered',
            'PaymentDate_year', 'PaymentDate_month', 'PaymentDate_day', 'PaymentDate_dayofweek'
        ]
        df = df[expected]

        pred = models['dt_regressor'].predict(df)[0]
        return jsonify({"predicted_percentage_profit_per_unit": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

# ───────────────────────────────────────────────
# Factory for Shoppers Intention Classifiers (NB, SVM, RF)
# ───────────────────────────────────────────────
def create_shoppers_predictor(model_key, encoders_key, scaler_key=None):
    def predictor():
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "No JSON payload"}), 400

            features = [
                'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
                'VisitorType', 'Weekend'
            ]

            missing = [f for f in features if f not in data]
            if missing:
                return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

            df = pd.DataFrame([data])

            # Encode categoricals
            for col in ['Month', 'VisitorType', 'Weekend']:
                if col in df.columns:
                    le = preprocessors[encoders_key].get(col)
                    if le is None:
                        return jsonify({"error": f"No encoder for column: {col}"}), 500
                    df[col] = le.transform(df[col])

            df = df[features]

            # Scale if applicable
            if scaler_key and scaler_key in preprocessors:
                X = preprocessors[scaler_key].transform(df)
            else:
                X = df.to_numpy()

            pred = models[model_key].predict(X)[0]
            prob = None
            if hasattr(models[model_key], 'predict_proba'):
                prob = models[model_key].predict_proba(X)[0].tolist()

            result = {"predicted_class": int(pred)}
            if prob:
                result["probabilities"] = [round(p, 4) for p in prob]

            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 422

    # Unique name to prevent endpoint collision
    predictor.__name__ = f"predict_{model_key.replace('-', '_')}"
    return predictor

# Register the three similar endpoints
app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'], endpoint='predict_naive_bayes')(
    create_shoppers_predictor('nb_classifier', 'label_encoders_2')
)

app.route('/api/v1/models/svm-classifier/predictions', methods=['POST'], endpoint='predict_svm')(
    create_shoppers_predictor('svm_classifier', 'label_encoders_4', scaler_key='scaler_4')
)

app.route('/api/v1/models/random-forest-classifier/predictions', methods=['POST'], endpoint='predict_random_forest')(
    create_shoppers_predictor('rf_classifier', 'label_encoders_5', scaler_key='scaler_5')
)

# ───────────────────────────────────────────────
# KNN Classifier
# ───────────────────────────────────────────────
@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn_classifier():
    try:
        data = request.get_json()
        required = {
            'Days_for_shipping_real', 'Days_for_shipment_scheduled',
            'Order_Item_Quantity', 'Sales', 'Order_Profit_Per_Order', 'Shipping_Mode'
        }
        if not data or not required.issubset(data.keys()):
            missing = required - set(data or {})
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame([data])

        # One-hot encode Shipping Mode
        encoded = preprocessors['onehot_encoder_3'].transform(df[['Shipping_Mode']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=preprocessors['onehot_encoder_3'].get_feature_names_out(),
            index=df.index
        )

        df = pd.concat([df.drop('Shipping_Mode', axis=1), encoded_df], axis=1)
        scaled = preprocessors['scaler_3'].transform(df)

        pred = models['knn_classifier'].predict(scaled)[0]
        return jsonify({"predicted_late_delivery_risk": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

# ───────────────────────────────────────────────
# Association Rules Recommender
# ───────────────────────────────────────────────
@app.route('/api/v1/recommender/association-rules', methods=['POST'])
def recommend_products():
    try:
        data = request.get_json()
        products = data.get('products', [])
        if not isinstance(products, list) or not products:
            return jsonify({"error": "Provide a non-empty list of products"}), 400

        input_set = set(products)
        recommendations = []

        for _, row in association_rules.iterrows():
            try:
                ants_str = row['antecedents']
                cons_str = row['consequents']
                antecedents = ast.literal_eval(ants_str)
                consequents = ast.literal_eval(cons_str)
                ants = set(antecedents) if isinstance(antecedents, (list, tuple, set)) else antecedents
                cons = set(consequents) if isinstance(consequents, (list, tuple, set)) else consequents
            except:
                continue

            if ants.issubset(input_set):
                for item in cons - input_set:
                    recommendations.append({
                        "recommended_product": item,
                        "confidence": float(row.get('confidence', 0)),
                        "lift": float(row.get('lift', 0)),
                        "support": float(row.get('support', 0)),
                        "based_on": list(ants)
                    })

        # Deduplicate and sort by lift descending
        seen = set()
        unique_recs = []
        for r in sorted(recommendations, key=lambda x: x['lift'], reverse=True):
            prod = r['recommended_product']
            if prod not in seen:
                seen.add(prod)
                unique_recs.append(r)

        return jsonify({
            "input_products": products,
            "recommendations": unique_recs[:10]  # limit to top 10
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 422

# ───────────────────────────────────────────────
# Optional: K-Means Cluster Predictor (advanced)
# ───────────────────────────────────────────────
@app.route('/api/v1/models/kmeans-cluster/predictions', methods=['POST'])
def predict_cluster():
    if models.get('kmeans') is None:
        return jsonify({"error": "K-Means model not available"}), 503

    try:
        data = request.get_json()
        features = data.get('features')
        if not features:
            return jsonify({"error": "Provide 'features' as list or object"}), 400

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame([features])  # assume correct order

        scaled = preprocessors['scaler_cluster'].transform(df)
        cluster = int(models['kmeans'].predict(scaled)[0])

        return jsonify({"predicted_cluster": cluster})
    except Exception as e:
        return jsonify({"error": str(e)}), 422

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)