import gradio as gr
import joblib
import numpy as np
import pandas as pd
import ast
import os

# ============================================================================
# Load trained models, encoders, scalers, and association rules
# ============================================================================
model_dir = "./model"

# Decision Tree Classifier
dt_classifier = joblib.load(os.path.join(model_dir,
                                         "decisiontree_classifier_baseline.pkl"))

# Naive Bayes Classifier
nb_classifier = joblib.load(os.path.join(model_dir,
                                         "naive_Bayes_classifier_optimum.pkl"))
label_encoders_2 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_2.pkl"))

# KNN Classifier
knn_classifier = joblib.load(os.path.join(model_dir,
                                          "knn_classifier_optimum.pkl"))
onehot_encoder_3 = joblib.load(os.path.join(model_dir,
                                            "onehot_encoder_3.pkl"))
scaler_3 = joblib.load(os.path.join(model_dir, "scaler_3.pkl"))

# SVM Classifier
svm_classifier = joblib.load(os.path.join(model_dir,
                                          "support_vector_classifier_optimum.pkl"))
label_encoders_4 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_4.pkl"))
scaler_4 = joblib.load(os.path.join(model_dir, "scaler_4.pkl"))

# Random Forest Classifier
rf_classifier = joblib.load(os.path.join(model_dir,
                                         "random_forest_classifier_optimum.pkl"))
label_encoders_5 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_5.pkl"))
scaler_5 = joblib.load(os.path.join(model_dir, "scaler_5.pkl"))

# Association Rules
association_rules = pd.read_csv(os.path.join(model_dir, "top_rules_7b.csv"))


# ============================================================================
# Decision Tree Classifier - Customer Churn
# ============================================================================
def predict_dt(monthly_fee, customer_age, support_calls):
    X = np.array([[monthly_fee, customer_age, support_calls]])
    prediction = dt_classifier.predict(X)
    return int(prediction[0])


# ============================================================================
# Naive Bayes Classifier - Online Shopper Purchase Intention
# ============================================================================
def predict_nb(Administrative, Administrative_Duration, Informational,
               Informational_Duration, ProductRelated, ProductRelated_Duration,
               BounceRates, ExitRates, PageValues, SpecialDay, Month,
               OperatingSystems, Browser, Region, TrafficType,
               VisitorType, Weekend):
    new_data = pd.DataFrame([{
        'Administrative': Administrative,
        'Administrative_Duration': Administrative_Duration,
        'Informational': Informational,
        'Informational_Duration': Informational_Duration,
        'ProductRelated': ProductRelated,
        'ProductRelated_Duration': ProductRelated_Duration,
        'BounceRates': BounceRates,
        'ExitRates': ExitRates,
        'PageValues': PageValues,
        'SpecialDay': SpecialDay,
        'Month': Month,
        'OperatingSystems': OperatingSystems,
        'Browser': Browser,
        'Region': Region,
        'TrafficType': TrafficType,
        'VisitorType': VisitorType,
        'Weekend': Weekend
    }])

    for col in ['Month', 'VisitorType', 'Weekend']:
        if col in label_encoders_2:
            new_data[col] = label_encoders_2[col].transform(new_data[col])

    prediction = nb_classifier.predict(new_data)[0]
    return f"{int(prediction)} ({'Will Purchase' if prediction == 1 else 'No Purchase'})"


# ============================================================================
# KNN Classifier - Late Delivery Risk
# ============================================================================
def predict_knn(Days_for_shipping_real, Days_for_shipment_scheduled,
                Order_Item_Quantity, Sales, Order_Profit_Per_Order,
                Shipping_Mode):
    new_data = pd.DataFrame([{
        'Days for shipping (real)': Days_for_shipping_real,
        'Days for shipment (scheduled)': Days_for_shipment_scheduled,
        'Order Item Quantity': Order_Item_Quantity,
        'Sales': Sales,
        'Order Profit Per Order': Order_Profit_Per_Order,
        'Shipping Mode': Shipping_Mode
    }])

    encoded = onehot_encoder_3.transform(new_data[['Shipping Mode']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=onehot_encoder_3.get_feature_names_out(['Shipping Mode']),
        index=new_data.index
    )
    new_data = pd.concat(
        [new_data.drop('Shipping Mode', axis=1), encoded_df], axis=1
    )
    new_data_scaled = scaler_3.transform(new_data)

    prediction = knn_classifier.predict(new_data_scaled)[0]
    return f"{int(prediction)} ({'Late Delivery' if prediction == 1 else 'On-time'})"


# ============================================================================
# Product Recommender - Association Rules
# ============================================================================
def recommend_products(products_text):
    input_products = set(
        [p.strip().lower() for p in products_text.split(",") if p.strip()]
    )
    if not input_products:
        return "Please enter at least one product."

    recommendations = []
    for _, rule in association_rules.iterrows():
        antecedents = ast.literal_eval(
            rule['antecedents'].replace('frozenset(', '').rstrip(')')
        )
        consequents = ast.literal_eval(
            rule['consequents'].replace('frozenset(', '').rstrip(')')
        )
        antecedents = set(antecedents)
        consequents = set(consequents)

        if antecedents.issubset(input_products):
            for product in consequents:
                if product not in input_products:
                    recommendations.append({
                        'product': product,
                        'confidence': round(rule['confidence'], 4),
                        'lift': round(rule['lift'], 4),
                        'based_on': ', '.join(antecedents)
                    })

    seen = set()
    unique = []
    for rec in sorted(recommendations, key=lambda x: x['lift'], reverse=True):
        if rec['product'] not in seen:
            seen.add(rec['product'])
            unique.append(rec)

    if not unique:
        return "No recommendations found. Try: whole milk, other vegetables"

    result = ""
    for rec in unique:
        result += (f"• {rec['product']} "
                   f"(confidence: {rec['confidence']}, lift: {rec['lift']}, "
                   f"based on: {rec['based_on']})\n")
    return result


# ============================================================================
# Build Gradio interfaces using Tabs
# ============================================================================
with gr.Blocks(title="ML Model Serving Dashboard") as demo:
    gr.Markdown("# ML Model Serving Dashboard")
    gr.Markdown("Interact with multiple ML models and a product recommender.")

    with gr.Tab("DT Classifier"):
        gr.Markdown("### Predict Customer Churn (Decision Tree)")
        with gr.Row():
            dt_fee = gr.Number(label="Monthly Fee", value=50)
            dt_age = gr.Number(label="Customer Age", value=30)
            dt_calls = gr.Number(label="Support Calls", value=3)
        dt_btn = gr.Button("Predict")
        dt_output = gr.Number(label="Predicted Class")
        dt_btn.click(predict_dt, [dt_fee, dt_age, dt_calls], dt_output)

    with gr.Tab("NB Classifier"):
        gr.Markdown("### Predict Online Shopper Purchase Intention (Naive Bayes)")
        with gr.Row():
            nb_admin = gr.Number(label="Administrative", value=0)
            nb_admin_dur = gr.Number(label="Admin Duration", value=0.0)
            nb_info = gr.Number(label="Informational", value=0)
        with gr.Row():
            nb_info_dur = gr.Number(label="Info Duration", value=0.0)
            nb_prod = gr.Number(label="ProductRelated", value=1)
            nb_prod_dur = gr.Number(label="ProductRelated Duration", value=0.0)
        with gr.Row():
            nb_bounce = gr.Number(label="BounceRates", value=0.02)
            nb_exit = gr.Number(label="ExitRates", value=0.05)
            nb_page = gr.Number(label="PageValues", value=0.0)
        with gr.Row():
            nb_special = gr.Number(label="SpecialDay", value=0.0)
            nb_month = gr.Dropdown(
                choices=["Jan", "Feb", "Mar", "May", "June",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                label="Month", value="Feb"
            )
            nb_os = gr.Number(label="OperatingSystems", value=1)
        with gr.Row():
            nb_browser = gr.Number(label="Browser", value=1)
            nb_region = gr.Number(label="Region", value=1)
            nb_traffic = gr.Number(label="TrafficType", value=1)
        with gr.Row():
            nb_visitor = gr.Dropdown(
                choices=["Returning_Visitor", "New_Visitor", "Other"],
                label="VisitorType", value="Returning_Visitor"
            )
            nb_weekend = gr.Dropdown(
                choices=["False", "True"],
                label="Weekend", value="False"
            )
        nb_btn = gr.Button("Predict")
        nb_output = gr.Textbox(label="Prediction")
        nb_btn.click(predict_nb,
                     [nb_admin, nb_admin_dur, nb_info, nb_info_dur,
                      nb_prod, nb_prod_dur, nb_bounce, nb_exit, nb_page,
                      nb_special, nb_month, nb_os, nb_browser, nb_region,
                      nb_traffic, nb_visitor, nb_weekend],
                     nb_output)

    with gr.Tab("KNN Classifier"):
        gr.Markdown("### Predict Late Delivery Risk (KNN)")
        with gr.Row():
            knn_real = gr.Number(label="Days for Shipping (Real)", value=3)
            knn_sched = gr.Number(label="Days for Shipment (Scheduled)",
                                  value=4)
        with gr.Row():
            knn_qty = gr.Number(label="Order Item Quantity", value=1)
            knn_sales = gr.Number(label="Sales", value=250.0)
        with gr.Row():
            knn_profit = gr.Number(label="Order Profit Per Order", value=64.17)
            knn_mode = gr.Dropdown(
                choices=["Standard Class", "Second Class",
                         "First Class", "Same Day"],
                label="Shipping Mode", value="Standard Class"
            )
        knn_btn = gr.Button("Predict")
        knn_output = gr.Textbox(label="Prediction")
        knn_btn.click(predict_knn,
                      [knn_real, knn_sched, knn_qty, knn_sales,
                       knn_profit, knn_mode],
                      knn_output)

    with gr.Tab("Product Recommender"):
        gr.Markdown("### Grocery Product Recommender (Association Rules)")
        gr.Markdown("Enter products separated by commas. "
                    "Examples: whole milk, other vegetables, yogurt")
        rec_input = gr.Textbox(
            label="Products",
            placeholder="whole milk, other vegetables",
            value="whole milk, other vegetables"
        )
        rec_btn = gr.Button("Get Recommendations")
        rec_output = gr.Textbox(label="Recommendations", lines=8)
        rec_btn.click(recommend_products, rec_input, rec_output)

if __name__ == "__main__":
    demo.launch()
