import streamlit as st
import joblib
import numpy as np
import pandas as pd
import ast
import os

# ============================================================================
# Load trained models, encoders, scalers, and association rules
# ============================================================================
# Use the script's own directory so it works both locally and on Streamlit Cloud
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

dt_classifier = joblib.load(os.path.join(model_dir,
                                         "decisiontree_classifier_baseline.pkl"))

nb_classifier = joblib.load(os.path.join(model_dir,
                                         "naive_Bayes_classifier_optimum.pkl"))
label_encoders_2 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_2.pkl"))

knn_classifier = joblib.load(os.path.join(model_dir,
                                          "knn_classifier_optimum.pkl"))
onehot_encoder_3 = joblib.load(os.path.join(model_dir,
                                            "onehot_encoder_3.pkl"))
scaler_3 = joblib.load(os.path.join(model_dir, "scaler_3.pkl"))

svm_classifier = joblib.load(os.path.join(model_dir,
                                          "support_vector_classifier_optimum.pkl"))
label_encoders_4 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_4.pkl"))
scaler_4 = joblib.load(os.path.join(model_dir, "scaler_4.pkl"))

rf_classifier = joblib.load(os.path.join(model_dir,
                                         "random_forest_classifier_optimum.pkl"))
label_encoders_5 = joblib.load(os.path.join(model_dir,
                                            "label_encoders_5.pkl"))
scaler_5 = joblib.load(os.path.join(model_dir, "scaler_5.pkl"))

association_rules = pd.read_csv(os.path.join(model_dir, "top_rules_7b.csv"))

# Streamlit page config
st.set_page_config(
    page_title="ML Model Serving Dashboard",
    page_icon="📊",
    layout="centered"
)

st.title("ML Model Serving Dashboard")
st.write("Interact with multiple ML models and a product recommender.")

# Tab layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "DT Classifier", "NB Classifier", "KNN Classifier",
    "SVM Classifier", "RF Classifier", "Recommender"
])

# ============================================================================
# Tab 1: Decision Tree Classifier - Customer Churn
# ============================================================================
with tab1:
    st.header("Customer Churn Prediction (Decision Tree)")
    with st.form("dt_form"):
        dt_fee = st.number_input("Monthly Fee", min_value=0.0, step=1.0,
                                 key="dt_fee")
        dt_age = st.number_input("Customer Age", min_value=0, step=1,
                                 key="dt_age")
        dt_calls = st.number_input("Support Calls", min_value=0, step=1,
                                   key="dt_calls")
        dt_submit = st.form_submit_button("Predict")

    if dt_submit:
        X = np.array([[dt_fee, dt_age, dt_calls]])
        prediction = dt_classifier.predict(X)
        st.success(f"### Predicted Class: {int(prediction[0])}")

# ============================================================================
# Tab 2: Naive Bayes Classifier - Online Shopper Purchase Intention
# ============================================================================
with tab2:
    st.header("Online Shopper Purchase Intention (Naive Bayes)")
    with st.form("nb_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            nb_admin = st.number_input("Administrative", value=0, key="nb_admin")
            nb_info = st.number_input("Informational", value=0, key="nb_info")
            nb_prod = st.number_input("ProductRelated", value=1, key="nb_prod")
            nb_bounce = st.number_input("BounceRates", value=0.02,
                                        format="%.4f", key="nb_bounce")
            nb_page = st.number_input("PageValues", value=0.0, key="nb_page")
            nb_os = st.number_input("OperatingSystems", value=1, key="nb_os")
        with col2:
            nb_admin_dur = st.number_input("Admin Duration", value=0.0,
                                           key="nb_admin_dur")
            nb_info_dur = st.number_input("Info Duration", value=0.0,
                                          key="nb_info_dur")
            nb_prod_dur = st.number_input("ProductRelated Duration", value=0.0,
                                          key="nb_prod_dur")
            nb_exit = st.number_input("ExitRates", value=0.05, format="%.4f",
                                      key="nb_exit")
            nb_special = st.number_input("SpecialDay", value=0.0,
                                         key="nb_special")
            nb_browser = st.number_input("Browser", value=1, key="nb_browser")
        with col3:
            nb_month = st.selectbox("Month",
                                    ["Jan", "Feb", "Mar", "May", "June",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                    key="nb_month")
            nb_region = st.number_input("Region", value=1, key="nb_region")
            nb_traffic = st.number_input("TrafficType", value=1,
                                         key="nb_traffic")
            nb_visitor = st.selectbox("VisitorType",
                                      ["Returning_Visitor", "New_Visitor",
                                       "Other"],
                                      key="nb_visitor")
            nb_weekend = st.selectbox("Weekend", [False, True],
                                      key="nb_weekend")
        nb_submit = st.form_submit_button("Predict")

    if nb_submit:
        new_data = pd.DataFrame([{
            'Administrative': nb_admin,
            'Administrative_Duration': nb_admin_dur,
            'Informational': nb_info,
            'Informational_Duration': nb_info_dur,
            'ProductRelated': nb_prod,
            'ProductRelated_Duration': nb_prod_dur,
            'BounceRates': nb_bounce,
            'ExitRates': nb_exit,
            'PageValues': nb_page,
            'SpecialDay': nb_special,
            'Month': nb_month,
            'OperatingSystems': nb_os,
            'Browser': nb_browser,
            'Region': nb_region,
            'TrafficType': nb_traffic,
            'VisitorType': nb_visitor,
            'Weekend': nb_weekend
        }])
        for col in ['Month', 'VisitorType', 'Weekend']:
            if col in label_encoders_2:
                new_data[col] = label_encoders_2[col].transform(new_data[col])
        prediction = nb_classifier.predict(new_data)[0]
        label = "Will Purchase" if prediction == 1 else "No Purchase"
        st.success(f"### Predicted: {int(prediction)} ({label})")

# ============================================================================
# Tab 3: KNN Classifier - Late Delivery Risk
# ============================================================================
with tab3:
    st.header("Late Delivery Risk Prediction (KNN)")
    with st.form("knn_form"):
        knn_real = st.number_input("Days for Shipping (Real)", value=3,
                                   key="knn_real")
        knn_sched = st.number_input("Days for Shipment (Scheduled)", value=4,
                                    key="knn_sched")
        knn_qty = st.number_input("Order Item Quantity", value=1,
                                  key="knn_qty")
        knn_sales = st.number_input("Sales", value=250.0, key="knn_sales")
        knn_profit = st.number_input("Order Profit Per Order", value=64.17,
                                     key="knn_profit")
        knn_mode = st.selectbox("Shipping Mode",
                                ["Standard Class", "Second Class",
                                 "First Class", "Same Day"],
                                key="knn_mode")
        knn_submit = st.form_submit_button("Predict")

    if knn_submit:
        new_data = pd.DataFrame([{
            'Days for shipping (real)': knn_real,
            'Days for shipment (scheduled)': knn_sched,
            'Order Item Quantity': knn_qty,
            'Sales': knn_sales,
            'Order Profit Per Order': knn_profit,
            'Shipping Mode': knn_mode
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
        label = "Late Delivery" if prediction == 1 else "On-time"
        st.success(f"### Predicted: {int(prediction)} ({label})")

# ============================================================================
# Tab 4: SVM Classifier - Online Shopper Purchase Intention
# ============================================================================
with tab4:
    st.header("Online Shopper Purchase Intention (SVM)")
    with st.form("svm_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            svm_admin = st.number_input("Administrative", value=0,
                                        key="svm_admin")
            svm_info = st.number_input("Informational", value=0,
                                       key="svm_info")
            svm_prod = st.number_input("ProductRelated", value=1,
                                       key="svm_prod")
            svm_bounce = st.number_input("BounceRates", value=0.02,
                                         format="%.4f", key="svm_bounce")
            svm_page = st.number_input("PageValues", value=0.0,
                                       key="svm_page")
            svm_os = st.number_input("OperatingSystems", value=1,
                                     key="svm_os")
        with col2:
            svm_admin_dur = st.number_input("Admin Duration", value=0.0,
                                            key="svm_admin_dur")
            svm_info_dur = st.number_input("Info Duration", value=0.0,
                                           key="svm_info_dur")
            svm_prod_dur = st.number_input("ProductRelated Duration", value=0.0,
                                           key="svm_prod_dur")
            svm_exit = st.number_input("ExitRates", value=0.05, format="%.4f",
                                       key="svm_exit")
            svm_special = st.number_input("SpecialDay", value=0.0,
                                          key="svm_special")
            svm_browser = st.number_input("Browser", value=1,
                                          key="svm_browser")
        with col3:
            svm_month = st.selectbox("Month",
                                     ["Jan", "Feb", "Mar", "May", "June",
                                      "Jul", "Aug", "Sep", "Oct", "Nov",
                                      "Dec"],
                                     key="svm_month")
            svm_region = st.number_input("Region", value=1, key="svm_region")
            svm_traffic = st.number_input("TrafficType", value=1,
                                          key="svm_traffic")
            svm_visitor = st.selectbox("VisitorType",
                                       ["Returning_Visitor", "New_Visitor",
                                        "Other"],
                                       key="svm_visitor")
            svm_weekend = st.selectbox("Weekend", [False, True],
                                       key="svm_weekend")
        svm_submit = st.form_submit_button("Predict")

    if svm_submit:
        new_data = pd.DataFrame([{
            'Administrative': svm_admin,
            'Administrative_Duration': svm_admin_dur,
            'Informational': svm_info,
            'Informational_Duration': svm_info_dur,
            'ProductRelated': svm_prod,
            'ProductRelated_Duration': svm_prod_dur,
            'BounceRates': svm_bounce,
            'ExitRates': svm_exit,
            'PageValues': svm_page,
            'SpecialDay': svm_special,
            'Month': svm_month,
            'OperatingSystems': svm_os,
            'Browser': svm_browser,
            'Region': svm_region,
            'TrafficType': svm_traffic,
            'VisitorType': svm_visitor,
            'Weekend': svm_weekend
        }])
        for col in ['Month', 'VisitorType', 'Weekend']:
            if col in label_encoders_4:
                new_data[col] = label_encoders_4[col].transform(new_data[col])
        new_data_scaled = scaler_4.transform(new_data)
        prediction = svm_classifier.predict(new_data_scaled)[0]
        label = "Will Purchase" if prediction == 1 else "No Purchase"
        st.success(f"### Predicted: {int(prediction)} ({label})")

# ============================================================================
# Tab 5: Random Forest Classifier - Online Shopper Purchase Intention
# ============================================================================
with tab5:
    st.header("Online Shopper Purchase Intention (Random Forest)")
    with st.form("rf_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_admin = st.number_input("Administrative", value=0,
                                       key="rf_admin")
            rf_info = st.number_input("Informational", value=0,
                                      key="rf_info")
            rf_prod = st.number_input("ProductRelated", value=1,
                                      key="rf_prod")
            rf_bounce = st.number_input("BounceRates", value=0.02,
                                        format="%.4f", key="rf_bounce")
            rf_page = st.number_input("PageValues", value=0.0,
                                      key="rf_page")
            rf_os = st.number_input("OperatingSystems", value=1,
                                    key="rf_os")
        with col2:
            rf_admin_dur = st.number_input("Admin Duration", value=0.0,
                                           key="rf_admin_dur")
            rf_info_dur = st.number_input("Info Duration", value=0.0,
                                          key="rf_info_dur")
            rf_prod_dur = st.number_input("ProductRelated Duration", value=0.0,
                                          key="rf_prod_dur")
            rf_exit = st.number_input("ExitRates", value=0.05, format="%.4f",
                                      key="rf_exit")
            rf_special = st.number_input("SpecialDay", value=0.0,
                                         key="rf_special")
            rf_browser = st.number_input("Browser", value=1, key="rf_browser")
        with col3:
            rf_month = st.selectbox("Month",
                                    ["Jan", "Feb", "Mar", "May", "June",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                    key="rf_month")
            rf_region = st.number_input("Region", value=1, key="rf_region")
            rf_traffic = st.number_input("TrafficType", value=1,
                                         key="rf_traffic")
            rf_visitor = st.selectbox("VisitorType",
                                      ["Returning_Visitor", "New_Visitor",
                                       "Other"],
                                      key="rf_visitor")
            rf_weekend = st.selectbox("Weekend", [False, True],
                                      key="rf_weekend")
        rf_submit = st.form_submit_button("Predict")

    if rf_submit:
        new_data = pd.DataFrame([{
            'Administrative': rf_admin,
            'Administrative_Duration': rf_admin_dur,
            'Informational': rf_info,
            'Informational_Duration': rf_info_dur,
            'ProductRelated': rf_prod,
            'ProductRelated_Duration': rf_prod_dur,
            'BounceRates': rf_bounce,
            'ExitRates': rf_exit,
            'PageValues': rf_page,
            'SpecialDay': rf_special,
            'Month': rf_month,
            'OperatingSystems': rf_os,
            'Browser': rf_browser,
            'Region': rf_region,
            'TrafficType': rf_traffic,
            'VisitorType': rf_visitor,
            'Weekend': rf_weekend
        }])
        for col in ['Month', 'VisitorType', 'Weekend']:
            if col in label_encoders_5:
                new_data[col] = label_encoders_5[col].transform(new_data[col])
        new_data_scaled = scaler_5.transform(new_data)
        prediction = rf_classifier.predict(new_data_scaled)[0]
        label = "Will Purchase" if prediction == 1 else "No Purchase"
        st.success(f"### Predicted: {int(prediction)} ({label})")

# ============================================================================
# Tab 6: Product Recommender - Association Rules
# ============================================================================
with tab6:
    st.header("Grocery Product Recommender (Association Rules)")
    st.write("Enter products separated by commas. "
             "Examples: whole milk, other vegetables, yogurt")
    with st.form("rec_form"):
        rec_input = st.text_input("Products",
                                  value="whole milk, other vegetables")
        rec_submit = st.form_submit_button("Get Recommendations")

    if rec_submit:
        input_products = set(
            [p.strip().lower() for p in rec_input.split(",") if p.strip()]
        )
        if not input_products:
            st.error("Please enter at least one product.")
        else:
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
                                'Recommended Product': product,
                                'Confidence': round(rule['confidence'], 4),
                                'Lift': round(rule['lift'], 4),
                                'Based On': ', '.join(antecedents)
                            })

            seen = set()
            unique = []
            for rec in sorted(recommendations, key=lambda x: x['Lift'],
                              reverse=True):
                if rec['Recommended Product'] not in seen:
                    seen.add(rec['Recommended Product'])
                    unique.append(rec)

            if unique:
                st.success(f"Found {len(unique)} recommendation(s)!")
                st.dataframe(pd.DataFrame(unique))
            else:
                st.warning("No recommendations found. "
                           "Try: whole milk, other vegetables")
