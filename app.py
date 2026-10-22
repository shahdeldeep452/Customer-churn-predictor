import __main__
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Same class used when training. Needed so the saved pipeline can be loaded."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).values
        self.upper_ = (q3 + self.factor * iqr).values
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


__main__.IQRCapper = IQRCapper   # the model was saved from a notebook, where the class lived in __main__

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("churn_model.joblib")


art = load_artifacts()
model = art["model"]
preprocessor = art["preprocessor"]
feature_names = art["feature_names"]
selected = art["selected_features"]
input_cols = art["input_columns"]


def tenure_group(t):
    if t <= 12:
        return "0-12"
    if t <= 24:
        return "13-24"
    if t <= 48:
        return "25-48"
    return "49-72"


st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's details to estimate the probability that they will leave.")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model performance"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        monthly = st.number_input("Monthly charges ($)", 15.0, 150.0, 70.0, 1.0)
        internet = st.selectbox("Internet service", ["Fiber", "DSL", "No internet"])
        tickets = st.number_input("Support tickets", 0, 15, 1)
        partner = st.selectbox("Partner", ["Yes", "No"])
    with c3:
        tech = st.selectbox("Tech support", ["Yes", "No"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])

    total = round(monthly * tenure, 2)
    row = {
        "gender": gender, "senior_citizen": int(senior == "Yes"), "partner": partner,
        "tenure_months": tenure, "contract": contract, "internet_service": internet,
        "tech_support": tech, "payment_method": payment, "paperless_billing": paperless,
        "monthly_charges": monthly, "total_charges": total, "support_tickets": tickets,
        "avg_monthly_spend": total / tenure, "tickets_per_month": tickets / tenure,
        "high_support_user": int(tickets >= 3), "tenure_group": tenure_group(tenure),
        "has_internet": int(internet != "No internet"),
    }
    X_in = pd.DataFrame([row])[input_cols]

    if st.button("Predict churn risk", type="primary"):
        X_proc = pd.DataFrame(preprocessor.transform(X_in), columns=feature_names)[selected]
        p = float(model.predict_proba(X_proc)[0, 1])

        if p >= 0.6:
            level, color = "High", "#C44E52"
        elif p >= 0.35:
            level, color = "Medium", "#E8A33D"
        else:
            level, color = "Low", "#55A868"

        g, m = st.columns(2)
        with g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100, number={"suffix": "%"},
                title={"text": "Churn probability"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 35], "color": "#E6F4EA"},
                                 {"range": [35, 60], "color": "#FDF1DC"},
                                 {"range": [60, 100], "color": "#F9E0E1"}]}))
            fig.update_layout(height=300, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width="stretch")
        with m:
            st.subheader(f"Risk level: {level}")
            tips = []
            if contract == "Month-to-month":
                tips.append("Offer a discount for moving to a 1- or 2-year contract.")
            if payment == "Electronic check":
                tips.append("Encourage automatic payment (card or bank transfer).")
            if tickets >= 3:
                tips.append("Many support tickets: proactive follow-up call.")
            if tech == "No" and internet != "No internet":
                tips.append("Offer a free trial of tech support.")
            if tenure <= 6:
                tips.append("New customer: run an onboarding campaign.")
            st.markdown("**Suggested retention actions**")
            for t in (tips or ["No specific risk factors found. Keep monitoring."]):
                st.write("• " + t)

with tab2:
    st.subheader("Model comparison (test set)")
    if Path("model_comparison.csv").exists():
        st.dataframe(pd.read_csv("model_comparison.csv"), hide_index=True, width="stretch")
    if Path("model_evaluation.png").exists():
        st.image("model_evaluation.png", caption="Confusion matrix and ROC curves")
    st.caption("Recall shows how many real churners the model catches. Accuracy is not used because only ~31% of customers churn.")
