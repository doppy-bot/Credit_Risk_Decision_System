import streamlit as st
import pandas as pd
import shap
import pickle
import os

# -------------------------------
# Title
# -------------------------------
st.set_page_config(page_title="Credit Risk System", layout="wide")
st.title("💳 Credit Risk Decision System")

# -------------------------------
# Load Files (Robust Path)
# -------------------------------
BASE_PATH = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE_PATH, "model.pkl"), "rb"))
data = pd.read_csv(os.path.join(BASE_PATH, "sample_input.csv"))

# -------------------------------
# Display Customer Data
# -------------------------------
st.subheader("📊 Customer Data")
st.write(data)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict_proba(data)[0][1]

st.subheader("📈 Prediction")
st.write(f"**Probability of Default:** {prediction:.2f}")

# -------------------------------
# Decision Logic
# -------------------------------
if prediction < 0.1:
    decision = "✅ Approve (Low Risk)"
elif prediction < 0.25:
    decision = "⚠️ Approve with High Interest"
else:
    decision = "❌ Reject"

st.subheader("💡 Decision")
st.write(decision)

# -------------------------------
# SHAP Explanation
# -------------------------------
st.subheader("🔍 Model Explanation (SHAP)")

try:
    data = data.astype(float)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    # Fix for Streamlit plotting
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    shap.plots.waterfall(shap_values[0])

except Exception as e:
    st.warning(f"SHAP visualization error: {e}")