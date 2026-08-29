import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Food_Delivery_Times.csv")

# Remove Order_ID
if "Order_ID" in df.columns:
    df = df.drop("Order_ID", axis=1)

target_col = "Delivery_Time_min"

feature_cols = [col for col in df.columns if col != target_col]

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;color:#FF4B4B'>
    🍕 Food Delivery Time Prediction System
    </h1>
    <p style='text-align:center'>
    Predict estimated delivery time using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Order Details")

col1, col2 = st.columns(2)

user_input = {}

for i, column in enumerate(feature_cols):

    current_col = col1 if i % 2 == 0 else col2

    if df[column].dtype == "object":

        options = sorted(df[column].dropna().unique())

        user_input[column] = current_col.selectbox(
            column,
            options
        )

    else:

        user_input[column] = current_col.number_input(
            column,
            value=float(df[column].median())
        )

# ---------------- PREDICT BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Delivery Time", use_container_width=True):

    input_df = pd.DataFrame([user_input])

    # Apply Label Encoding
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    prediction = model.predict(input_df)[0]

    st.success(
        f"⏱ Estimated Delivery Time: **{round(prediction,2)} Minutes**"
    )

    st.metric(
        label="Predicted Delivery Time",
        value=f"{round(prediction,2)} min"
    )

# ---------------- DATA PREVIEW ----------------
with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head())

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit & Random Forest</center>",
    unsafe_allow_html=True
)