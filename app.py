import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("House Price India.csv")

    # Convert columns as required
    df['number of bathrooms'] = df['number of bathrooms'].astype(int)
    df['waterfront present'] = df['waterfront present'].astype(int)

    return df

df = load_data()

# Select features and target
features = ['number of bedrooms', 'number of bathrooms', 'living area',
            'lot area', 'waterfront present', 'number of views']

# Fallback in case 'price' column name is different
y = df['price'] if 'price' in df.columns else df.iloc[:, -1]
X = df[features]

# Train the model
@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(X, y)

# Streamlit UI
st.title("🏡 House Price India - Prediction App")
st.markdown("Upload a file or enter values manually to predict house prices.")

st.sidebar.header("Choose Input Type")
mode = st.sidebar.radio("Select mode:", ["Manual Input", "Upload CSV"])

# Manual input form
def manual_input():
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
    living_area = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=20000, value=5000)
    waterfront = st.selectbox("Waterfront Present (0 = No, 1 = Yes)", [0, 1])
    views = st.slider("Number of Views", min_value=0, max_value=10, value=1)

    data = {
        'number of bedrooms': bedrooms,
        'number of bathrooms': bathrooms,
        'living area': living_area,
        'lot area': lot_area,
        'waterfront present': waterfront,
        'number of views': views
    }
    return pd.DataFrame([data])

# Predict function
def predict(df):
    return model.predict(df)

# App logic
if mode == "Manual Input":
    input_df = manual_input()
    if st.button("Predict Price"):
        result = predict(input_df)
        st.success(f"💰 Predicted Price: ₹ {round(result[0], 2)}")
else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)

        # Convert necessary columns
        if 'number of bathrooms' in test_df.columns:
            test_df['number of bathrooms'] = test_df['number of bathrooms'].astype(int)
        if 'waterfront present' in test_df.columns:
            test_df['waterfront present'] = test_df['waterfront present'].astype(int)

        missing = [col for col in features if col not in test_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            preds = predict(test_df[features])
            test_df['Predicted Price'] = preds
            st.write("📊 Prediction Results:", test_df)
            st.download_button("📥 Download CSV", test_df.to_csv(index=False), "predicted_prices.csv")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | LocalTunnel Ready ✅")
