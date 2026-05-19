import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page settings
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Medical Insurance Cost Prediction using SVR")
st.write("Predict insurance cost based on personal details.")

# Load dataset
data = pd.read_csv("medical_insurance.csv")

# Encode categorical columns
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data["sex"] = le_sex.fit_transform(data["sex"])
data["smoker"] = le_smoker.fit_transform(data["smoker"])
data["region"] = le_region.fit_transform(data["region"])

# Split features and target
X = data.drop("charges", axis=1)
y = data["charges"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = SVR(kernel="rbf")

model.fit(X_train_scaled, y_train)

# Sidebar
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=28.5
)

children = st.sidebar.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=1
)

smoker = st.sidebar.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.sidebar.selectbox(
    "Region",
    ["southeast", "southwest", "northeast", "northwest"]
)

# Prediction
if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [le_sex.transform([sex])[0]],
        "bmi": [bmi],
        "children": [children],
        "smoker": [le_smoker.transform([smoker])[0]],
        "region": [le_region.transform([region])[0]]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Insurance Cost: ₹ {prediction[0]:,.2f}"
    )

# Evaluation Metrics
st.subheader("📊 Model Evaluation")

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write(f"MAE : {mae:.2f}")
st.write(f"MSE : {mse:.2f}")
st.write(f"RMSE : {rmse:.2f}")
st.write(f"R2 Score : {r2:.2f}")

# Dataset preview
st.subheader("📁 Dataset Preview")
st.dataframe(data.head())