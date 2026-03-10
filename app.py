import streamlit as st
import pandas as pd
from joblib import load

# --- Background Image ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t3.ftcdn.net/jpg/12/96/43/92/360_F_1296439246_C65lAcWsM7L5qs2lNT6CCuZHZXEIIe4a.jpg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return load("Final_Linear_Model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Page Config ---
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Predict car prices using a trained Linear Regression model.")

# --- Dropdown Options ---
makes = ["Maruti", "Hyundai", "Tata", "Honda", "Mahindra"]
models_dict = {
    "Maruti": ["Swift", "Baleno", "Dzire"],
    "Hyundai": ["i20", "Verna", "Creta"],
    "Tata": ["Nexon", "Altroz", "Tiago"],
    "Honda": ["City", "Amaze", "WR-V"],
    "Mahindra": ["Thar", "XUV500", "Scorpio"]
}
fuel_types = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
transmissions = ["Manual", "Automatic"]
locations = ["Mumbai", "Pune", "Delhi", "Bangalore", "Chennai"]
owners = ["First", "Second", "Third", "Fourth & Above"]
seller_types = ["Dealer", "Individual", "Trustmark Dealer"]
drivetrains = ["FWD", "RWD", "AWD", "4WD"]

owner_map = {"First": 0, "Second": 1, "Third": 2, "Fourth & Above": 3}

# --- Columns Layout ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    make = st.selectbox("Car Make", makes)
    year = st.number_input("Year of Manufacture", 1990, 2025, 2018)
    kilometer = st.number_input("Kilometers Driven", 0, 500000, 50000)
    fuel_type = st.selectbox("Fuel Type", fuel_types)
    transmission = st.selectbox("Transmission Type", transmissions)

with col2:
    model_name = st.selectbox("Car Model", models_dict[make])
    location = st.selectbox("Location", locations)
    owner = st.selectbox("Owner Type", owners)
    seller_type = st.selectbox("Seller Type", seller_types)
    drivetrain = st.selectbox("Drivetrain", drivetrains)

with col3:
    length = st.number_input("Length (mm)", 3000, 6000, 4000)
    width = st.number_input("Width (mm)", 1200, 2500, 1700)
    height = st.number_input("Height (mm)", 1200, 2500, 1500)
    seating_capacity = st.slider("Seating Capacity", 2, 10, 5)
    fuel_tank = st.number_input("Fuel Tank Capacity (Litres)", 20, 120, 45)

with col4:
    engine = st.number_input("Engine (CC)", 600, 6000, 1500)
    max_power = st.number_input("Max Power (BHP)", 30, 600, 80)
    max_rpm = st.number_input("Max Power (RPM)", 1000, 10000, 6000)
    st.write("")  # empty space for alignment

# --- Prepare Input Data ---
input_data = pd.DataFrame({
    "Make": [make],
    "Model": [model_name],
    "Kilometer": [kilometer],
    "Fuel Type": [fuel_type],
    "Transmission": [transmission],
    "Location": [location],
    "Owner": [owner_map[owner]],
    "Seller Type": [seller_type],
    "Drivetrain": [drivetrain],
    "Length": [length],
    "Width": [width],
    "Height": [height],
    "Seating Capacity": [seating_capacity],
    "Fuel Tank Capacity": [fuel_tank],
    "Car_Age": [2025 - year],
    "Engine (cc)": [engine],
    "Max Power (BHP)": [max_power],
    "Max Power (RPM)": [max_rpm],
})

# Encode categorical columns
cat_cols = ["Make", "Model", "Fuel Type", "Transmission", "Location", "Seller Type", "Drivetrain"]
for col in cat_cols:
    input_data[col] = input_data[col].astype("category").cat.codes

st.write("### Input Summary")
st.dataframe(input_data)

# --- Predict ---
if st.button("Predict Car Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
