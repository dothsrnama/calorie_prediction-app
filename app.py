# importing libraries
import streamlit as st
import joblib
import pandas as pd


# page tab title and modificaitons.
st.set_page_config(
	page_title="Calorie Burn Predictor",
	page_icon = "🔥",
	layout="centered")

# writing a title and short description on page.
st.title("🔥 Calorie Burn Predictor")
st.write("Input your activity metrics below and estimate calories burned using our **ML model**")
# LOADING MODEL AND SCALER DIRECTLY
@st.cache_resource
def load_artifacts():
	model=joblib.load("model_artifacts/calorie_model.pkl")
	scaler = joblib.load("model_artifacts/scaler.pkl")
	return model, scaler
model, scaler = load_artifacts()
# sidebar controls for user demograpgic inputs
st.sidebar.header("User Info")

gender_selection = st.sidebar.selectbox("Gender",options=['Female','Male'])
gender = 1 if gender_selection == 'Male' else 0
age = st.sidebar.slider("Age (yrs)", min_value=18, max_value=80, value=25)
height = st.sidebar.slider("Height (cm)", min_value=120.0, max_value=220.0, value = 170.0)
weight = st.sidebar.slider("Weight (kgs)", min_value=40.0, max_value = 150.0, value = 170.0)

st.header("Workout Metrics")
duration = st.number_input("Workout Duration (mins)", min_value=1.0, max_value=300.0, value=30.0)
heart_rate = st.number_input("Average Heart Rate (bpm)",min_value=60.0, max_value=200.0, value= 130.0)
body_temp = st.number_input("Body Temp (Celcius)", min_value=35.0, max_value=42.0, value=38.5)

# Defining API target and submint button.
API_URL = "http://127.0.0.1:8000/predict"

if st.button("Predict Calories Burned", type="primary"):
	payload = {
		"Gender":gender,
		"Age":age,
		"Height":height,
		"Weight":weight,
		"Duration":duration,
		"Heart_Rate": heart_rate,
		"Body_Temp":body_temp}
	# creating a loading spinner animation
	with st.spinner("Calculating.."):
		try:
			response = requests.post(API_URL, json=payload)
			if response.status_code == 200:
				result = response.json()
				calories = result.get("predicted_calories")
				st.success(f"Estimated Calories Burned: **{calories} kcal**")
			else:
				st.error(f"Error from API: {response.text}")
		except Exception as e:
			st.error(f"Could not connect to FastAPI server. Make sure it is running on port 8000! Error:{e}")
 
