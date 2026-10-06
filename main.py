# importing libraries
from fastapi import FastAPI, HTTPException # HTTPException for error handling
from pydantic import BaseModel, Field 
# base model - defining data schemas and validating incoming JSON request 
# payloads.Field - used for validation rules and metadata for API.
import joblib
import pandas as pd
import numpy as np
# rest libraries used for deserializing and data preprocessing

# API initilization and it's metadata

app = FastAPI(
	title = "Calorie Prediction",
	descriptioin="API for predicting burned calories using a ML&DL models.",
	version="1.0.0")

# loading model artifacts.

try:
	model = joblib.load('model_artifacts/calorie_model.pkl')
	scaler = joblib.load('model_artifacts/scaler.pkl')
except Exception as e:
	raise RuntimeError(f"Error loading model artifacts: {e}")

# defining Input data schema and validation rules

class UserActivityInput(BaseModel):
	Gender: int = Field(..., description="0 for Female, 1 for Male") # Data encoding for gender as integer feature.
	Age: int = Field(..., ge=1, le=120, description="Age in years") # age as required interger greater than or equal to 1 anf less than equal to 120.
	Height: float = Field(...,ge=50.0, le=250.0, description="Height in cm")
	Weight: float = Field(...,ge=20.0, le=300.0, description="Weight in kg")
	Duration: float = Field(...,ge=1.0, le=300.0, description="Exercise duration in mins")
	Heart_Rate: float = Field(...,ge=40.0, le=220.0, description="Average heat rate in bpm")
	Body_Temp: float = Field(...,ge=35.0,le=43.0, description="Body temperature in celcius")

# Creating server health checkpoint

@app.get("/")
def health_check():
	return{"status":"healthy","service":"Calorie Prediction API"}

# creating Prediction endpoint logic

@app.post("/predict")
def predict_calories(data: UserActivityInput):
	try:
		input_data = pd.DataFrame([{
			'Gender':data.Gender,
			'Age': data.Age,
			'Height': data.Height,
			'Weight': data.Weight,
			'Duration': data.Duration,
			'Heart_Rate': data.Heart_Rate,
			'Body_Temp' : data.Body_Temp}])

		# scaling the data
		scaled_features = scaler.transform(input_data)
		
		# predicting using hybrid model.
		prediction = model.predict(scaled_features)[0]

		return{
			"predicted_calories": round(float(prediction),2),"unit":"kcal"}
	except Exception as e:
		raise HTTPException(status_code=500, detail = str(e))

