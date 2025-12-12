import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#Download the model from the model hub
model_path = hf_hub_download(repo_id= "rojasnath/tourism-package-model", filename="best_model_v1.joblib")

#Load the model
model = joblib.load(model_path)

#Streamlit UI for Customer Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("Tourism Package Purchase Prediction App is an internal tool for Visit With Us staff that predicts whether a customer will purchase the new Wellness Tourism Package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the package.")

#Collect user input
Age= st.number_input("Age (customer's age in years)", min_value=18, max_value=120, value=30)
TypeofContact= st.selectbox("How did the customer contact?", ["Company Invited", "Self Inquiry"])
CityTier= st.selectbox("Customer's City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation= st.selectbox("Customer's Occupation", ["Salaried", "Freelancer"])
Gender= st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting= st.number_input("Total number of adult visitors", min_value=1, max_value=20, value=2)
PreferredPropertyStar= st.number_input("Preferred hotel rating", min_value=3, max_value=5, value=4)
MaritalStatus= st.selectbox("Marital status", ["Single", "Married", "Divorced"])
NumberOfTrips= st.number_input("Average number of trips in a year", min_value=0, max_value=15, value=2)
Passport= st.selectbox("Valid passport holder?", ["Yes", "No"])
OwnCar= st.selectbox("Is customer a car owner?", ["Yes", "No"])
NumberOfChildrenVisiting= st.number_input("Number of children below 5 years age", min_value=0, max_value=10, value=2)
Designation= st.selectbox("Customer's designation in their current organization", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome= st.number_input("Gross monthly income of the customer", min_value=5000, max_value=50000, value=15000)
PitchSatisfactionScore= st.number_input("Customer Satisfaction Score (of the sales pitch)", min_value=1, max_value=5, value=5)
ProductPitched= st.selectbox("Type of product pitched to the customer",["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups= st.number_input("Total number of follow-ups by the salesperson", min_value=0, max_value=5, value=2)
DurationOfPitch= st.number_input("Duration of the sales pitch (in mins)", min_value=5, max_value=50, value=15)

#Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

#Set the classification threshold
classification_threshold = 0.45

#Make prediction
if st.button("Predict"):
  prediction_proba = model.predict_proba(input_data)[0, 1]
  prediction = (prediction_proba >= classification_threshold).astype(int)
  result = "purchase" if prediction == 1 else "not purchase"
  st.write(f"Based on the information provided, the customer is likely to {result}.")
