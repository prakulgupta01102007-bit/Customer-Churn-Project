import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing import StandardScaler
import streamlit as st
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "customer_churn_dataset-training-master.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.dropna(axis =0)
le_gender = joblib.load(BASE_DIR /"gender_encoder.joblib")
le_contract = joblib.load(BASE_DIR /"contract_encoder.joblib")
le_subs = joblib.load(BASE_DIR /"subs_encoder.joblib")
df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["contract_length"] = le_contract.fit_transform(df["contract length"])
df["subs"] = le_subs.fit_transform(df["subscription type"])

df["problem_ratio"] = df["support calls"]/(df["usage frequency"]+1)

df['calls per tenure'] = df['support calls']/(df['tenure']+1)


df = df.drop(columns = ['customerid'])
x = df[['calls per tenure','gender_encoded','tenure','contract_length','problem_ratio','usage frequency','support calls', 'payment delay', 'subs','total spend','age', 'last interaction']]
st.title("CUSTOMER CHURN PREDICTOR")
model_choice = st.selectbox(
    "Choose Model(For Difference Between Models Refer To The Comparison Page )",
    ["Logistic Regression" , "Random Forest"]
)
age = st.number_input("Age")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input("Tenure")
usage_frequency = st.number_input("Usage Frequency")
support_calls = st.number_input("Support Calls")
payment_delay = st.number_input("Payment Delay")
subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)
contract = st.selectbox(
 "Contract Length",
 ["Annual","Monthly","Quarterly"]
)
total_spend = st.number_input("Total Spend")
last_interaction = st.number_input("Last Interaction")
le_gender = joblib.load(BASE_DIR /"gender_encoder.joblib")
le_contract = joblib.load(BASE_DIR /"contract_encoder.joblib")
le_subs = joblib.load(BASE_DIR/"subs_encoder.joblib")
if st.button("Predict Churn"):

    # Encoding
  gender_encoded = le_gender.transform([gender])[0]

  subs = le_subs.transform([subscription_type])[0]

  contract_length = le_contract.transform(
        [contract]
    )[0]


    # Feature Engineering (same as training)
  calls_per_tenure = support_calls/(tenure+1)

  problem_ratio = support_calls/(usage_frequency+1)


    # Creating input dataframe in exact training order
  input_data_unscaled = pd.DataFrame(
        [[
            calls_per_tenure,
            gender_encoded,
            tenure,
            contract_length,
            problem_ratio,
            usage_frequency,
            support_calls,
            payment_delay,
            subs,
            total_spend,
            age,
            last_interaction
        ]]
    )
  input_columns = [
    "calls_per_tenure",
    "gender_encoded",
    "tenure",
    "contract_length",
    "problem_ratio",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "subs",
    "total_spend",
    "age",
    "last_interaction"
]
# # corre = df[cols].corr(numeric_only = True)
# # print(corre)
# # #correlation matrix for feature selection and engineering
# # print(df.head())

  if (model_choice == "Logistic Regression") :
    model = joblib.load(BASE_DIR /"Lr.joblib")
    scaler1 = joblib.load(BASE_DIR/"Lrscaler.joblib")
    input_data = scaler1.transform(input_data_unscaled)
    y = model.predict_proba(input_data)[0][1]
    explainer = shap.LinearExplainer(model,x)
    shap_values = explainer(input_data)  
    sample_shap = shap_values[0].values
    explanation = pd.DataFrame({
      "Feature": input_columns,
      "SHAP Value": sample_shap,
     "Feature Value": input_data[0]
    })
    explanation1 = explanation.sort_values(by = "SHAP Value")
    explanation2 = explanation.sort_values(by = "SHAP Value", ascending= False)
        
    explanation1 = explanation1.head(3)
    explanation2 = explanation2.head(3)
    
    y = np.round(y,5)

    st.write(f"THE PROBABILITY OF CHURNING IS {y*100}%")
    if(y>0.5) :
      st.write("Risk Is High")
    else :
      st.write("Risk Is Low")
    st.write(
      "The Factors Increasing Probability Are ->"
    )
    st.dataframe(explanation2, hide_index=True)
    st.write(
          "The Factors Decreasing Probability Are ->"
        )
    st.dataframe(explanation1, hide_index=True)


  elif(model_choice =="Random Forest"):

    model = joblib.load(BASE_DIR /"Fr.joblib")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data_unscaled)
    feature_values = np.asarray(input_data_unscaled).reshape(-1)
    sample_shap = shap_values[0, :, 1].values
    
    explanation = pd.DataFrame({
     "Feature": input_columns,
     "SHAP Value": sample_shap,
     "Feature Value": feature_values
})
    explanation1 = explanation.sort_values(by = "SHAP Value")
    explanation2 = explanation.sort_values(by = "SHAP Value", ascending= False)
    
    explanation1 = explanation1.head(3)
    explanation2 = explanation2.head(3)
    y = model.predict_proba(input_data_unscaled)[0][1]
    y = np.round(y,5)
    st.write(f"THE PROBABILITY OF CHURNING IS {y*100}%")
    if(y>0.5) :
      st.write("Risk Is High")
    else :
      st.write("Risk Is Low")
    st.write(
          "The Factors Increasing Probability Are ->"
        )
    st.dataframe(explanation2, hide_index=True)
    st.write(
              "The Factors Decreasing Probability Are ->"
            )
    st.dataframe(explanation1, hide_index=True)
    
    
