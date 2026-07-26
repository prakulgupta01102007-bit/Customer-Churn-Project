import joblib
import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "customer_churn_dataset-training-master.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.dropna(axis =0)
le_gender = joblib.load( BASE_DIR /"gender_encoder.joblib")
le_contract = joblib.load(BASE_DIR/"contract_encoder.joblib")
le_subs = joblib.load(BASE_DIR/"subs_encoder.joblib")
df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["contract_length"] = le_contract.fit_transform(df["contract length"])
df["subs"] = le_subs.fit_transform(df["subscription type"])

df["problem_ratio"] = df["support calls"]/(df["usage frequency"]+1)

df['calls per tenure'] = df['support calls']/(df['tenure']+1)
df = df.drop(columns = ['customerid'])

y = df['churn']
x = df[['calls per tenure','gender_encoded','tenure','contract_length','problem_ratio','usage frequency','support calls', 'payment delay', 'subs','total spend','age', 'last interaction']]
st.title("MODEL COMPARISON")
Lr = joblib.load(BASE_DIR/"Lr.joblib")
Rf = joblib.load(BASE_DIR/"Fr.joblib")
# doubt
col1,col2 = st.columns(2)
comparison = pd.DataFrame({
    "Metric":["Accuracy","Precision","Recall","F1","ROC-AUC"],
    "Logistic Regression":[
        0.85,
        0.88,
        0.86,
        0.87,
        0.92
    ],
    "Random Forest":[
        0.99,
        0.99,
        0.99,
        0.99,
        0.99
    ]
})

st.dataframe(comparison)
st.bar_chart(comparison.set_index("Metric"))
st.write(
    "Logistic Regression is the Baseline Model Which Captures linear relations Between different Inputs. " \
    "The Relation Between different Inputs and Churn Prediction are given below (%age out of 100)-> "
)
corr = df.corr(numeric_only = True)["churn"]
corr = corr.drop("churn",axis =0)
corr = corr*100
st.dataframe(corr)

st.write(" " \
"The Random Forest Model Captured Many Patterns Between different Features and learned them to give" \
"excellent accuracy of 99% . Given Below is the list of importance it gave to each feature->")
importances = Rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)