from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import joblib
import shap 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "customer_churn_dataset-training-master.csv")
df.columns = df.columns.str.strip().str.lower()
print(df.head())
print(df.count().sum())
df = df.dropna(axis =0)
print(df.count().sum())

le_gender = LabelEncoder()
le_contract = LabelEncoder()
le_subs = LabelEncoder()
df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["contract_length"] = le_contract.fit_transform(df["contract length"])
df["subs"] = le_subs.fit_transform(df["subscription type"])

df["problem_ratio"] = df["support calls"]/(df["usage frequency"]+1)

df['calls per tenure'] = df['support calls']/(df['tenure']+1)


df = df.drop(columns = ['customerid'])
y = df['churn']
x = df[['calls per tenure','gender_encoded','tenure','contract_length','problem_ratio','usage frequency','support calls', 'payment delay', 'subs','total spend','age', 'last interaction']]

x_train , x_test , y_train , y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x = scaler.transform(x)

model = LogisticRegression(max_iter=1000)
print(y.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
probs = model.predict_proba(x_test)
print(probs[:10])
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="accuracy"
)

print(scores)
print("Average:", scores.mean())

scores1 = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="roc_auc"
)
print(scores1.mean())

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
explainer = shap.Explainer(
    model,
    x
)
x_head = x.head(1000)
shap_values = explainer(x_head)

joblib.dump(model,"Lr.joblib")
joblib.dump(scaler,"Lrscaler.joblib")
joblib.dump(le_gender, "gender_encoder.joblib")
joblib.dump(le_contract, "contract_encoder.joblib")
joblib.dump(le_subs, "subs_encoder.joblib")
joblib.dump(explainer,"leexplain.joblib")
