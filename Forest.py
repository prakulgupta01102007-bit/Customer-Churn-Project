from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "customer_churn_dataset-training-master.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.dropna(axis =0)

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
print(df.duplicated().sum())
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs = -1
)
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train:", accuracy_score(y_train, train_pred))
print("Test:", accuracy_score(y_test, test_pred))
from sklearn.tree import export_text

tree_rules = export_text(
    model.estimators_[0],
    feature_names=list(x.columns)
)

print(tree_rules[:3000])

print(df.corr(numeric_only=True)["churn"].sort_values())
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# print(importance_df)
joblib.dump(model,"Fr.joblib")