import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import streamlit as st


df = pd.read_csv("employee_attrition.csv")
print("Dataset Loaded Successfully")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nAttrition Count")
print(df["Attrition"].value_counts())

print("\nTarget Percentage")
print(df["Attrition"].value_counts(normalize=True)*100)

print("\nUnique Values")
print(df.nunique())

print("\nData Types")
print(df.dtypes)

columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=columns_to_drop, inplace=True)
print("\nColumns Removed Successfully")

print("\nUpdated Shape")
print(df.shape)

df.to_csv("clean_employee_attrition.csv", index=False)
print("Clean Dataset Saved Successfully")


plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(7,4))
sns.countplot(x="Department", data=df)
plt.title("Department Distribution")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(
    x="Department",
    hue="Attrition",
    data=df
)
plt.title("Department vs Attrition")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(
    x="OverTime",
    hue="Attrition",
    data=df
)
plt.title("OverTime vs Attrition")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyIncome"], bins=30)
plt.title("Monthly Income")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="JobSatisfaction", data=df)
plt.title("Job Satisfaction")
plt.show()

plt.figure(figsize=(18,12))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["MonthlyIncome"])
plt.title("Monthly Income Outliers")
plt.show()


ml_df = df.copy()

le = LabelEncoder()
ml_df["Attrition"] = le.fit_transform(
    ml_df["Attrition"]
)

categorical_columns = ml_df.select_dtypes(
    include="object"
).columns
print(categorical_columns)

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col])
    encoders[col] = le

X = ml_df.drop(
    "Attrition",
    axis=1
)
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(  
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(
    X_train
)
X_test = scaler.transform(
    X_test
)

models = {
"Logistic Regression":
LogisticRegression(),
"Decision Tree":
DecisionTreeClassifier(),
"Random Forest":
RandomForestClassifier(),
"KNN":
KNeighborsClassifier(),
"Gradient Boosting":
GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(
        X_train,
        y_train
    )
    prediction = model.predict(
        X_test
    )
    accuracy = accuracy_score(
        y_test,
        prediction
    )
    f1 = f1_score(
        y_test,
        prediction
    )
    results[name] = {
        "Accuracy": accuracy,
        "F1 Score": f1
    }
results_df = pd.DataFrame(results).T
print("\nModel Comparison")
print(results_df)

best_model_name = max(
    results,
    key=lambda x: results[x]["F1 Score"]
)
print("\nBest Model:")
print(best_model_name)

best_model = models[best_model_name]

best_model.fit(
    X_train,
    y_train
)
print("Final Model Trained Successfully")

y_pred = best_model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

if hasattr(best_model,"feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )
    print(importance.head(10))
    plt.figure(figsize=(8,5))
    sns.barplot(
        data=importance.head(10),
        x="Importance",
        y="Feature"
    )
    plt.title(
        "Top 10 Important Features"
    )
    plt.show()

joblib.dump(
    best_model,
    "employee_attrition_model.pkl"
)
joblib.dump(
    scaler,
    "scaler.pkl"
)
print("Model Saved Successfully")


model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
joblib.dump(encoders, "encoders.pkl")


X = ml_df.drop("Attrition", axis=1)
print("Features used for training:")
print(X.columns.tolist())































