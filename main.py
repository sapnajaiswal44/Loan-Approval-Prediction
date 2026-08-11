import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('loan_data_new.csv')
# Strip any hidden spaces from column names
df.columns = df.columns.str.strip()

print("\n--- First 5 Rows ---")
print(df.head())

# 2. Exploratory Data Analysis & Data Cleaning
print("\n--- Dataset Info ---")
print(df.info())

# Drop duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
df = df.drop_duplicates()

# Fill missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Categorical Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Identify Target Column
target_col = None
possible_targets = ['loan_status', 'Loan_Status', 'loan_status_encoded', 'status']
for target in possible_targets:
    if target in df.columns:
        target_col = target
        break

if not target_col:
    target_col = df.columns[-1]

print(f"\nTarget column selected: '{target_col}'")

# Split Features (X) and Target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Models
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluate Models
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n=== {name} Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

dt_metrics = evaluate_model(dt_model, "Decision Tree")
rf_metrics = evaluate_model(rf_model, "Random Forest")

# 6. Compare & Conclude
comparison_df = pd.DataFrame([dt_metrics, rf_metrics], index=['Decision Tree', 'Random Forest'])
print("\n--- Final Performance Comparison ---")
print(comparison_df)

best_model = comparison_df['F1-Score'].idxmax()
print(f"\nConclusion: Based on F1-Score, '{best_model}' is the better-performing algorithm for this dataset.")