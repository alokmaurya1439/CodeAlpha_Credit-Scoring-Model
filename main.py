import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

#gi--- 1. Data Generation (Simulated Data) ---
# In a real project, you would load your dataset here (e.g., pd.read_csv('your_credit_data.csv'))

np.random.seed(42) # for reproducibility

n_samples = 1000

data = {
    'income': np.random.normal(50000, 15000, n_samples).round(2),
    'debt': np.random.normal(15000, 7000, n_samples).round(2),
    'age': np.random.randint(20, 70, n_samples),
    'loan_amount': np.random.normal(10000, 5000, n_samples).round(2),
    'credit_score_raw': np.random.normal(650, 80, n_samples).round(0), # Simulate a raw score
    'payment_history_late_days': np.random.randint(0, 30, n_samples), # Average late days
    'num_credit_cards': np.random.randint(0, 5, n_samples),
    'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n_samples, p=[0.6, 0.3, 0.1]),
    'loan_purpose': np.random.choice(['Home', 'Car', 'Education', 'Debt Consolidation', 'Other'], n_samples, p=[0.2, 0.2, 0.1, 0.3, 0.2]),
    'default': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

# Ensure 'income' and 'debt' are not negative
df['income'] = df['income'].apply(lambda x: max(10000, x))
df['debt'] = df['debt'].apply(lambda x: max(0, x))

# Create a more realistic 'default' target based on some features
# People with lower income, higher debt, worse payment history, and lower raw credit scores are more likely to default.
df['default'] = 0 # Start with everyone not defaulting
df.loc[(df['income'] < 40000) & (df['debt'] > 20000), 'default'] = 1
df.loc[df['credit_score_raw'] < 600, 'default'] = 1
df.loc[df['payment_history_late_days'] > 15, 'default'] = 1
df.loc[df['employment_type'] == 'Unemployed', 'default'] = 1
df.loc[(df['default'] == 1) & (np.random.rand(len(df)) < 0.3), 'default'] = 0 # Add some randomness back
df.loc[(df['default'] == 0) & (np.random.rand(len(df)) < 0.05), 'default'] = 1 # Add some 'good' people defaulting randomly

# Let's make sure there's a reasonable balance, though it might still be imbalanced
print(f"Initial 'default' distribution:\n{df['default'].value_counts(normalize=True)}\n")


# --- 2. Feature Engineering ---
# Example: Debt-to-Income Ratio, Credit Utilization Estimate
df['debt_to_income_ratio'] = df['debt'] / df['income']
df['estimated_credit_utilization'] = (df['num_credit_cards'] * 2000 + df['loan_amount']) / (df['num_credit_cards'] * 5000 + df['loan_amount'] + 1) # Simplified estimate
df['age_loan_ratio'] = df['age'] / (df['loan_amount'] + 1)


# --- 3. Data Preprocessing ---

# Define target variable and features
X = df.drop('default', axis=1)
y = df['default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Train set 'default' distribution:\n{y_train.value_counts(normalize=True)}\n")
print(f"Test set 'default' distribution:\n{y_test.value_counts(normalize=True)}\n")


# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. Model Training ---

# Create the full pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'))])
                                 # 'class_weight='balanced'' can help with imbalanced datasets

# Train the model
model_pipeline.fit(X_train, y_train)

# --- 5. Model Evaluation ---

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of defaulting

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Non-Default', 'Predicted Default'],
            yticklabels=['Actual Non-Default', 'Actual Default'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- Model Interpretation (Logistic Regression Coefficients) ---")
# Accessing coefficients and feature names after one-hot encoding
# This part is a bit tricky with pipelines, needs to extract information carefully

# Get trained OneHotEncoder from the pipeline
ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
feature_names_ohe = ohe.get_feature_names_out(categorical_features)

# Combine original numerical features with new one-hot encoded feature names
all_feature_names = numerical_features + list(feature_names_ohe)

# Get coefficients from the Logistic Regression model
coefficients = model_pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame for better visualization
coef_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefficients})
coef_df['Absolute_Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False).reset_index(drop=True)

print(coef_df)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(15)) # Show top 15 features by magnitude
plt.title('Logistic Regression Feature Coefficients (Top 15 by Magnitude)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n--- Example Prediction ---")
# Create a hypothetical new applicant
new_applicant_data = pd.DataFrame([{
    'income': 70000,
    'debt': 10000,
    'age': 35,
    'loan_amount': 5000,
    'credit_score_raw': 720,
    'payment_history_late_days': 2,
    'num_credit_cards': 3,
    'employment_type': 'Salaried',
    'loan_purpose': 'Home'
}])

# Remember to apply the same feature engineering steps to new data
new_applicant_data['debt_to_income_ratio'] = new_applicant_data['debt'] / new_applicant_data['income']
new_applicant_data['estimated_credit_utilization'] = (new_applicant_data['num_credit_cards'] * 2000 + new_applicant_data['loan_amount']) / \
                                                     (new_applicant_data['num_credit_cards'] * 5000 + new_applicant_data['loan_amount'] + 1)
new_applicant_data['age_loan_ratio'] = new_applicant_data['age'] / (new_applicant_data['loan_amount'] + 1)

# Predict creditworthiness
prediction = model_pipeline.predict(new_applicant_data)
prediction_proba = model_pipeline.predict_proba(new_applicant_data)[:, 1]

if prediction[0] == 0:
    print(f"\nThe new applicant is predicted as 'Creditworthy' (No Default Risk).")
else:
    print(f"\nThe new applicant is predicted as 'Not Creditworthy' (High Default Risk).")

print(f"Probability of Default: {prediction_proba[0]:.4f}")
