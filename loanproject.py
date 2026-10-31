import pandas as pd
import numpy as np
import os
import joblib
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Generate dataset if it doesn't exist yet
csv_path = 'data/loan_data.csv'
if not os.path.exists(csv_path):
    np.random.seed(42)
    n_samples = 1000
    data = {
        'Income': np.random.randint(20000, 150000, n_samples),
        'LoanAmount': np.random.randint(5000, 40000, n_samples),
        'CreditScore': np.random.randint(500, 850, n_samples),
        'EmploymentStatus': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'LoanApproved': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    }
    df_temp = pd.DataFrame(data)
    df_temp.loc[10:15, 'Income'] = np.nan
    df_temp.loc[20:22, 'CreditScore'] = np.nan
    df_temp = pd.concat([df_temp, df_temp.iloc[:5]], ignore_index=True)
    df_temp.to_csv(csv_path, index=False)
    print("Generated missing loan_data.csv successfully!")

# 1. Load Data
df = pd.read_csv(csv_path)
print(f"Initial Dataset Shape: {df.shape}")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load Data
df = pd.read_csv('data/loan_data.csv')
print(f"Initial Dataset Shape: {df.shape}")

# 2. Data Processing (Requirement 1)
# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Separate Features and Target
X = df.drop(columns=['LoanApproved'])
y = df['LoanApproved']

# Identify numeric and categorical columns
numeric_features = ['Income', 'LoanAmount', 'CreditScore']
categorical_features = ['EmploymentStatus', 'Education']

# Define preprocessing for numerical features (impute missing with median, scale features)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (impute missing with most frequent, one-hot encode)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Development (Requirement 2: Train at least 4 models and compare)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

best_model_name = None
best_accuracy = 0
best_pipeline = None

print("\n--- Model Evaluation & Comparison ---")
for name, model in models.items():
    # Create a full pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    # Report metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    
    print(f"\nModel: {name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_pipeline = pipeline

print(f"\n🏆 Best Model Selected: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# 4. Hyperparameter Tuning (Requirement 3)
print("\n--- Performing Hyperparameter Tuning on Best Model ---")
if best_model_name == "Random Forest":
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10]
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1]
    }
else:
    param_grid = {
        'classifier__C': [0.1, 1, 10]
    }

grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_
tuned_acc = accuracy_score(y_test, tuned_model.predict(X_test))
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Tuned Model Accuracy: {tuned_acc:.4f}")

# 5. Model Saving (Requirement 4)
joblib.dump(tuned_model, 'models/best_model.pkl')
print("\nModel successfully saved to 'models/best_model.pkl' using joblib!")