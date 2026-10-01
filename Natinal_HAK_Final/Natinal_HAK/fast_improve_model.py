import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FAST MODEL ACCURACY IMPROVEMENT")
print("=" * 80)

# Load data
df = pd.read_csv('cleaned_dataset.csv')
print(f"\nDataset loaded: {df.shape}")

# Prepare features
X = df.drop(['student_id', 'dropout'], axis=1)
y = df['dropout']

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'department', 'scholarship', 'parental_education',
                    'extra_curricular', 'sports_participation']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

print("\n" + "=" * 80)
print("TRAINING OPTIMIZED MODELS (Fast version)")
print("=" * 80)

# Model 1: Random Forest with good parameters
print("\n1. Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# Model 2: XGBoost with optimized parameters
print("\n2. Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# Model 3: Gradient Boosting with optimized parameters
print("\n3. Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")

# Model 4: Voting Ensemble
print("\n4. Training Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('gb', gb)
    ],
    voting='soft',
    weights=[2, 2, 1]
)
voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
voting_acc = accuracy_score(y_test, voting_pred)
print(f"Voting Ensemble Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")

# Compare all models
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results = {
    'Random Forest': rf_acc,
    'XGBoost': xgb_acc,
    'Gradient Boosting': gb_acc,
    'Voting Ensemble': voting_acc
}

for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:25s}: {acc:.4f} ({acc*100:.2f}%)")

# Select best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print(f"\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name}")
print(f"ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("=" * 80)

# Get best model object
model_map = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'Gradient Boosting': gb,
    'Voting Ensemble': voting_clf
}

final_model = model_map[best_model_name]

# Detailed metrics
print("\nDetailed Classification Report:")
y_pred_final = final_model.predict(X_test_scaled)
y_pred_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred_final, target_names=['No Dropout', 'Dropout']))

print(f"\nPrecision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_final):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")

# Cross-validation score
print("\n5-Fold Cross-Validation Scores:")
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"Std CV Accuracy: {cv_scores.std():.4f}")

# Save models
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

joblib.dump(final_model, 'models/best_model_improved.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print(f"Best model saved: models/best_model_improved.pkl")
print(f"Model type: {best_model_name}")
print(f"Final Accuracy: {best_accuracy*100:.2f}%")

# Save accuracy report
with open('models/model_accuracy_report.txt', 'w') as f:
    f.write("MODEL ACCURACY REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
    f.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)\n\n")
    f.write("All Model Accuracies:\n")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  {model}: {acc:.4f} ({acc*100:.2f}%)\n")
    f.write("\n" + classification_report(y_test, y_pred_final, target_names=['No Dropout', 'Dropout']))

print("\nAccuracy report saved: models/model_accuracy_report.txt")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)
