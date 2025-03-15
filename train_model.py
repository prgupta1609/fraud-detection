import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time

def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"ROC-AUC Score: {auc_roc:.4f}")
    print(f"Average prediction time: {prediction_time/len(y_test)*1000:.4f} ms per transaction")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    
    return auc_roc, y_pred

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Handle class imbalance using SMOTE
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts()}")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)
lr_score, lr_preds = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
rf_score, rf_preds = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Save the best performing model
if rf_score > lr_score:
    print("\nRandom Forest performed better. Saving this model...")
    best_model = rf_model
    model_name = "Random Forest"
else:
    print("\nLogistic Regression performed better. Saving this model...")
    best_model = lr_model
    model_name = "Logistic Regression"

joblib.dump(best_model, 'fraud_detection_model.pkl')

# Save model information
model_info = {
    'model_name': model_name,
    'auc_score': rf_score if rf_score > lr_score else lr_score,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(model_info, 'model_info.pkl')

print("\nModel training completed successfully!")