import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Check for missing values
print("Checking for missing values...")
print(df.isnull().sum())

# Display basic statistics
print("\nDataset statistics:")
print(f"Total transactions: {len(df)}")
print(f"Fraudulent transactions: {df['Class'].sum()}")
print(f"Legitimate transactions: {len(df) - df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
print("Saving preprocessed data...")
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

print("Preprocessing completed successfully!")