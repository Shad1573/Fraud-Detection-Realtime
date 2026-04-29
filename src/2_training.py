import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 1. Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Load Data
print("Reading dataset... (this might take a few seconds)")
df = pd.read_csv('data/creditcard.csv')

# 3. Prepare Features and Target
# X = everything except the answer | y = the answer (0 or 1)
X = df.drop(['Class'], axis=1)
y = df['Class']

# 4. Split data (80% to learn, 20% to test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} transactions. Balancing the weights...")

# 5. Build the Model
# We use scale_pos_weight because fraud is rare (1 in 500). 
# This tells the AI: "Missing one fraud is 500x worse than a false alarm."
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=500 
)

# 6. Train the AI
model.fit(X_train, y_train)

# 7. Evaluate the results
print("\n--- Model Performance Report ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save the "Brain"
joblib.dump(model, 'models/fraud_model.pkl')
print("\nSuccess! AI model saved in 'models/fraud_model.pkl'")