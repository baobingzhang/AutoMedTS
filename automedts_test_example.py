
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import automedts.classification
import joblib
import os
import logging

# 1. Load raw data
data_dir = './data/surgical_trajectory'
exp_files = [f for f in os.listdir(data_dir) if f.endswith('_Exp.csv')]
nov_files = [f for f in os.listdir(data_dir) if f.endswith('_Nov.csv')]

exp_data = pd.concat(
    [pd.read_csv(os.path.join(data_dir, f)) for f in exp_files],
    ignore_index=True
)
nov_data = pd.concat(
    [pd.read_csv(os.path.join(data_dir, f)) for f in nov_files],
    ignore_index=True
)

# 2. Split novice into train / test
X_nov = nov_data.iloc[:, :-1]
y_nov = nov_data.iloc[:, -1]
X_nov_train, X_nov_test, y_nov_train, y_nov_test = train_test_split(
    X_nov, y_nov, test_size=0.5, random_state=42
)

# 3. Build train / test sets (raw, no windowing here)
X_train = pd.concat([exp_data.iloc[:, :-1], X_nov_train], ignore_index=True)
y_train = pd.concat([exp_data.iloc[:, -1], y_nov_train], ignore_index=True)

X_test, y_test = X_nov_test, y_nov_test

# 4. Create AutoML classifier with sliding‑window enabled
automl = automedts.classification.automedtsClassifier(
    time_left_for_this_task=3600,
    per_run_time_limit=360,
    ensemble_kwargs={'ensemble_size': 100}
)

# # 5. Train (internally will call create_windows on X_train/y_train)
automl.fit(X_train.values, y_train.values)

# 6. Save model
# joblib.dump(automl, 'surgery_stage_model.pkl')
# print("Model saved as 'surgery_stage_model.pkl'")


# # 7. Predict (internally will window X_test/y_test before predicting)
print("Making predictions on test set...")
y_pred, y_slided = automl.predict(X_test.values, y=y_test.values)

# 8. Evaluate
print(f"\nAccuracy on novice dataset: {accuracy_score(y_slided, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_slided, y_pred))

# 9. Model composition
print("\nModel Composition Summary:")
print(automl.show_models())
