import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import automedts.classification
import joblib
import os

# --- 配置部分 ---
model_path = 'surgery_stage_model.pkl'
data_dir = './data/surgical_trajectory'

# --- Step 1: 加载数据 ---
exp_files = [f for f in os.listdir(data_dir) if f.endswith('_Exp.csv')]
nov_files = [f for f in os.listdir(data_dir) if f.endswith('_Nov.csv')]

exp_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in exp_files])
nov_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in nov_files])

X_nov = nov_data.iloc[:, :-1]
y_nov = nov_data.iloc[:, -1]
X_nov_train, X_nov_test, y_nov_train, y_nov_test = train_test_split(X_nov, y_nov, test_size=0.5, random_state=42)

X_train = pd.concat([exp_data.iloc[:, :-1], X_nov_train])
y_train = pd.concat([exp_data.iloc[:, -1], y_nov_train])
X_test = X_nov_test
y_test = y_nov_test

# --- Step 2: 加载或训练模型 ---
if os.path.exists(model_path):
    print(f"Loading existing model from '{model_path}'...")
    automl = joblib.load(model_path)
else:
    print("Model not found. Training a new one...")
    automl = automedts.classification.automedtsClassifier(
        time_left_for_this_task=200,
        per_run_time_limit=120,
        ensemble_kwargs={'ensemble_size': 50}
    )
    automl.fit(X_train, y_train)
    joblib.dump(automl, model_path)
    print(f"Model trained and saved as '{model_path}'")

# --- Step 3: 使用模型进行预测 ---
print("Making predictions on test set...")
y_pred, y_slid= automl.predict(X_test, y = y_test)

# --- Step 4: 输出评估结果 ---
print(f"\n Accuracy on novice dataset: {accuracy_score(y_slid, y_pred):.4f}")
print("\n Classification Report:")
print(classification_report(y_slid, y_pred))

# --- Step 5: 模型组成信息 ---
print("\n Model Composition Summary:")
print(automl.show_models())
