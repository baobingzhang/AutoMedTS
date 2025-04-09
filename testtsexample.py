import os
import glob
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 替代 autosklearn 的自定义包
from automedts.classification import automedtsClassifier

# ✅ 设置随机种子，保证可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 数据路径
data_dir = "data/Surgical_trajectory"
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

# ✅ 按 subject 文件名组织
subject_data = {}
for file_path in all_files:
    subject_id = os.path.basename(file_path).split("_")[0]  # e.g., 'subject-03'
    df = pd.read_csv(file_path)
    subject_data.setdefault(subject_id, []).append(df)

# ✅ 合并同一 subject 下的所有片段（如 subject-04 有多个片段）
subject_data = {
    k: pd.concat(v, ignore_index=True)
    for k, v in subject_data.items()
}

# ✅ 拆分：按 subject 名称划分训练 / 测试集（8:2）
subject_ids = sorted(subject_data.keys())
random.shuffle(subject_ids)
split_idx = int(len(subject_ids) * 0.8)
train_ids = subject_ids[:split_idx]
test_ids = subject_ids[split_idx:]

print("Train subjects:", train_ids)
print("Test subjects:", test_ids)

# 构造训练集
train_df = pd.concat([subject_data[sid] for sid in train_ids], ignore_index=True)
test_df = pd.concat([subject_data[sid] for sid in test_ids], ignore_index=True)

# ✅ 打印列名确认
print("Train columns:", train_df.columns)

# ✅ 划分特征和标签（这里 label 是你的目标列）
assert "labels" in train_df.columns, "Missing 'label' column in training data"
X_train = train_df.drop(columns=["labels"])
y_train = train_df["labels"]

assert "labels" in test_df.columns, "Missing 'label' column in testing data"
X_test = test_df.drop(columns=["labels"])
y_test = test_df["labels"]

print("Training samples shape:", X_train.shape)
print("Testing samples shape:", X_test.shape)

# ✅ 构建 AutoML 模型
automl = automedtsClassifier(
    time_left_for_this_task=600,             # 总运行时间（秒）
    per_run_time_limit=30,                  # 每次训练的最长时间
    ensemble_kwargs={"ensemble_size": 10},  # 避免弃用警告
    seed=SEED,
    tmp_folder="/tmp/automedts_tmp",        # 临时缓存
)

print("Training the AutoML classifier...")
automl.fit(X_train, y_train)
print("Training complete!")

# ✅ 开始预测
y_pred = automl.predict(X_test)
print("Predicting complete.")

# ✅ 输出评估结果
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ 展示模型细节
print("Final ensemble models:")
print(automl.show_models())
