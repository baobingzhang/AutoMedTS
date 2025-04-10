import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import automedts.classification
import joblib
import os

print("1")
# 1. 加载数据文件路径
data_dir = './data/surgical_trajectory'  # 当前数据文件所在的目录
print("2")
# 区分经验丰富的医生和实习医师的数据集
exp_files = [f for f in os.listdir(data_dir) if f.endswith('_Exp.csv')]
nov_files = [f for f in os.listdir(data_dir) if f.endswith('_Nov.csv')]
print("3")
# 2. 加载所有经验丰富医生的数据集并合并
exp_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in exp_files])
print("4")
# 3. 加载所有实习医师的数据集并合并
nov_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in nov_files])
print("5")
# 4. 将实习医生的数据划分为两部分，一部分用于训练，一部分用于测试
X_nov = nov_data.iloc[:, :-1]
y_nov = nov_data.iloc[:, -1]
print("6")
X_nov_train, X_nov_test, y_nov_train, y_nov_test = train_test_split(X_nov, y_nov, test_size=0.5, random_state=42)
print("7")
# 5. 将经验医生的数据与部分实习医生数据合并，作为新的训练集
X_train = pd.concat([exp_data.iloc[:, :-1], X_nov_train])
y_train = pd.concat([exp_data.iloc[:, -1], y_nov_train])
print("8")
# 剩下的实习医生数据作为测试集
X_test = X_nov_test
y_test = y_nov_test

print("9")


# 6. 创建AutoML分类器并训练，增加总时间
automl = automedts.classification.automedtsClassifier(
    time_left_for_this_task=300,  # 训练总时间增加到3600秒（1小时）
    per_run_time_limit=120,        # 每个模型的最大运行时间（秒）
    ensemble_kwargs={'ensemble_size': 50}  # 集成模型参数
)
print("10")
# 训练模型
automl.fit(X_train, y_train)
print("11")
# 保存模型
joblib.dump(automl, 'surgery_stage_model.pkl')
print("Model saved as 'surgery_stage_model.pkl'")
print("12")
# 7. 使用模型对剩余的实习医师数据进行预测
y_pred = automl.predict(X_test)
print("13")

# 8. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on novice dataset: {accuracy}")
print(classification_report(y_test, y_pred))
print("14")

# 9. 显示使用的模型组合信息
print("Model composition summary:")
print(automl.show_models())