import autosklearn.classification
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# 设置日志输出等级
logging.basicConfig(level=logging.INFO)

# 加载内置数据集
X, y = load_digits(return_X_y=True)

print("数据加载完成，总样本数：", X.shape[0])

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("训练样本数：", X_train.shape[0], "，测试样本数：", X_test.shape[0])

# 构建 AutoML 分类器（可设定运行时间）
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,              # 总运行时间（秒）
    per_run_time_limit=30,                   # 每个模型评估最大时间
    tmp_folder='/tmp/autosklearn_tmp',       # 中间缓存
    disable_evaluator_output=False,          # 显示过程
    ensemble_size=10,                        # 显式设置集成规模（防止时间用光没集成）
    seed=1
)

print("开始模型训练……")
automl.fit(X_train, y_train)
print("训练完成！")

print("开始模型预测……")
y_pred = automl.predict(X_test)
print("预测完成！")

# 输出结果
acc = accuracy_score(y_test, y_pred)
print("模型准确率：", acc)

print("最终使用的模型集成：")
print(automl.show_models())
