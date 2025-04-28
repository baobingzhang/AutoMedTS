
<p align="center">
  <img src="https://raw.githubusercontent.com/baobingzhang/AutoMedTS/main/ailap_logo.png" alt="AiLap Logo" width="180"/>
</p>




# AutoMedTS

> **Automating Clinical Time Series Intelligence**  
> An end-to-end automated ML framework tailored for clinical time-series data. Streamline data preprocessing, model selection, hyperparameter optimization, and uncertainty-aware ensembling—minimal manual effort required. Originally developed to deliver real-time action recognition and feedback
in laparoscopic surgery training. It integrates multiple class-imbalance handling strategies (including power transform, temperature-scaled softmax, and SMOTE). AutoMedTS automatically searches for optimal models and hyperparameters, providing an end-to-end pipeline for training and inference. The framework is easily extensible to other medical time-series tasks and offers comprehensive evaluation metrics (Macro-F1, Balanced Accuracy, MCC) along with visualization tools for performance analysis and feedback.

---

## 🚀 Key Features

- **One-Click Pipeline**  
  From raw time-series to deployable model in a single `fit()` call.
- **Advanced Imbalance Handling**  
  SMOTE, temperature-scaled Softmax, power-law mapping, jitter augmentation.
- **Meta-Learning Warm-Start**  
  Seed Bayesian optimizer with high-quality configurations from prior tasks.
- **Uncertainty-Aware Ensembling**  
  Greedy forward-selection with variance penalty for robust predictions.
- **Real-Time Inference**  
  High throughput and low latency—ideal for live surgical feedback.
- **Modular & Extensible**  
  Swap in custom preprocessors, resamplers, models, or metrics.

---

## 🔧 Installation

```bash
# From PyPI
pip install automedts

# Or from GitHub source
git clone https://github.com/your-username/AutoMedTS.git
cd AutoMedTS
pip install -e .




## 🔧 Example

```python
import pandas as pd
from automedts.classification import AutoMedTSClassifier

# 1. Load your surgical trajectory data (features in all cols except last, labels in last col)
df = pd.read_csv("data/surgical_trajectory/Nov_01_Nov.csv")
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# 2. Instantiate and train the AutoMedTS classifier (300 s total, 30 s per model)
clf = AutoMedTSClassifier(time_left_for_this_task=300, per_run_time_limit=30)
clf.fit(X, y)

# 3. (Optional) Predict on new data
# y_pred, _ = clf.predict(X_new, y=y_new)








