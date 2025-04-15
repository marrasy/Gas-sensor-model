import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load Data ======
df = pd.read_excel('DataNT.xlsx')  # Replace with your actual Excel filename

# Split labels (first 6 columns) and features (sensor1~sensor30)
X = df.iloc[:, 6:].values         # Features: sensor1 to sensor30
y = df.iloc[:, 0:6].values        # Labels: 6 gas presence (1/0)

# Normalize feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ====== Train Random Forest ======

clf = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)



clf.fit(X_train, y_train)


# Predict
y_pred = clf.predict(X_test)

# ====== Classification Report ======
label_names = ['Ethyl butyrate', 'Ethyl hexanoate', '2-heptanone',
               'Pinene', 'Allyl methyl sulfide', 'Diallyl sulfide']

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

# ====== Confusion Matrix Visualization ======
from sklearn.metrics import multilabel_confusion_matrix, f1_score, accuracy_score

# 假设你已经有 y_test, y_pred 和 label_names
conf_matrices = multilabel_confusion_matrix(y_test, y_pred)
f1_scores = f1_score(y_test, y_pred, average=None)
accuracies = []

print("各气体标签的F1-score、混淆矩阵和准确率：\n")
for i, (label, cm) in enumerate(zip(label_names, conf_matrices)):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    accuracies.append(acc)

    print(f"【{label}】")
    print(f"F1-score: {f1_scores[i]:.2f}")
    print(f"混淆矩阵:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    print(f"准确率: {acc:.2f}")
    print("-" * 40)

# 如果你还想输出平均指标：
print(f"\n平均F1-score: {f1_scores.mean():.2f}")
print(f"平均准确率: {sum(accuracies) / len(accuracies):.2f}")


