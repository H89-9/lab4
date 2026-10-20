import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns=["Creditability"])
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

models = {
    "Random Forest 1": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest 2": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_subtrain, y_subtrain)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = model.predict(X_val)

    roc_auc = roc_auc_score(y_val, probabilities)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(y_val, predictions, zero_division=0)

    results.append({
        "Модель": name,
        "ROC-AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

table = pd.DataFrame(results)
table = table.sort_values("ROC-AUC", ascending=False)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print(table.to_string(index=False, float_format="%.4f"))
print("\nЛучшая модель на валидации:")
print(table.iloc[0]["Модель"])



from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []
print("\nКРОСС-ВАЛИДАЦИЯ")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    cv_results.append({
        "Модель": name,
        "Средний ROC-AUC": scores.mean(),
        "Отклонение": scores.std()
    })

cv_table = pd.DataFrame(cv_results)
cv_table = cv_table.sort_values(
    "Средний ROC-AUC",
    ascending=False
)

print(cv_table.to_string(index=False, float_format="%.4f"))
print("\nЛучшая модель по кросс-валидации:")
print(cv_table.iloc[0]["Модель"])