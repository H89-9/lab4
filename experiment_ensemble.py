import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("german.csv", sep=";")

print("Количество строк:", len(df))
print("Количество столбцов:", len(df.columns))
print("Пропущенных значений:", df.isna().sum().sum())

print("\nРаспределение классов:")
print(df["Creditability"].value_counts())

X = df.drop(columns=["Creditability"])
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nОбучающая выборка:", len(X_train))
print("Тестовая выборка:", len(X_test))


from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

data = EnsembleInput(
    train_features=X_subtrain,
    train_target=y_subtrain,
    test_features=X_val,
    test_target=y_val,
    random_seed=42
)

result = run_ensemble_experiment(data)

print("\nРЕЗУЛЬТАТЫ RANDOM FOREST")
print("Модель:", result.model_name)
print("ROC-AUC:", result.roc_auc)
print("Accuracy:", result.accuracy)
print("Precision:", result.precision)
print("Recall:", result.recall)
print("F1:", result.f1)
print("\nПервые 5 прогнозов:")

for actual, predicted, probability in zip(
    y_val.iloc[:5],
    result.predictions[:5],
    result.probabilities[:5]
):
    print(
        f"Реальное: {actual}, "
        f"Прогноз: {predicted}, "
        f"Вероятность: {probability:.3f}"
    )

final_data = EnsembleInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    random_seed=42
)

final_result = run_ensemble_experiment(final_data)

print("\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("Модель:", final_result.model_name)
print("ROC-AUC:", final_result.roc_auc)
print("Accuracy:", final_result.accuracy)
print("Precision:", final_result.precision)
print("Recall:", final_result.recall)
print("F1:", final_result.f1)