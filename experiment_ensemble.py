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