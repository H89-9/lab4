import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()

    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=data.random_seed,
    n_jobs=-1
)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    return EnsembleResult(
        model_name="Random Forest",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )