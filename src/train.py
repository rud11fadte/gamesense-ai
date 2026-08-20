import os

import joblib
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    create_preprocessor,
    load_data,
    remove_unnecessary_columns,
    separate_features_target,
)
from feature_engineering import create_features


DATA_PATH = "data/raw/player-churn.csv"
MODEL_DIR = "models"


def build_models(preprocessor):
    """Create ML pipelines."""

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():

    print("=" * 60)
    print("GAMESENSE AI - MODEL TRAINING")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print(f"\nOriginal dataset: {df.shape}")

    df, removed_columns = remove_unnecessary_columns(df)

    print(
        f"Removed {len(removed_columns)} constant columns."
    )

    df = create_features(df)

    X, y = separate_features_target(df)

    print(f"Feature matrix: {X.shape}")
    print(f"Target: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    preprocessor = create_preprocessor(X_train)

    models = build_models(preprocessor)

    results = {}
    trained_models = {}

    print("\nTraining models...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results[name] = metrics
        trained_models[name] = model

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall:    {metrics['recall']:.4f}"
        )
        print(
            f"F1 Score:  {metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:   {metrics['roc_auc']:.4f}"
        )
        print()

    results_df = pd.DataFrame(results).T

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.round(4))

    best_model_name = results_df["f1"].idxmax()

    print(
        f"\nBest model based on F1 Score: "
        f"{best_model_name}"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "gamesense_best_model.joblib",
    )

    joblib.dump(
        trained_models[best_model_name],
        model_path,
    )

    results_df.to_csv(
        os.path.join(
            MODEL_DIR,
            "model_comparison.csv",
        )
    )

    print(
        f"\nBest model saved to: {model_path}"
    )


if __name__ == "__main__":
    main()