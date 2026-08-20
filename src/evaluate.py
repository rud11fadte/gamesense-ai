import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_preprocessing import (
    load_data,
    remove_unnecessary_columns,
    separate_features_target,
)
from feature_engineering import create_features


MODEL_PATH = "models/gamesense_best_model.joblib"


def main():

    df = load_data()

    df, _ = remove_unnecessary_columns(df)

    df = create_features(df)

    X, y = separate_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    print("\nClassification Report")
    print("=" * 60)
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Churned",
                "Churned",
            ],
        )
    )

    print("Confusion Matrix")
    print("=" * 60)
    print(confusion_matrix(y_test, predictions))

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=[
            "Not Churned",
            "Churned",
        ],
    )

    display.ax_.set_title(
        "GameSense AI - Confusion Matrix"
    )

    plt.tight_layout()
    plt.savefig(
        "models/confusion_matrix.png",
        dpi=300,
    )

    print(
        "\nConfusion matrix saved to "
        "models/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()