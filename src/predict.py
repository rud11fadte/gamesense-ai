import joblib
import pandas as pd


MODEL_PATH = "models/gamesense_best_model.joblib"


def predict_player(player_data):
    """Predict churn probability for a player."""

    model = joblib.load(MODEL_PATH)

    player_df = pd.DataFrame([player_data])

    probability = model.predict_proba(
        player_df
    )[0][1]

    prediction = probability >= 0.5

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": bool(prediction),
        "churn_probability": probability,
        "risk_level": risk,
    }


if __name__ == "__main__":

    print("=" * 50)
    print("GAMESENSE AI - PLAYER CHURN PREDICTION")
    print("=" * 50)

    print(
        "\nThe trained model is ready for predictions."
    )

    print(
        "Use predict_player() from another Python "
        "program to submit player telemetry."
    )