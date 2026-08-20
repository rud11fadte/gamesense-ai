import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/raw/player-churn.csv"


def load_data(path=DATA_PATH):
    """Load the raw player churn dataset."""
    return pd.read_csv(path)


def remove_unnecessary_columns(df):
    """Remove identifier and constant columns."""

    df = df.copy()

    if "player_id" in df.columns:
        df = df.drop(columns=["player_id"])

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        df = df.drop(columns=constant_columns)

    return df, constant_columns


def separate_features_target(df):
    """Separate features and target."""

    X = df.drop(columns=["player_churn"])
    y = df["player_churn"].astype(int)

    return X, y


def create_preprocessor(X):
    """Create preprocessing pipelines."""

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor