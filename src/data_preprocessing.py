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
    """
    Remove identifier columns and columns
    that contain only one unique value.
    """

    df = df.copy()

    # Player ID is an identifier and should not be used as a feature.
    if "player_id" in df.columns:
        df = df.drop(columns=["player_id"])

    # Remove columns with only one unique value.
    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        df = df.drop(columns=constant_columns)

    return df, constant_columns


def separate_features_target(df):
    """Separate input features from the target variable."""

    X = df.drop(columns=["player_churn"])
    y = df["player_churn"]

    return X, y


def create_preprocessor(X):
    """Create a preprocessing pipeline for numerical and categorical features."""

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
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
                    sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numerical_features, categorical_features


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    df, removed_columns = remove_unnecessary_columns(df)

    print(
        f"Removed {len(removed_columns)} constant columns."
    )

    print("Removed columns:")
    for column in removed_columns:
        print(f"  - {column}")

    X, y = separate_features_target(df)

    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nTarget distribution:")
    print(y.value_counts())

    preprocessor, numerical_features, categorical_features = (
        create_preprocessor(X)
    )

    print(
        f"\nNumerical features: {len(numerical_features)}"
    )

    print(
        f"Categorical features: {len(categorical_features)}"
    )

    print("\nPreprocessing configuration created successfully.")