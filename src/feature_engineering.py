import pandas as pd


def create_features(df):
    """
    Create meaningful player engagement features.
    """

    df = df.copy()

    # Remove identifiers and potential target leakage.
    columns_to_drop = [
        "player_id",
        "player_type",
    ]

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors="ignore",
    )

    # Aggregate session activity.
    session_columns = [
        col
        for col in df.columns
        if "begin_session_count" in col
    ]

    if session_columns:
        df["total_session_activity"] = (
            df[session_columns].sum(axis=1)
        )

    # Aggregate stage activity.
    stage_columns = [
        col
        for col in df.columns
        if "begin_stage_count" in col
    ]

    if stage_columns:
        df["total_stage_activity"] = (
            df[stage_columns].sum(axis=1)
        )

    # Activity intensity.
    if "player_lifetime" in df.columns:
        df["sessions_per_lifetime"] = (
            df["session_count"]
            / (df["player_lifetime"] + 1)
        )

    return df