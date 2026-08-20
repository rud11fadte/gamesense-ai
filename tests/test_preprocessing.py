import pandas as pd

from src.data_preprocessing import (
    load_data,
    remove_unnecessary_columns,
)


def test_dataset_loads():

    df = load_data()

    assert not df.empty
    assert "player_churn" in df.columns


def test_player_id_removed():

    df = load_data()

    processed_df, _ = remove_unnecessary_columns(df)

    assert "player_id" not in processed_df.columns


def test_target_exists():

    df = load_data()

    assert "player_churn" in df.columns