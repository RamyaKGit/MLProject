import pytest
import pandas as pd
from src.pipeline.predict_pipeline import CustomData


def test_custom_data_to_dataframe():
    """
    Test Case 1: Verify that CustomData correctly converts input attributes
    into a valid pandas DataFrame with correct columns and data types.
    """
    custom_data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74
    )

    df = custom_data.get_data_as_frame()

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert row count
    assert len(df) == 1

    # Assert expected columns are present
    expected_columns = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "reading_score",
        "writing_score"
    ]
    assert list(df.columns) == expected_columns

    # Assert data values match input attributes
    assert df.loc[0, "gender"] == "female"
    assert df.loc[0, "reading_score"] == 72
    assert df.loc[0, "writing_score"] == 74
