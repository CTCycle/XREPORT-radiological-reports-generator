import pandas as pd
import importlib.util
import os


def load_normalize_string_columns():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "XREPORT",
        "server",
        "repositories",
        "queries",
        "common.py",
    )
    spec = importlib.util.spec_from_file_location("xreport_query_common", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load query common module for tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_string_columns


normalize_string_columns = load_normalize_string_columns()


def test_normalize_string_columns_handles_pandas_string_dtype() -> None:
    dataframe = pd.DataFrame({"name": ["alpha", None]})

    normalized = normalize_string_columns(dataframe)

    assert normalized["name"].dtype == object
    assert normalized.loc[0, "name"] == "alpha"
    assert normalized.loc[1, "name"] is None


def test_normalize_string_columns_handles_object_string_column() -> None:
    dataframe = pd.DataFrame(
        {"name": pd.Series(["alpha", pd.NA], dtype=object)}
    )

    normalized = normalize_string_columns(dataframe)

    assert normalized["name"].dtype == object
    assert normalized.loc[0, "name"] == "alpha"
    assert normalized.loc[1, "name"] is None


def test_normalize_string_columns_ignores_mixed_object_column() -> None:
    dataframe = pd.DataFrame(
        {"payload": pd.Series(["alpha", pd.NA, 3], dtype=object)}
    )

    normalized = normalize_string_columns(dataframe)

    assert normalized.loc[0, "payload"] == "alpha"
    assert normalized.loc[1, "payload"] is pd.NA
    assert normalized.loc[2, "payload"] == 3
