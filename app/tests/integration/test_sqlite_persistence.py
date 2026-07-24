from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.repositories.schemas import Base


###############################################################################
def test_sqlite_schema_contract() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    session_factory = sessionmaker(bind=engine, future=True)

    Base.metadata.create_all(engine)
    tables = set(inspect(engine).get_table_names())

    assert {"datasets", "dataset_versions", "dataset_records"} <= tables

    inference_columns = {
        column["name"]: column
        for column in inspect(engine).get_columns("inference_runs")
    }
    assert {
        "provider",
        "model_ref",
        "model_revision",
        "generation_profile",
        "generation_config_json",
        "clinical_context",
        "request_id",
        "status",
        "execution_time_seconds",
    } <= set(inference_columns)
    assert inference_columns["checkpoint_id"]["nullable"] is True

    with session_factory() as session:
        session.execute(text("SELECT 1"))

    engine.dispose()
