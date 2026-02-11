from __future__ import annotations

import pandas as pd

from XREPORT.server.common.constants import (
    PROCESSING_RUNS_TABLE,
    TRAINING_SAMPLES_TABLE,
)
from XREPORT.server.repositories.database.backend import XREPORTDatabase, database


###############################################################################
class TrainingRepositoryQueries:
    def __init__(self, db: XREPORTDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    def load_training_dataset(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.database.load_from_database(
            TRAINING_SAMPLES_TABLE, limit=limit, offset=offset
        )

    # -------------------------------------------------------------------------
    def save_training_dataset(self, dataset: pd.DataFrame) -> None:
        self.database.save_into_database(dataset, TRAINING_SAMPLES_TABLE)

    # -------------------------------------------------------------------------
    def upsert_training_dataset(self, dataset: pd.DataFrame) -> None:
        self.database.upsert_into_database(dataset, TRAINING_SAMPLES_TABLE)

    # -------------------------------------------------------------------------
    def load_training_metadata(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.database.load_from_database(
            PROCESSING_RUNS_TABLE, limit=limit, offset=offset
        )

    # -------------------------------------------------------------------------
    def save_training_metadata(self, metadata: pd.DataFrame) -> None:
        self.database.save_into_database(metadata, PROCESSING_RUNS_TABLE)
