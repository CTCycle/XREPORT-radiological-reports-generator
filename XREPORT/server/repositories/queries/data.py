from __future__ import annotations

import pandas as pd

from XREPORT.server.repositories.database.backend import XREPORTDatabase, database


###############################################################################
class DataRepositoryQueries:
    def __init__(self, db: XREPORTDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    @property
    def backend(self):
        return self.database.backend

    # -------------------------------------------------------------------------
    def load_table(
        self,
        table_name: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> pd.DataFrame:
        return self.database.load_from_database(table_name, limit=limit, offset=offset)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.database.count_rows(table_name)

    # -------------------------------------------------------------------------
    def save_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.save_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.upsert_into_database(dataset, table_name)
