from __future__ import annotations

from typing import Any, cast

import pandas as pd

from APP.server.utils.constants import (
    GEONAMES_TABLE,
    GEONAMES_COLUMNS,
    GIBS_LAYERS_TABLE,
    GIBS_LAYER_COLUMNS,
    SEARCH_SESSION_COLUMNS,
    SEARCH_SESSIONS_TABLE,
)
from APP.server.database.database import database


###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -----------------------------------------------------------------------------
    def upsert_this(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        frame = pd.DataFrame.from_records(records)
        frame = frame.reindex(columns=COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        database.upsert_into_database(frame, GEONAMES_TABLE)

    # -----------------------------------------------------------------------------
    def upsert_gibs_layers(self, layers: list[dict[str, Any]]) -> None:
        if not layers:
            return
        frame = pd.DataFrame.from_records(layers)
        frame = frame.reindex(columns=GIBS_LAYER_COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        database.upsert_into_database(frame, GIBS_LAYERS_TABLE)

    # -----------------------------------------------------------------------------
    def insert_search_session(self, session: dict[str, Any]) -> None:
        if not session:
            return
        frame = pd.DataFrame.from_records([session])
        frame = frame.reindex(columns=SEARCH_SESSION_COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        database.upsert_into_database(frame, SEARCH_SESSIONS_TABLE)
