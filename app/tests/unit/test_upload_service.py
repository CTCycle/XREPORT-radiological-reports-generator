from __future__ import annotations

import pandas as pd
import pytest

from server.services.errors import PayloadTooLargeError
from server.services.upload import (
    MAX_DATASET_UPLOAD_BYTES,
    UploadService,
    UploadState,
)


def test_upload_accepts_exact_payload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = UploadState()
    service = UploadService(state)
    parser_called = False

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parser_called
        parser_called = True
        return pd.DataFrame({"text": ["report"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    response = service.upload_dataset(
        "at_limit.csv", b"x" * MAX_DATASET_UPLOAD_BYTES
    )

    assert parser_called is True
    assert response.success is True
    assert response.upload_id


def test_upload_rejects_payload_over_limit_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_called = False

    def fake_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parser_called
        parser_called = True
        return pd.DataFrame({"text": ["report"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    with pytest.raises(PayloadTooLargeError):
        UploadService(UploadState()).upload_dataset(
            "over_limit.csv", b"x" * (MAX_DATASET_UPLOAD_BYTES + 1)
        )

    assert parser_called is False


def test_upload_ids_keep_parsed_contents_independent() -> None:
    state = UploadState()
    service = UploadService(state)

    first = service.upload_dataset(
        "first.csv", b"image,text\nfirst.png,First report\n"
    )
    second = service.upload_dataset(
        "second.csv", b"image,text\nsecond.png,Second report\n"
    )

    assert first.upload_id != second.upload_id
    first_data = state.get(first.upload_id)
    second_data = state.get(second.upload_id)
    assert first_data is not None
    assert second_data is not None
    assert first_data["filename"] == "first.csv"
    assert first_data["dataframe"].to_dict("records") == [
        {"image": "first.png", "text": "First report"}
    ]
    assert second_data["filename"] == "second.csv"
    assert second_data["dataframe"].to_dict("records") == [
        {"image": "second.png", "text": "Second report"}
    ]
