from __future__ import annotations
from unittest.mock import Mock

from fastapi import APIRouter

from server.api.upload import UploadEndpoint
from server.domain.training import DatasetUploadResponse
from tests.conftest import run_async_in_thread

###############################################################################
class UploadFileStub:

    # -------------------------------------------------------------------------
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content

    # -------------------------------------------------------------------------
    async def read(self) -> bytes:
        return self._content

###############################################################################
def test_upload_endpoint_reads_file_and_forwards_plain_inputs() -> None:
    expected_response = DatasetUploadResponse(
        success=True,
        filename="dataset.csv",
        dataset_name="dataset",
        row_count=1,
        column_count=2,
        columns=["image", "text"],
        message="Successfully parsed dataset.csv",
    )
    service = Mock()
    service.upload_dataset.return_value = expected_response

    endpoint = UploadEndpoint(router=APIRouter(), upload_service=service)
    upload_file = UploadFileStub(filename="dataset.csv", content=b"image,text\na,b\n")

    response = run_async_in_thread(endpoint.upload_dataset(upload_file))

    service.upload_dataset.assert_called_once_with(
        filename="dataset.csv",
        contents=b"image,text\na,b\n",
    )
    assert response == expected_response
