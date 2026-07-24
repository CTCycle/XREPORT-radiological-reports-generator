from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import Request

from server.api.errors import handle_service_error
from server.services.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    InternalServiceError,
    NotFoundError,
    PayloadTooLargeError,
    ServiceError,
    UnsupportedOperationError,
)


###############################################################################
@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [
        (BadRequestError, 400),
        (ForbiddenError, 403),
        (NotFoundError, 404),
        (ConflictError, 409),
        (PayloadTooLargeError, 413),
        (InternalServiceError, 500),
        (UnsupportedOperationError, 501),
    ],
)
def test_service_error_handler_preserves_http_contract(
    error_type: type[ServiceError], expected_status: int
) -> None:
    request = Request({"type": "http", "method": "GET", "path": "/"})
    response = asyncio.run(handle_service_error(request, error_type("stable detail")))

    assert response.status_code == expected_status
    assert json.loads(response.body) == {"detail": "stable detail"}
