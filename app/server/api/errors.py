from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

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


SERVICE_ERROR_STATUS_CODES: dict[type[ServiceError], int] = {
    BadRequestError: status.HTTP_400_BAD_REQUEST,
    ForbiddenError: status.HTTP_403_FORBIDDEN,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    PayloadTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    InternalServiceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    UnsupportedOperationError: status.HTTP_501_NOT_IMPLEMENTED,
}


###############################################################################
async def handle_service_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ServiceError):
        raise exc
    return JSONResponse(
        status_code=SERVICE_ERROR_STATUS_CODES[type(exc)],
        content={"detail": exc.detail},
    )


###############################################################################
def register_service_error_handlers(application: FastAPI) -> None:
    application.add_exception_handler(ServiceError, handle_service_error)
