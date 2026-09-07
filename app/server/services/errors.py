from __future__ import annotations

###############################################################################
class ServiceError(Exception):
    """Base exception for expected service-layer failures."""

    # -------------------------------------------------------------------------
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

###############################################################################
class BadRequestError(ServiceError):
    pass

###############################################################################
class ForbiddenError(ServiceError):
    pass

###############################################################################
class NotFoundError(ServiceError):
    pass

###############################################################################
class ConflictError(ServiceError):
    pass

###############################################################################
class PayloadTooLargeError(ServiceError):
    pass

###############################################################################
class UnsupportedOperationError(ServiceError):
    pass

###############################################################################
class InternalServiceError(ServiceError):
    pass
