from __future__ import annotations

import threading
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


###############################################################################
def singleton(cls: type[T]) -> type[T]:
    """
    Per-process, thread-safe singleton decorator.

    - First construction creates the instance.
    - Subsequent constructions return the same instance.
    - __init__ runs exactly once.
    """
    lock = threading.Lock()
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args: Any, **kwargs) -> None:
        if getattr(self, "_singleton_initialized", False):
            return
        original_init(self, *args, **kwargs)
        setattr(self, "_singleton_initialized", True)

    setattr(cls, "__init__", __init__)

    def _new(cls_, *args: Any, **kwargs) -> Any:
        if getattr(cls_, "_instance", None) is None:
            with lock:
                if getattr(cls_, "_instance", None) is None:
                    instance = object.__new__(cls_)
                    setattr(cls_, "_instance", instance)
        return getattr(cls_, "_instance")

    setattr(cls, "__new__", _new)
    return cls
