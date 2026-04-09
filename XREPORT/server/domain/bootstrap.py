from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class EnvironmentBootstrapState:
    lock: Lock = field(default_factory=Lock)
    bootstrapped: bool = False
