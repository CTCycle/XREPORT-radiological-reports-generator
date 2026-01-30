from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from keras.mixed_precision import set_global_policy

from XREPORT.server.utils.logger import logger


###############################################################################
class DeviceConfig:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def set_device(self) -> torch.device:
        use_gpu = self.configuration.get("use_device_GPU", False)
        device_name = "cuda" if use_gpu else "cpu"
        mixed_precision = self.configuration.get("use_mixed_precision", False)

        device = torch.device("cpu")
        if device_name == "cuda" and torch.cuda.is_available():
            device_id = self.configuration.get("device_ID", 0)
            device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
            logger.info(f"GPU (cuda:{device_id}) is set as the active device.")
            if mixed_precision:
                set_global_policy("mixed_float16")
                logger.info("Mixed precision policy is active during training")
        else:
            if device_name == "cuda":
                logger.info("No GPU found. Falling back to CPU.")
            logger.info("CPU is set as the active device.")

        return device


###############################################################################
class DeviceDataLoader:
    def __init__(self, dataloader: Any, device: torch.device) -> None:
        self.dataloader = dataloader
        self.device = device

    # -------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        for batch in self.dataloader:
            batch = self.to_device(batch)
            if isinstance(batch, list):
                yield tuple(batch)
            else:
                yield batch

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataloader)

    # -------------------------------------------------------------------------
    def to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        if isinstance(data, list):
            return [self.to_device(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.to_device(item) for item in data)
        if isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        return data
