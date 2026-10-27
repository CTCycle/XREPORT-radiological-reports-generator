from __future__ import annotations

from typing import Any

from server.common.utils.logger import logger

###############################################################################
class DeviceConfig:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def set_device(self) -> str:
        """Select the training device and dtype policy.

        Keras' torch backend places new variables and input batches on CUDA
        whenever it is available, so callers must build and fit inside
        ``keras.device(<returned name>)`` for a CPU choice to take effect.
        """
        import torch
        from keras.mixed_precision import set_global_policy

        use_gpu = self.configuration.get("use_device_GPU", False)
        mixed_precision = self.configuration.get("use_mixed_precision", False)

        if not use_gpu or not torch.cuda.is_available():
            if use_gpu:
                logger.info("No GPU found. Falling back to CPU.")
            set_global_policy("float32")
            logger.info("CPU is set as the active device.")
            return "cpu"

        device_id = int(self.configuration.get("device_ID", 0))
        device_count = torch.cuda.device_count()
        if device_id >= device_count:
            raise ValueError(
                f"GPU device {device_id} is not available; "
                f"{device_count} CUDA device(s) detected."
            )
        torch.cuda.set_device(device_id)
        logger.info(f"GPU (cuda:{device_id}) is set as the active device.")

        policy = "float32"
        if mixed_precision:
            # bfloat16 needs no loss scaling; float16 is wrapped by Keras in a
            # LossScaleOptimizer at compile time.
            policy = (
                "mixed_bfloat16" if torch.cuda.is_bf16_supported() else "mixed_float16"
            )
            logger.info(f"Mixed precision policy {policy} is active during training")
        set_global_policy(policy)

        return f"cuda:{device_id}"
