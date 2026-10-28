from __future__ import annotations

from xreport_shared.common.utils.logger import logger
from xreport_shared.domain.system import CudaStatus, MemoryUsage, SystemStatusResponse

BYTES_PER_GB = 1024**3

###############################################################################
class SystemStatusService:

    # -------------------------------------------------------------------------
    @staticmethod
    def read_ram() -> MemoryUsage | None:
        try:
            import psutil

            memory = psutil.virtual_memory()
            return MemoryUsage(
                used_gb=round((memory.total - memory.available) / BYTES_PER_GB, 2),
                total_gb=round(memory.total / BYTES_PER_GB, 2),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAM metrics unavailable: %s", exc)
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def read_gpu() -> tuple[CudaStatus, MemoryUsage | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return CudaStatus(available=False), None
            name = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA probe failed: %s", exc)
            return CudaStatus(available=False), None
        try:
            free, total = torch.cuda.mem_get_info(0)
            vram = MemoryUsage(
                used_gb=round((total - free) / BYTES_PER_GB, 2),
                total_gb=round(total / BYTES_PER_GB, 2),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("VRAM metrics unavailable: %s", exc)
            vram = None
        return CudaStatus(available=True, device_name=name), vram

    # -------------------------------------------------------------------------
    def get_status(self) -> SystemStatusResponse:
        cuda, vram = self.read_gpu()
        return SystemStatusResponse(ram=self.read_ram(), cuda=cuda, vram=vram)

###############################################################################
def get_system_status_service() -> SystemStatusService:
    return SystemStatusService()
