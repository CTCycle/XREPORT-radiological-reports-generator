from __future__ import annotations

from fastapi import APIRouter, Query, status
from fastapi.responses import FileResponse

from XREPORT.server.domain.training import (
    BrowseResponse,
    DatasetNamesResponse,
    DatasetStatusResponse,
    ImagePathRequest,
    ImagePathResponse,
    LoadDatasetRequest,
    LoadDatasetResponse,
    ProcessingMetadataResponse,
    DeleteResponse,
    ProcessDatasetRequest,
    ImageCountResponse,
    ImageMetadataResponse,
)
from XREPORT.server.domain.jobs import (
    JobStartResponse,
    JobStatusResponse,
    JobCancelResponse,
)
from XREPORT.server.services.preparation import PreparationService, get_preparation_service


###############################################################################
class PreparationEndpoint:
    def __init__(
        self,
        router: APIRouter,
        service: PreparationService | None = None,
    ) -> None:
        self.router = router
        self.service = get_preparation_service() if service is None else service

    def get_dataset_status(self) -> DatasetStatusResponse:
        return self.service.get_dataset_status()

    def get_dataset_names(self) -> DatasetNamesResponse:
        return self.service.get_dataset_names()

    def get_processed_dataset_names(self) -> DatasetNamesResponse:
        return self.service.get_processed_dataset_names()

    def get_processing_metadata(self, dataset_name: str) -> ProcessingMetadataResponse:
        return self.service.get_processing_metadata(dataset_name)

    def delete_dataset(self, dataset_name: str) -> DeleteResponse:
        return self.service.delete_dataset(dataset_name)

    def validate_image_path(self, request: ImagePathRequest) -> ImagePathResponse:
        return self.service.validate_image_path(request)

    def load_dataset(self, request: LoadDatasetRequest) -> LoadDatasetResponse:
        return self.service.load_dataset(request)

    def process_dataset(self, request: ProcessDatasetRequest) -> JobStartResponse:
        return self.service.process_dataset(request)

    def get_preparation_job_status(self, job_id: str) -> JobStatusResponse:
        return self.service.get_preparation_job_status(job_id)

    def cancel_preparation_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_preparation_job(job_id)

    def browse_directory(
        self,
        path: str = Query("", description="Directory path to browse. Empty returns drives."),
    ) -> BrowseResponse:
        return self.service.browse_directory(path)

    def get_dataset_image_count(self, dataset_name: str) -> ImageCountResponse:
        return self.service.get_dataset_image_count(dataset_name)

    def get_dataset_image_metadata(
        self, dataset_name: str, index: int
    ) -> ImageMetadataResponse:
        return self.service.get_dataset_image_metadata(dataset_name, index)

    def get_dataset_image_content(self, dataset_name: str, index: int):
        path = self.service.get_dataset_image_content(dataset_name, index)
        return FileResponse(path)

    def add_routes(self) -> None:
        self.router.add_api_route(
            "/dataset/status",
            self.get_dataset_status,
            methods=["GET"],
            response_model=DatasetStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/names",
            self.get_dataset_names,
            methods=["GET"],
            response_model=DatasetNamesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/processed/names",
            self.get_processed_dataset_names,
            methods=["GET"],
            response_model=DatasetNamesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/metadata/{dataset_name}",
            self.get_processing_metadata,
            methods=["GET"],
            response_model=ProcessingMetadataResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/{dataset_name}",
            self.delete_dataset,
            methods=["DELETE"],
            response_model=DeleteResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/images/validate",
            self.validate_image_path,
            methods=["POST"],
            response_model=ImagePathResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/load",
            self.load_dataset,
            methods=["POST"],
            response_model=LoadDatasetResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/process",
            self.process_dataset,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/dataset/{dataset_name}/images/count",
            self.get_dataset_image_count,
            methods=["GET"],
            response_model=ImageCountResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/{dataset_name}/images/{index}",
            self.get_dataset_image_metadata,
            methods=["GET"],
            response_model=ImageMetadataResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset/{dataset_name}/images/{index}/content",
            self.get_dataset_image_content,
            methods=["GET"],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_preparation_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_preparation_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/browse",
            self.browse_directory,
            methods=["GET"],
            response_model=BrowseResponse,
            status_code=status.HTTP_200_OK,
        )


###############################################################################
def get_router() -> APIRouter:
    router = APIRouter(prefix="/preparation", tags=["preparation"])
    PreparationEndpoint(router=router).add_routes()
    return router


router = get_router()
