import type {
  CheckpointInfo as ApiCheckpointInfo,
  CheckpointMetadataResponse as ApiCheckpointMetadataResponse,
  CheckpointsResponse as ApiCheckpointsResponse,
  DatasetInfo as ApiDatasetInfo,
  DatasetNamesResponse as ApiDatasetNamesResponse,
  DatasetStatusResponse as ApiDatasetStatusResponse,
  DatasetUploadResponse as ApiDatasetUploadResponse,
  DeleteResponse as ApiDeleteResponse,
  FolderSelectionResponse as ApiFolderSelectionResponse,
  ImageCountResponse as ApiImageCountResponse,
  ImageMetadataResponse as ApiImageMetadataResponse,
  ImagePathResponse as ApiImagePathResponse,
  LoadDatasetRequest as ApiLoadDatasetRequest,
  ProcessDatasetRequest as ApiProcessDatasetRequest,
  ProcessingMetadataResponse as ApiProcessingMetadataResponse,
  StartTrainingRequest as ApiStartTrainingRequest,
} from '@xreport/shared/app/types/generated/studio/types.gen';

export type ImagePathResponse = ApiImagePathResponse;
export type DatasetUploadResponse = ApiDatasetUploadResponse;
export type LoadDatasetRequest = ApiLoadDatasetRequest;
export interface LoadDatasetResponse {
  success: boolean;
  total_images: number;
  matched_records: number;
  unmatched_records: number;
  requires_confirmation: boolean;
  partial_import: boolean;
  message: string;
}
export type FolderSelectionResponse = ApiFolderSelectionResponse;
export type DatasetStatusResponse = ApiDatasetStatusResponse;
export type DatasetInfo = ApiDatasetInfo;
export type ProcessingMetadataResponse = ApiProcessingMetadataResponse;
export type CheckpointMetadataResponse = ApiCheckpointMetadataResponse;
export type DeleteResponse = ApiDeleteResponse;
export type DatasetNamesResponse = ApiDatasetNamesResponse;
export type ImageCountResponse = ApiImageCountResponse;
export type ImageMetadataResponse = ApiImageMetadataResponse;
export type ProcessDatasetRequest = ApiProcessDatasetRequest;
export type StartTrainingConfig = ApiStartTrainingRequest;
export type CheckpointInfo = ApiCheckpointInfo;
export type CheckpointsResponse = ApiCheckpointsResponse;

/** View model extracted from a completed dataset-processing job result. */
export interface ProcessDatasetResponse {
    success: boolean;
    total_samples: number;
    train_samples: number;
    validation_samples: number;
    vocabulary_size: number;
    message: string;
}
