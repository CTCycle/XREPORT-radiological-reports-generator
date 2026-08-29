import type { components } from './api.generated';

type Schemas = components['schemas'];

export type ImagePathResponse = Schemas['ImagePathResponse'];
export type DatasetUploadResponse = Schemas['DatasetUploadResponse'];
export type LoadDatasetRequest = Schemas['LoadDatasetRequest'];
export type LoadDatasetResponse = Schemas['LoadDatasetResponse'];
export type DirectoryItem = Schemas['DirectoryItem'];
export type BrowseResponse = Schemas['BrowseResponse'];
export type DatasetStatusResponse = Schemas['DatasetStatusResponse'];
export type DatasetInfo = Schemas['DatasetInfo'];
export type ProcessingMetadataResponse = Schemas['ProcessingMetadataResponse'];
export type CheckpointMetadataResponse = Schemas['CheckpointMetadataResponse'];
export type DeleteResponse = Schemas['DeleteResponse'];
export type DatasetNamesResponse = Schemas['DatasetNamesResponse'];
export type ImageCountResponse = Schemas['ImageCountResponse'];
export type ImageMetadataResponse = Schemas['ImageMetadataResponse'];
export type ProcessDatasetRequest = Schemas['ProcessDatasetRequest'];
export type StartTrainingConfig = Schemas['StartTrainingRequest'];
export type CheckpointInfo = Schemas['CheckpointInfo'];
export type CheckpointsResponse = Schemas['CheckpointsResponse'];

/** View model extracted from a completed dataset-processing job result. */
export interface ProcessDatasetResponse {
    success: boolean;
    total_samples: number;
    train_samples: number;
    validation_samples: number;
    vocabulary_size: number;
    message: string;
}
