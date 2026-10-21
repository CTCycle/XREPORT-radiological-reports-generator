import type {
  JobCancelResponse as ApiJobCancelResponse,
  JobListResponse as ApiJobListResponse,
  JobStartResponse as ApiJobStartResponse,
  JobStatusResponse as ApiJobStatusResponse,
} from './generated/types.gen';

export type JobLifecycleStatus = ApiJobStatusResponse['status'];
export type JobStartResponse = ApiJobStartResponse;
export type JobStatusResponse = ApiJobStatusResponse;
export type JobListResponse = ApiJobListResponse;
export type JobCancelResponse = ApiJobCancelResponse;
