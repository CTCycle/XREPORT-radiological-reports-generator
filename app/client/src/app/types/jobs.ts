import type { components } from './api.generated';

type Schemas = components['schemas'];

export type JobLifecycleStatus = Schemas['JobStatusResponse']['status'];
export type JobStartResponse = Schemas['JobStartResponse'];
export type JobStatusResponse = Schemas['JobStatusResponse'];
export type JobListResponse = Schemas['JobListResponse'];
export type JobCancelResponse = Schemas['JobCancelResponse'];
