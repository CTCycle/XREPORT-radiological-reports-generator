import type { components } from './api.generated';

type Schemas = components['schemas'];

export type PixelDistribution = Schemas['PixelDistribution'];
export type ImageStatistics = Schemas['ImageStatistics'];
export type TextStatistics = Schemas['TextStatistics'];
export type ValidationRequest = Schemas['ValidationRequest'];
export type ValidationReport = Schemas['ValidationReportResponse'];

/** View model extracted from a completed dataset-validation job result. */
export interface ValidationResponse {
    success: boolean;
    message: string;
    pixel_distribution?: PixelDistribution | null;
    image_statistics?: ImageStatistics | null;
    text_statistics?: TextStatistics | null;
}
