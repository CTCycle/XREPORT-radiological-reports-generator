import type {
  ImageStatistics as ApiImageStatistics,
  PixelDistribution as ApiPixelDistribution,
  TextStatistics as ApiTextStatistics,
  ValidationReportResponse as ApiValidationReportResponse,
  ValidationRequest as ApiValidationRequest,
} from './generated/types.gen';

export type PixelDistribution = ApiPixelDistribution;
export type ImageStatistics = ApiImageStatistics;
export type TextStatistics = ApiTextStatistics;
export type ValidationRequest = ApiValidationRequest;
export type ValidationReport = ApiValidationReportResponse;

/** View model extracted from a completed dataset-validation job result. */
export interface ValidationResponse {
    success: boolean;
    message: string;
    pixel_distribution?: PixelDistribution | null;
    image_statistics?: ImageStatistics | null;
    text_statistics?: TextStatistics | null;
}
