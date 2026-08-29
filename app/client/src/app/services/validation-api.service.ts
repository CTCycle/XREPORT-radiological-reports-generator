import { Injectable, inject } from '@angular/core';
import { asRecord, readBoolean, readNumber, readNumberArray, readString } from '../common/parsers';
import { CheckpointEvaluationReport, CheckpointEvaluationRequest } from '../types/inferenceApi';
import {
  ImageStatistics,
  PixelDistribution,
  TextStatistics,
  ValidationReport,
  ValidationRequest,
  ValidationResponse,
} from '../types/validationApi';
import { JobStartResponse } from '../types/jobs';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class ValidationApiService {
  private readonly request = inject(ApiRequestService);

  evaluateCheckpoint(checkpoint: string, metrics: string[], numSamples = 10, metricConfigs?: Record<string, { data_fraction?: number; num_samples?: number }>, seed?: number) {
    const evaluationRequest: CheckpointEvaluationRequest = { checkpoint, metrics, num_samples: numSamples, metric_configs: metricConfigs, seed };
    return this.request.request<JobStartResponse>('POST', '/api/validation/checkpoint', evaluationRequest);
  }
  getCheckpointEvaluationReport(checkpoint: string) { return this.request.request<CheckpointEvaluationReport>('GET', `/api/validation/checkpoint/reports/${encodeURIComponent(checkpoint)}`); }
  run(request: ValidationRequest) { return this.request.request<JobStartResponse>('POST', '/api/validation/run', request); }
  getReport(datasetName: string) { return this.request.request<ValidationReport>('GET', `/api/validation/reports/${encodeURIComponent(datasetName)}`); }

  parseResponse(result: Record<string, unknown>): ValidationResponse {
    const pixel = asRecord(result['pixel_distribution']);
    const image = asRecord(result['image_statistics']);
    const text = asRecord(result['text_statistics']);
    const bins = pixel ? readNumberArray(pixel['bins']) : undefined;
    const counts = pixel ? readNumberArray(pixel['counts']) : undefined;
    const pixelDistribution: PixelDistribution | undefined = bins && counts ? { bins, counts } : undefined;
    const imageStatistics: ImageStatistics | undefined = image ? this.parseImageStatistics(image) : undefined;
    const textStatistics: TextStatistics | undefined = text ? this.parseTextStatistics(text) : undefined;
    return { success: readBoolean(result['success']) ?? true, message: readString(result['message']) ?? 'Validation completed', pixel_distribution: pixelDistribution, image_statistics: imageStatistics, text_statistics: textStatistics };
  }

  private parseImageStatistics(payload: Record<string, unknown>): ImageStatistics | undefined {
    const values = ['count', 'mean_height', 'mean_width', 'mean_pixel_value', 'std_pixel_value', 'mean_noise_std', 'mean_noise_ratio'].map((key) => readNumber(payload[key]));
    return values.every((value): value is number => value !== undefined) ? { count: values[0], mean_height: values[1], mean_width: values[2], mean_pixel_value: values[3], std_pixel_value: values[4], mean_noise_std: values[5], mean_noise_ratio: values[6] } : undefined;
  }

  private parseTextStatistics(payload: Record<string, unknown>): TextStatistics | undefined {
    const values = ['count', 'total_words', 'unique_words', 'avg_words_per_report', 'min_words_per_report', 'max_words_per_report'].map((key) => readNumber(payload[key]));
    return values.every((value): value is number => value !== undefined) ? { count: values[0], total_words: values[1], unique_words: values[2], avg_words_per_report: values[3], min_words_per_report: values[4], max_words_per_report: values[5] } : undefined;
  }
}
