import { DatasetInfo } from './trainingApi';

export type ValidationMetric = 'pixels_distribution' | 'text_statistics' | 'image_statistics';

export interface ValidationWizardConfirmPayload {
    metrics: ValidationMetric[];
    row: DatasetInfo | null;
    sampleFraction: number;
}
