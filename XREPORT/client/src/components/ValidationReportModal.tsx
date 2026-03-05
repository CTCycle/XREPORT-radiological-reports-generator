import { Calendar, Sliders, ListChecks } from 'lucide-react';
import ValidationDashboard from './ValidationDashboard';
import { ValidationResponse } from '../services/validationService';
import ReportModalLayout from './shared/ReportModalLayout';
import './ValidationReportModal.css';

interface ValidationReportModalProps {
    isOpen: boolean;
    datasetName: string | null;
    isLoading: boolean;
    validationResult: ValidationResponse | null;
    error: string | null;
    progress?: number | null;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null;
    metadata?: {
        date?: string | null;
        sampleSize?: number | null;
        metrics?: string[];
    } | null;
    onClose: () => void;
}

const METRIC_LABELS: Record<string, string> = {
    pixels_distribution: 'Pixel intensity histogram',
    text_statistics: 'Text statistics',
    image_statistics: 'Image statistics',
};

function formatMetrics(metrics: string[] | undefined) {
    if (!metrics || metrics.length === 0) {
        return ['No metrics recorded'];
    }
    return metrics.map(metric => METRIC_LABELS[metric] || metric.replace(/_/g, ' '));
}

function formatSampleSize(sampleSize: number | null | undefined) {
    if (sampleSize === null || sampleSize === undefined) {
        return 'Sample size: N/A';
    }
    const pct = Math.max(0, Math.min(sampleSize, 1)) * 100;
    return `Sample size: ${pct.toFixed(0)}%`;
}

export default function ValidationReportModal({
    isOpen,
    datasetName,
    isLoading,
    validationResult,
    error,
    progress,
    status,
    metadata,
    onClose,
}: ValidationReportModalProps) {
    const metrics = formatMetrics(metadata?.metrics);
    const sampleLabel = formatSampleSize(metadata?.sampleSize);
    const dateLabel = metadata?.date ? `Generated: ${metadata.date}` : 'Generated: N/A';
    const chips = [
        {
            id: 'date',
            icon: <Calendar size={14} />,
            text: dateLabel,
        },
        {
            id: 'sample',
            icon: <Sliders size={14} />,
            text: sampleLabel,
        },
        {
            id: 'metrics',
            icon: <ListChecks size={14} />,
            text: `${metrics.length} metrics`,
        },
    ];

    return (
        <ReportModalLayout
            isOpen={isOpen}
            title="Validation Report"
            subtitleLabel="Dataset"
            subtitleValue={datasetName || 'Unknown'}
            chips={chips}
            metrics={metrics}
            onClose={onClose}
        >
            <ValidationDashboard
                isLoading={isLoading}
                validationResult={validationResult}
                error={error}
                progress={progress}
                status={status}
            />
        </ReportModalLayout>
    );
}
