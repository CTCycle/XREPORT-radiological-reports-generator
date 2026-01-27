import { X, Calendar, Sliders, ListChecks } from 'lucide-react';
import ValidationDashboard from './ValidationDashboard';
import { ValidationResponse } from '../services/validationService';
import './ValidationReportModal.css';

interface ValidationReportModalProps {
    isOpen: boolean;
    datasetName: string | null;
    isLoading: boolean;
    validationResult: ValidationResponse | null;
    error: string | null;
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
    metadata,
    onClose,
}: ValidationReportModalProps) {
    if (!isOpen) return null;

    const metrics = formatMetrics(metadata?.metrics);
    const sampleLabel = formatSampleSize(metadata?.sampleSize);
    const dateLabel = metadata?.date ? `Generated: ${metadata.date}` : 'Generated: N/A';

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="report-modal" onClick={(e) => e.stopPropagation()}>
                <div className="report-header">
                    <div>
                        <h3>Validation Report</h3>
                        <p className="report-subtitle">
                            Dataset: <strong>{datasetName || 'Unknown'}</strong>
                        </p>
                    </div>
                    <button className="report-close" onClick={onClose} aria-label="Close validation report">
                        <X size={18} />
                    </button>
                </div>

                <div className="report-meta">
                    <div className="report-chip">
                        <Calendar size={14} />
                        <span>{dateLabel}</span>
                    </div>
                    <div className="report-chip">
                        <Sliders size={14} />
                        <span>{sampleLabel}</span>
                    </div>
                    <div className="report-chip">
                        <ListChecks size={14} />
                        <span>{metrics.length} metrics</span>
                    </div>
                </div>

                <div className="report-metrics">
                    {metrics.map(metric => (
                        <span key={metric} className="report-metric-pill">
                            {metric}
                        </span>
                    ))}
                </div>

                <div className="report-body">
                    <ValidationDashboard
                        isLoading={isLoading}
                        validationResult={validationResult}
                        error={error}
                    />
                </div>
            </div>
        </div>
    );
}
