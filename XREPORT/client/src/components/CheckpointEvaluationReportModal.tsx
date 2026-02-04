import { X, Calendar, ListChecks } from 'lucide-react';
import CheckpointEvaluationDashboard from './CheckpointEvaluationDashboard';
import { CheckpointEvaluationReport } from '../services/inferenceService';
import './ValidationReportModal.css';

interface CheckpointEvaluationReportModalProps {
    isOpen: boolean;
    checkpointName: string | null;
    isLoading: boolean;
    report: CheckpointEvaluationReport | null;
    error: string | null;
    progress?: number | null;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null;
    onClose: () => void;
}

const METRIC_LABELS: Record<string, string> = {
    evaluation_report: 'Evaluation report',
    bleu_score: 'BLEU score',
};

function formatMetricLabel(
    metric: string,
    config: { data_fraction?: number; num_samples?: number } | undefined
) {
    const label = METRIC_LABELS[metric] || metric.replace(/_/g, ' ');
    const parts: string[] = [];
    if (typeof config?.data_fraction === 'number') {
        parts.push(`${Math.round(config.data_fraction * 100)}%`);
    }
    if (typeof config?.num_samples === 'number') {
        parts.push(`${config.num_samples} samples`);
    }
    if (parts.length === 0) {
        return label;
    }
    return `${label} · ${parts.join(' · ')}`;
}

export default function CheckpointEvaluationReportModal({
    isOpen,
    checkpointName,
    isLoading,
    report,
    error,
    progress,
    status,
    onClose,
}: CheckpointEvaluationReportModalProps) {
    if (!isOpen) return null;

    const metrics = report?.metrics ?? [];
    const metricConfigs = report?.metric_configs ?? {};
    const dateLabel = report?.date ? `Generated: ${report.date}` : 'Generated: N/A';

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="report-modal" onClick={(e) => e.stopPropagation()}>
                <div className="report-header">
                    <div>
                        <h3>Checkpoint Evaluation Report</h3>
                        <p className="report-subtitle">
                            Checkpoint: <strong>{checkpointName || 'Unknown'}</strong>
                        </p>
                    </div>
                    <button className="report-close" onClick={onClose} aria-label="Close evaluation report">
                        <X size={18} />
                    </button>
                </div>

                <div className="report-meta">
                    <div className="report-chip">
                        <Calendar size={14} />
                        <span>{dateLabel}</span>
                    </div>
                    <div className="report-chip">
                        <ListChecks size={14} />
                        <span>{metrics.length} metrics</span>
                    </div>
                </div>

                <div className="report-metrics">
                    {(metrics.length > 0 ? metrics : ['No metrics recorded']).map(metric => (
                        <span key={metric} className="report-metric-pill">
                            {metric === 'No metrics recorded'
                                ? metric
                                : formatMetricLabel(metric, metricConfigs[metric])}
                        </span>
                    ))}
                </div>

                <div className="report-body">
                    <CheckpointEvaluationDashboard
                        isLoading={isLoading}
                        results={report?.results ?? null}
                        error={error}
                        progress={progress}
                        status={status}
                    />
                </div>
            </div>
        </div>
    );
}
