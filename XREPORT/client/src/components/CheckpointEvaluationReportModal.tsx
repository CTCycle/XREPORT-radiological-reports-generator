import { Calendar, ListChecks } from 'lucide-react';
import CheckpointEvaluationDashboard from './CheckpointEvaluationDashboard';
import { CheckpointEvaluationReport } from '../services/inferenceService';
import ReportModalLayout from './shared/ReportModalLayout';
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
    const metrics = report?.metrics ?? [];
    const metricConfigs = report?.metric_configs ?? {};
    const dateLabel = report?.date ? `Generated: ${report.date}` : 'Generated: N/A';
    const metricPills = (metrics.length > 0 ? metrics : ['No metrics recorded']).map(metric => (
        metric === 'No metrics recorded'
            ? metric
            : formatMetricLabel(metric, metricConfigs[metric])
    ));
    const chips = [
        {
            id: 'date',
            icon: <Calendar size={14} />,
            text: dateLabel,
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
            title="Checkpoint Evaluation Report"
            subtitleLabel="Checkpoint"
            subtitleValue={checkpointName || 'Unknown'}
            chips={chips}
            metrics={metricPills}
            onClose={onClose}
        >
            <CheckpointEvaluationDashboard
                isLoading={isLoading}
                results={report?.results ?? null}
                error={error}
                progress={progress}
                status={status}
            />
        </ReportModalLayout>
    );
}
