import { Loader } from 'lucide-react';
import './ValidationDashboard.css';
import { CheckpointEvaluationResults } from '../services/inferenceService';
import { useJobProgressState, JobExecutionStatus } from '../hooks/useJobProgressState';
import JobProgress from './shared/JobProgress';
import DashboardStatusHeader from './shared/DashboardStatusHeader';

interface CheckpointEvaluationDashboardProps {
    isLoading: boolean;
    results: CheckpointEvaluationResults | null;
    error: string | null;
    progress?: number | null;
    status?: JobExecutionStatus;
}

function formatMetric(value: number | undefined, decimals: number) {
    return typeof value === 'number' ? value.toFixed(decimals) : '--';
}

export default function CheckpointEvaluationDashboard({
    isLoading,
    results,
    error,
    progress,
    status,
}: CheckpointEvaluationDashboardProps) {
    const hasResults = results && (
        typeof results.loss === 'number' ||
        typeof results.accuracy === 'number' ||
        typeof results.bleu_score === 'number'
    );
    const { isRunning, showProgress } = useJobProgressState(isLoading, status);

    return (
        <div className="validation-dashboard">
            <DashboardStatusHeader
                title="Checkpoint Evaluation Results"
                hasResults={Boolean(hasResults)}
                isRunning={isRunning}
                error={error}
            />

            <JobProgress show={showProgress} progress={progress} status={status} />

            {isLoading ? (
                <div className="loading-container">
                    <Loader size={32} className="spin" />
                    <span className="loading-text">Running checkpoint evaluation...</span>
                </div>
            ) : error ? (
                <div className="idle-message error">
                    {error}
                </div>
            ) : hasResults ? (
                <div className="validation-content">
                    <div className="stats-grid">
                        <div className="stats-section">
                            <div className="stats-section-title">
                                Metrics
                            </div>
                            <div className="stats-row">
                                <span className="stat-label">Loss</span>
                                <span className="stat-value">{formatMetric(results?.loss, 4)}</span>
                            </div>
                            <div className="stats-row">
                                <span className="stat-label">Accuracy</span>
                                <span className="stat-value">{formatMetric(results?.accuracy, 4)}</span>
                            </div>
                            <div className="stats-row">
                                <span className="stat-label">BLEU Score</span>
                                <span className="stat-value">{formatMetric(results?.bleu_score, 4)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="idle-message">
                    Select evaluation options and run the checkpoint evaluation to see results.
                </div>
            )}
        </div>
    );
}
