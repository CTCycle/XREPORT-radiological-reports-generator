import { BarChart2, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import './ValidationDashboard.css';
import { CheckpointEvaluationResults } from '../services/inferenceService';
import JobProgress from './shared/JobProgress';

interface CheckpointEvaluationDashboardProps {
    isLoading: boolean;
    results: CheckpointEvaluationResults | null;
    error: string | null;
    progress?: number | null;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null;
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
    const isRunning = status === 'running' || status === 'pending';
    const showProgress = isLoading || isRunning;

    return (
        <div className="validation-dashboard">
            <div className="dashboard-header">
                <div className="dashboard-title">
                    <BarChart2 size={20} />
                    Checkpoint Evaluation Results
                </div>
                {hasResults && (
                    <div className="dashboard-status success">
                        <CheckCircle size={14} />
                        Complete
                    </div>
                )}
                {isRunning && !hasResults && (
                    <div className="dashboard-status running">
                        <Loader size={14} className="spin" />
                        Running
                    </div>
                )}
                {error && (
                    <div className="dashboard-status error">
                        <AlertCircle size={14} />
                        Error
                    </div>
                )}
            </div>

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
