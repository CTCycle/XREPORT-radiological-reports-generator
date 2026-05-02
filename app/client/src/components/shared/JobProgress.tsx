interface JobProgressProps {
    show: boolean;
    progress?: number | null;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null;
}

export default function JobProgress({ show, progress, status }: JobProgressProps) {
    if (!show) {
        return null;
    }

    const progressValue = Number.isFinite(progress ?? NaN)
        ? Math.min(100, Math.max(0, progress ?? 0))
        : 0;
    const statusLabel = status ? status.replace(/_/g, ' ') : 'running';

    return (
        <div className="validation-progress">
            <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progressValue}%` }}></div>
            </div>
            <div className="progress-meta">
                <span>{progressValue.toFixed(0)}%</span>
                <span className="progress-status">{statusLabel}</span>
            </div>
        </div>
    );
}
