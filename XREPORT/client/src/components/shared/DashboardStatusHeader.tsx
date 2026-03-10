import { AlertCircle, BarChart2, CheckCircle, Loader } from 'lucide-react';

interface DashboardStatusHeaderProps {
    title: string;
    hasResults: boolean;
    isRunning: boolean;
    error: string | null;
}

export default function DashboardStatusHeader({
    title,
    hasResults,
    isRunning,
    error,
}: DashboardStatusHeaderProps) {
    return (
        <div className="dashboard-header">
            <div className="dashboard-title">
                <BarChart2 size={20} />
                {title}
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
    );
}
