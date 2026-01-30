import { Activity, Square, Clock, TrendingDown, Target, Percent } from 'lucide-react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ReferenceLine,
} from 'recharts';
import './TrainingDashboard.css';
import { TrainingDashboardState } from '../types';

interface TrainingDashboardProps {
    onStopTraining?: () => void;
    dashboardState: TrainingDashboardState;
}

function formatTime(seconds: number): string {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
        return `${hrs}h ${mins}m ${secs}s`;
    }
    if (mins > 0) {
        return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
}

// Chart colors for different metrics
const CHART_COLORS = {
    loss: '#d97706',
    val_loss: '#ef4444',
    MaskedAccuracy: '#2563eb',
    val_MaskedAccuracy: '#0d9488',
    accuracy: '#2563eb',
    val_accuracy: '#0d9488',
};

export default function TrainingDashboard({
    onStopTraining,
    dashboardState
}: TrainingDashboardProps) {
    // Extract state from props
    const { chartData, availableMetrics, epochBoundaries } = dashboardState;
    const metrics = {
        isTraining: dashboardState.isTraining,
        currentEpoch: dashboardState.currentEpoch,
        totalEpochs: dashboardState.totalEpochs,
        loss: dashboardState.loss,
        valLoss: dashboardState.valLoss,
        accuracy: dashboardState.accuracy,
        valAccuracy: dashboardState.valAccuracy,
        progressPercent: dashboardState.progressPercent,
        elapsedSeconds: dashboardState.elapsedSeconds,
        logEntries: dashboardState.logEntries,
    };

    const handleStopTraining = () => {
        if (onStopTraining) {
            onStopTraining();
        }
    };

    // Group metrics into loss and accuracy for separate charts
    const lossMetrics = availableMetrics.filter(m => m.toLowerCase().includes('loss'));
    const accuracyMetrics = availableMetrics.filter(m =>
        m.toLowerCase().includes('accuracy') || m.toLowerCase().includes('maskedaccuracy')
    );

    return (
        <div className="training-dashboard">
            <div className="dashboard-header">
                <div className="dashboard-title">
                    <Activity size={20} />
                    Training Dashboard
                </div>
                <div className={`dashboard-status ${metrics.isTraining ? 'training' : 'idle'}`}>
                    <div className={`status-indicator ${metrics.isTraining ? 'training' : 'idle'}`} />
                    {metrics.isTraining ? 'Training in Progress' : 'Idle'}
                </div>
            </div>

            <div className="dashboard-metrics-grid">
                <div className="dashboard-metric-card">
                    <div className="metric-label">Epoch</div>
                    <div className="metric-value">
                        {metrics.currentEpoch} / {metrics.totalEpochs || '--'}
                    </div>
                </div>
                <div className="dashboard-metric-card">
                    <div className="metric-label">
                        <TrendingDown size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        Train Loss
                    </div>
                    <div className="metric-value loss">
                        {metrics.loss.toFixed(3)}
                    </div>
                </div>
                <div className="dashboard-metric-card">
                    <div className="metric-label">
                        <TrendingDown size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        Val Loss
                    </div>
                    <div className="metric-value loss">
                        {metrics.valLoss.toFixed(3)}
                    </div>
                </div>
                <div className="dashboard-metric-card">
                    <div className="metric-label">
                        <Target size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        Train Acc
                    </div>
                    <div className="metric-value accuracy">
                        {(metrics.accuracy * 100).toFixed(3)}%
                    </div>
                </div>
                <div className="dashboard-metric-card">
                    <div className="metric-label">
                        <Target size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        Val Acc
                    </div>
                    <div className="metric-value accuracy">
                        {(metrics.valAccuracy * 100).toFixed(3)}%
                    </div>
                </div>
            </div>

            <div className="progress-section">
                <div className="progress-header">
                    <span className="progress-label">
                        <Percent size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        Progress: {metrics.progressPercent}%
                    </span>
                    <span className="progress-time">
                        <Clock size={14} style={{ display: 'inline', marginRight: '4px' }} />
                        {formatTime(metrics.elapsedSeconds)}
                    </span>
                </div>
                <div className="progress-bar-row">
                    <div className="progress-bar-container">
                        <div
                            className="progress-bar"
                            style={{ width: `${metrics.progressPercent}%` }}
                        />
                    </div>
                    {metrics.isTraining && (
                        <button
                            className="btn-stop"
                            onClick={handleStopTraining}
                        >
                            <Square size={16} />
                            Stop Training
                        </button>
                    )}
                </div>
            </div>



            <div className="training-charts-container">
                {chartData.length > 0 ? (
                    <>
                        {/* Loss Chart */}
                        {lossMetrics.length > 0 && (
                            <div className="chart-section">
                                <div className="chart-title">Loss</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.4)" />
                                        <XAxis
                                            dataKey="batch"
                                            stroke="#94a3b8"
                                            tick={{ fill: '#64748b', fontSize: 12 }}
                                            tickFormatter={(value) => Math.round(value).toString()}
                                        />
                                        <YAxis
                                            stroke="#94a3b8"
                                            tick={{ fill: '#64748b', fontSize: 12 }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                background: '#ffffff',
                                                border: '1px solid #e2e8f0',
                                                borderRadius: '8px',
                                                color: '#0f172a',
                                            }}
                                            labelFormatter={(value) => `Epoch ${Math.round(value)}`}
                                        />
                                        <Legend />
                                        {epochBoundaries.map((boundary, index) => (
                                            <ReferenceLine
                                                key={`epoch-${index}`}
                                                x={boundary}
                                                stroke="rgba(148, 163, 184, 0.6)"
                                                strokeDasharray="3 3"
                                                label={{
                                                    value: `E${index + 1}`,
                                                    position: 'top',
                                                    fill: '#94a3b8',
                                                    fontSize: 10,
                                                }}
                                            />
                                        ))}
                                        {lossMetrics.map((metric) => (
                                            <Line
                                                key={metric}
                                                type="monotone"
                                                dataKey={metric}
                                                stroke={CHART_COLORS[metric as keyof typeof CHART_COLORS] || '#ffd700'}
                                                strokeWidth={2}
                                                dot={false}
                                                name={metric.replace('_', ' ')}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        )}

                        {/* Accuracy Chart */}
                        {accuracyMetrics.length > 0 && (
                            <div className="chart-section">
                                <div className="chart-title">Accuracy</div>
                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.4)" />
                                        <XAxis
                                            dataKey="batch"
                                            stroke="#94a3b8"
                                            tick={{ fill: '#64748b', fontSize: 12 }}
                                            tickFormatter={(value) => Math.round(value).toString()}
                                        />
                                        <YAxis
                                            stroke="#94a3b8"
                                            tick={{ fill: '#64748b', fontSize: 12 }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                background: '#ffffff',
                                                border: '1px solid #e2e8f0',
                                                borderRadius: '8px',
                                                color: '#0f172a',
                                            }}
                                            labelFormatter={(value) => `Epoch ${Math.round(value)}`}
                                        />
                                        <Legend />
                                        {epochBoundaries.map((boundary, index) => (
                                            <ReferenceLine
                                                key={`epoch-${index}`}
                                                x={boundary}
                                                stroke="rgba(148, 163, 184, 0.6)"
                                                strokeDasharray="3 3"
                                                label={{
                                                    value: `E${index + 1}`,
                                                    position: 'top',
                                                    fill: '#94a3b8',
                                                    fontSize: 10,
                                                }}
                                            />
                                        ))}
                                        {accuracyMetrics.map((metric) => (
                                            <Line
                                                key={metric}
                                                type="monotone"
                                                dataKey={metric}
                                                stroke={CHART_COLORS[metric as keyof typeof CHART_COLORS] || '#22c55e'}
                                                strokeWidth={2}
                                                dot={false}
                                                name={metric.replace('_', ' ')}
                                            />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </>
                ) : (
                    <>
                        <div className="chart-section">
                            <div className="chart-title">Loss</div>
                            <div className="chart-placeholder">
                                Waiting for training data...
                            </div>
                        </div>
                        <div className="chart-section">
                            <div className="chart-title">Accuracy</div>
                            <div className="chart-placeholder">
                                Waiting for training data...
                            </div>
                        </div>
                    </>
                )}
            </div>

            <div className="dashboard-logs">
                <div className="log-header">Training Log</div>
                {metrics.logEntries.length > 0 ? (
                    <pre className="log-body">
                        {metrics.logEntries.join('\n')}
                    </pre>
                ) : (
                    <div className="log-empty">No training output yet.</div>
                )}
            </div>
        </div>
    );
}
