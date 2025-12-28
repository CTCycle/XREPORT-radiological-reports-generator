import { useState, useEffect, useRef, useCallback } from 'react';
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
import { TrainingDashboardState, ChartDataPoint } from '../types';

interface TrainingDashboardProps {
    onStopTraining?: () => void;
    shouldConnect?: boolean;
    dashboardState: TrainingDashboardState;
    onDashboardStateChange: (updater: Partial<TrainingDashboardState> | ((prev: TrainingDashboardState) => TrainingDashboardState)) => void;
    onChartDataChange: (chartData: ChartDataPoint[]) => void;
    onAvailableMetricsChange: (metrics: string[]) => void;
    onEpochBoundariesChange: (boundaries: number[]) => void;
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
    loss: '#f59e0b',
    val_loss: '#fbbf24',
    MaskedAccuracy: '#22c55e',
    val_MaskedAccuracy: '#4ade80',
    accuracy: '#22c55e',
    val_accuracy: '#4ade80',
};

export default function TrainingDashboard({
    onStopTraining,
    shouldConnect = false,
    dashboardState,
    onDashboardStateChange,
    onChartDataChange,
    onAvailableMetricsChange,
    onEpochBoundariesChange
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
    };

    const [connected, setConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const shouldConnectRef = useRef(shouldConnect);

    // Keep refs updated to avoid stale closures while preventing effect re-runs
    const callbacksRef = useRef({
        onDashboardStateChange,
        onChartDataChange,
        onAvailableMetricsChange,
        onEpochBoundariesChange,
    });

    useEffect(() => {
        callbacksRef.current = {
            onDashboardStateChange,
            onChartDataChange,
            onAvailableMetricsChange,
            onEpochBoundariesChange,
        };
    });

    useEffect(() => {
        shouldConnectRef.current = shouldConnect;
    }, [shouldConnect]);

    const connectWebSocket = useCallback(() => {
        // Don't connect if already connected or connecting
        if (wsRef.current?.readyState === WebSocket.OPEN ||
            wsRef.current?.readyState === WebSocket.CONNECTING) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/training/ws`;

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('Training WebSocket connected');
            setConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const callbacks = callbacksRef.current;

                if (data.type === 'training_update') {
                    callbacks.onDashboardStateChange({
                        isTraining: true,
                        currentEpoch: data.epoch || 0,
                        totalEpochs: data.total_epochs || 0,
                        loss: data.loss || 0,
                        valLoss: data.val_loss || 0,
                        accuracy: data.accuracy || 0,
                        valAccuracy: data.val_accuracy || 0,
                        progressPercent: data.progress_percent || 0,
                        elapsedSeconds: data.elapsed_seconds || 0,
                    });
                } else if (data.type === 'training_started' || data.type === 'training_resumed') {
                    callbacks.onDashboardStateChange(prev => ({
                        ...prev,
                        isTraining: true,
                        totalEpochs: data.total_epochs || prev.totalEpochs,
                    }));
                    // Clear chart data on new training session
                    if (data.type === 'training_started') {
                        callbacks.onChartDataChange([]);
                        callbacks.onAvailableMetricsChange([]);
                    }
                } else if (data.type === 'training_completed' || data.type === 'training_error') {
                    callbacks.onDashboardStateChange(prev => ({
                        ...prev,
                        isTraining: false,
                        progressPercent: data.type === 'training_completed' ? 100 : prev.progressPercent,
                    }));
                } else if (data.type === 'connection_established') {
                    callbacks.onDashboardStateChange(prev => ({
                        ...prev,
                        isTraining: data.is_training || false,
                        currentEpoch: data.current_epoch || 0,
                        totalEpochs: data.total_epochs || 0,
                        loss: data.loss || 0,
                        valLoss: data.val_loss || 0,
                        accuracy: data.accuracy || 0,
                        valAccuracy: data.val_accuracy || 0,
                        progressPercent: data.progress_percent || 0,
                        elapsedSeconds: data.elapsed_seconds || 0,
                    }));
                } else if (data.type === 'ping') {
                    ws.send(JSON.stringify({ type: 'pong' }));
                } else if (data.type === 'training_plot' && data.chart_data) {
                    callbacks.onChartDataChange(data.chart_data);
                    if (data.metrics) {
                        callbacks.onAvailableMetricsChange(data.metrics);
                    }
                    if (data.epoch_boundaries) {
                        callbacks.onEpochBoundariesChange(data.epoch_boundaries);
                    }
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        ws.onclose = () => {
            console.log('Training WebSocket disconnected');
            setConnected(false);
            wsRef.current = null;

            // Only reconnect if we should still be connected
            if (shouldConnectRef.current) {
                reconnectTimeoutRef.current = window.setTimeout(() => {
                    connectWebSocket();
                }, 3000);
            }
        };

        ws.onerror = (error) => {
            console.error('Training WebSocket error:', error);
        };
    }, []); // No dependencies - uses refs for all external values

    useEffect(() => {
        if (shouldConnect) {
            connectWebSocket();
        }

        return () => {
            // Clear reconnect timeout
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
                reconnectTimeoutRef.current = null;
            }
            // Close WebSocket when effect cleans up
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [shouldConnect, connectWebSocket]);

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
                    {metrics.isTraining ? 'Training in Progress' : (connected ? 'Idle' : 'Disconnected')}
                </div>
            </div>

            {metrics.isTraining || metrics.currentEpoch > 0 ? (
                <>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-label">Epoch</div>
                            <div className="metric-value">
                                {metrics.currentEpoch} / {metrics.totalEpochs}
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">
                                <TrendingDown size={14} style={{ display: 'inline', marginRight: '4px' }} />
                                Train Loss
                            </div>
                            <div className="metric-value loss">
                                {metrics.loss.toFixed(4)}
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">
                                <TrendingDown size={14} style={{ display: 'inline', marginRight: '4px' }} />
                                Val Loss
                            </div>
                            <div className="metric-value loss">
                                {metrics.valLoss.toFixed(4)}
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">
                                <Target size={14} style={{ display: 'inline', marginRight: '4px' }} />
                                Train Acc
                            </div>
                            <div className="metric-value accuracy">
                                {(metrics.accuracy * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">
                                <Target size={14} style={{ display: 'inline', marginRight: '4px' }} />
                                Val Acc
                            </div>
                            <div className="metric-value accuracy">
                                {(metrics.valAccuracy * 100).toFixed(2)}%
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
                        <div className="progress-bar-container">
                            <div
                                className="progress-bar"
                                style={{ width: `${metrics.progressPercent}%` }}
                            />
                        </div>
                    </div>

                    {metrics.isTraining && (
                        <div className="dashboard-actions">
                            <button
                                className="btn-stop"
                                onClick={handleStopTraining}
                            >
                                <Square size={16} />
                                Stop Training
                            </button>
                        </div>
                    )}

                    {chartData.length > 0 && (
                        <div className="training-charts-container">
                            {/* Loss Chart */}
                            {lossMetrics.length > 0 && (
                                <div className="chart-section">
                                    <div className="chart-title">Loss</div>
                                    <ResponsiveContainer width="100%" height={200}>
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis
                                                dataKey="batch"
                                                stroke="#9ca3af"
                                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                                tickFormatter={(value) => Math.round(value).toString()}
                                            />
                                            <YAxis
                                                stroke="#9ca3af"
                                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: 'rgba(30, 30, 35, 0.95)',
                                                    border: '1px solid rgba(255, 215, 0, 0.2)',
                                                    borderRadius: '8px',
                                                }}
                                                labelFormatter={(value) => `Batch ${Math.round(value)}`}
                                            />
                                            <Legend />
                                            {epochBoundaries.map((boundary, index) => (
                                                <ReferenceLine
                                                    key={`epoch-${index}`}
                                                    x={boundary}
                                                    stroke="rgba(255,255,255,0.3)"
                                                    strokeDasharray="3 3"
                                                    label={{
                                                        value: `E${index + 1}`,
                                                        position: 'top',
                                                        fill: '#9ca3af',
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
                                    <ResponsiveContainer width="100%" height={200}>
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis
                                                dataKey="batch"
                                                stroke="#9ca3af"
                                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                                tickFormatter={(value) => Math.round(value).toString()}
                                            />
                                            <YAxis
                                                stroke="#9ca3af"
                                                tick={{ fill: '#9ca3af', fontSize: 12 }}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: 'rgba(30, 30, 35, 0.95)',
                                                    border: '1px solid rgba(255, 215, 0, 0.2)',
                                                    borderRadius: '8px',
                                                }}
                                                labelFormatter={(value) => `Batch ${Math.round(value)}`}
                                            />
                                            <Legend />
                                            {epochBoundaries.map((boundary, index) => (
                                                <ReferenceLine
                                                    key={`epoch-${index}`}
                                                    x={boundary}
                                                    stroke="rgba(255,255,255,0.3)"
                                                    strokeDasharray="3 3"
                                                    label={{
                                                        value: `E${index + 1}`,
                                                        position: 'top',
                                                        fill: '#9ca3af',
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
                        </div>
                    )}
                </>
            ) : (
                <div className="idle-message">
                    No training session active. Start a new training or resume from a checkpoint.
                </div>
            )}
        </div>
    );
}
