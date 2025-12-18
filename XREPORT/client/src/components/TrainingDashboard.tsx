import { useState, useEffect, useRef, useCallback } from 'react';
import { Activity, Square, Clock, TrendingDown, Target, Percent } from 'lucide-react';
import './TrainingDashboard.css';

export interface TrainingMetrics {
    isTraining: boolean;
    currentEpoch: number;
    totalEpochs: number;
    loss: number;
    valLoss: number;
    accuracy: number;
    valAccuracy: number;
    progressPercent: number;
    elapsedSeconds: number;
}

interface TrainingDashboardProps {
    onStopTraining?: () => void;
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

export default function TrainingDashboard({ onStopTraining }: TrainingDashboardProps) {
    const [metrics, setMetrics] = useState<TrainingMetrics>({
        isTraining: false,
        currentEpoch: 0,
        totalEpochs: 0,
        loss: 0,
        valLoss: 0,
        accuracy: 0,
        valAccuracy: 0,
        progressPercent: 0,
        elapsedSeconds: 0,
    });
    const [connected, setConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);

    const connectWebSocket = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/pipeline/ws`;

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('Training WebSocket connected');
            setConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'training_update') {
                    setMetrics({
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
                    setMetrics(prev => ({
                        ...prev,
                        isTraining: true,
                        totalEpochs: data.total_epochs || prev.totalEpochs,
                    }));
                } else if (data.type === 'training_completed' || data.type === 'training_error') {
                    setMetrics(prev => ({
                        ...prev,
                        isTraining: false,
                        progressPercent: data.type === 'training_completed' ? 100 : prev.progressPercent,
                    }));
                } else if (data.type === 'connection_established') {
                    setMetrics(prev => ({
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
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        ws.onclose = () => {
            console.log('Training WebSocket disconnected');
            setConnected(false);
            wsRef.current = null;

            // Reconnect after 3 seconds
            reconnectTimeoutRef.current = window.setTimeout(() => {
                connectWebSocket();
            }, 3000);
        };

        ws.onerror = (error) => {
            console.error('Training WebSocket error:', error);
        };
    }, []);

    useEffect(() => {
        connectWebSocket();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connectWebSocket]);

    const handleStopTraining = () => {
        if (onStopTraining) {
            onStopTraining();
        }
    };

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
                                Accuracy
                            </div>
                            <div className="metric-value accuracy">
                                {(metrics.accuracy * 100).toFixed(2)}%
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
                </>
            ) : (
                <div className="idle-message">
                    No training session active. Start a new training or resume from a checkpoint.
                </div>
            )}
        </div>
    );
}
