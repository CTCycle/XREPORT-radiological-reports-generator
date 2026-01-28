import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Activity,
    ChevronLeft,
    ChevronRight,
    Cpu,
    Info,
    Play,
    RefreshCw,
    RotateCcw,
    Settings,
    Trash2,
    X,
} from 'lucide-react';
import './TrainingPage.css';
import { useTrainingPageState } from '../AppStateContext';
import TrainingDashboard from '../components/TrainingDashboard';
import { ChartDataPoint, TrainingConfig } from '../types';
import {
    CheckpointInfo,
    DatasetInfo,
    JobStatusResponse,
    StartTrainingConfig,
    deleteCheckpoint,
    deleteDataset,
    getCheckpointMetadata,
    getCheckpoints,
    getDatasetNames,
    getProcessingMetadata,
    getTrainingJobStatus,
    getTrainingStatus,
    pollJobStatus,
    resumeTraining,
    startTraining,
    stopTraining,
} from '../services/trainingService';

type MetadataEntry = {
    label: string;
    value: string;
};

type MetadataSection = {
    title: string;
    entries: MetadataEntry[];
};

type MetadataModalState = {
    title: string;
    subtitle?: string;
    sections?: MetadataSection[];
    error?: string;
};

const formatMetadataValue = (value: unknown) => {
    if (value === null || value === undefined || value === '') {
        return 'N/A';
    }
    if (typeof value === 'object') {
        try {
            return JSON.stringify(value, null, 2);
        } catch {
            return String(value);
        }
    }
    return String(value);
};

const stripHashFields = (data: Record<string, unknown>) => {
    return Object.fromEntries(
        Object.entries(data).filter(([key]) => !key.toLowerCase().includes('hash'))
    );
};

const buildEntries = (data: Record<string, unknown>): MetadataEntry[] => {
    const sanitized = stripHashFields(data);
    return Object.entries(sanitized).map(([label, value]) => ({
        label,
        value: formatMetadataValue(value),
    }));
};

function MetadataModal({
    state,
    onClose,
}: {
    state: MetadataModalState | null;
    onClose: () => void;
}) {
    if (!state) return null;

    return (
        <div className="metadata-backdrop" onClick={onClose}>
            <div className="metadata-modal" onClick={(event) => event.stopPropagation()}>
                <div className="metadata-header">
                    <div>
                        <h3>{state.title}</h3>
                        {state.subtitle && <p className="metadata-subtitle">{state.subtitle}</p>}
                    </div>
                    <button className="metadata-close" onClick={onClose} aria-label="Close metadata dialog">
                        <X size={18} />
                    </button>
                </div>
                <div className="metadata-body">
                    {state.error && <div className="metadata-error">{state.error}</div>}
                    {!state.error && state.sections?.map((section) => (
                        <div className="metadata-section" key={section.title}>
                            <h4>{section.title}</h4>
                            <div className="metadata-grid">
                                {section.entries.map((entry) => (
                                    <div className="metadata-row" key={`${section.title}-${entry.label}`}>
                                        <span className="metadata-label">{entry.label}</span>
                                        <span className="metadata-value">{entry.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function WizardSteps({ steps, current }: { steps: string[]; current: number }) {
    return (
        <div className="wizard-steps">
            {steps.map((step, index) => {
                const isActive = index === current;
                const isComplete = index < current;
                return (
                    <div className="wizard-step" key={step}>
                        <span className={`wizard-step-dot ${isActive || isComplete ? 'active' : ''}`}>
                            {index + 1}
                        </span>
                        <span className={`wizard-step-label ${isActive ? 'active' : ''}`}>{step}</span>
                        {index < steps.length - 1 && (
                            <span className={`wizard-step-line ${isComplete ? 'active' : ''}`} />
                        )}
                    </div>
                );
            })}
        </div>
    );
}

interface NewTrainingWizardProps {
    isOpen: boolean;
    config: TrainingConfig;
    onConfigChange: (key: keyof TrainingConfig, value: TrainingConfig[keyof TrainingConfig]) => void;
    onClose: () => void;
    onConfirm: () => void;
    isLoading: boolean;
    selectedDatasetLabel: string;
    error: string | null;
}

function NewTrainingWizard({
    isOpen,
    config,
    onConfigChange,
    onClose,
    onConfirm,
    isLoading,
    selectedDatasetLabel,
    error,
}: NewTrainingWizardProps) {
    const steps = ['Model', 'Dataset', 'Training'];
    const [currentPage, setCurrentPage] = useState(0);

    useEffect(() => {
        if (isOpen) {
            setCurrentPage(0);
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const isLastPage = currentPage === steps.length - 1;
    const canConfirm = Boolean(selectedDatasetLabel);

    return (
        <div className="training-modal-backdrop" onClick={onClose}>
            <div className="training-wizard-modal" onClick={(event) => event.stopPropagation()}>
                <div className="training-wizard-header">
                    <div>
                        <h3>New Training Wizard</h3>
                        <p>Dataset: <strong>{selectedDatasetLabel || 'No dataset selected'}</strong></p>
                    </div>
                    <button className="training-wizard-close" onClick={onClose} aria-label="Close wizard">
                        <X size={18} />
                    </button>
                </div>
                <WizardSteps steps={steps} current={currentPage} />

                <div className="training-wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Cpu size={16} />
                                <span>Model Architecture</span>
                            </div>
                            <div className="wizard-grid">
                                <div className="form-group">
                                    <label className="form-label">Encoders</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.numEncoders}
                                        onChange={(e) => onConfigChange('numEncoders', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Decoders</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.numDecoders}
                                        onChange={(e) => onConfigChange('numDecoders', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Embedding Dims</label>
                                    <input
                                        type="number"
                                        step="8"
                                        className="form-input"
                                        value={config.embeddingDims}
                                        onChange={(e) => onConfigChange('embeddingDims', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Attention Heads</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.attnHeads}
                                        onChange={(e) => onConfigChange('attnHeads', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Temperature</label>
                                    <input
                                        type="number"
                                        step="0.05"
                                        className="form-input"
                                        value={config.trainTemp}
                                        onChange={(e) => onConfigChange('trainTemp', parseFloat(e.target.value))}
                                    />
                                </div>
                            </div>
                            <div className="wizard-toggle-grid">
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.freezeImgEncoder}
                                        onChange={(e) => onConfigChange('freezeImgEncoder', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Freeze Encoder</span>
                                </label>
                            </div>
                        </div>
                    )}

                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Settings size={16} />
                                <span>Dataset Configuration</span>
                            </div>
                            <div className="wizard-toggle-grid">
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.useImgAugment}
                                        onChange={(e) => onConfigChange('useImgAugment', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Image Augmentation</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.shuffleWithBuffer}
                                        onChange={(e) => onConfigChange('shuffleWithBuffer', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Shuffle Buffered</span>
                                </label>
                            </div>
                            {config.shuffleWithBuffer && (
                                <div className="wizard-grid">
                                    <div className="form-group">
                                        <label className="form-label">Buffer Size</label>
                                        <input
                                            type="number"
                                            step="10"
                                            className="form-input"
                                            value={config.shuffleBufferSize}
                                            onChange={(e) => onConfigChange('shuffleBufferSize', parseInt(e.target.value, 10))}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {currentPage === 2 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Activity size={16} />
                                <span>Training Parameters</span>
                            </div>
                            <div className="wizard-grid">
                                <div className="form-group">
                                    <label className="form-label">Epochs</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.epochs}
                                        onChange={(e) => onConfigChange('epochs', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Batch Size</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.batchSize}
                                        onChange={(e) => onConfigChange('batchSize', parseInt(e.target.value, 10))}
                                    />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Training Seed</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={config.trainSeed}
                                        onChange={(e) => onConfigChange('trainSeed', parseInt(e.target.value, 10))}
                                    />
                                </div>
                            </div>
                            <div className="wizard-toggle-grid">
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.saveCheckpoints}
                                        onChange={(e) => onConfigChange('saveCheckpoints', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Save Checkpoints</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.runTensorboard}
                                        onChange={(e) => onConfigChange('runTensorboard', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Tensorboard</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.mixedPrecision}
                                        onChange={(e) => onConfigChange('mixedPrecision', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Mixed Precision</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={config.useScheduler}
                                        onChange={(e) => onConfigChange('useScheduler', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">LR Scheduler</span>
                                </label>
                            </div>
                            {config.useScheduler && (
                                <div className="wizard-grid wizard-grid-scheduler">
                                    <div className="form-group">
                                        <label className="form-label">Target Learning Rate</label>
                                        <input
                                            type="number"
                                            step="0.0001"
                                            className="form-input"
                                            value={config.targetLR}
                                            onChange={(e) => onConfigChange('targetLR', parseFloat(e.target.value))}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Warmup Steps</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={config.warmupSteps}
                                            onChange={(e) => onConfigChange('warmupSteps', parseInt(e.target.value, 10))}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="training-wizard-footer">
                    {error && <span className="wizard-error">{error}</span>}
                    <div className="wizard-actions">
                        <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
                            Cancel
                        </button>
                        {currentPage > 0 && (
                            <button
                                className="btn btn-secondary"
                                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 0))}
                                disabled={isLoading}
                            >
                                <ChevronLeft size={16} />
                                Back
                            </button>
                        )}
                        {!isLastPage && (
                            <button
                                className="btn btn-primary"
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, steps.length - 1))}
                                disabled={isLoading}
                            >
                                Next
                                <ChevronRight size={16} />
                            </button>
                        )}
                        {isLastPage && (
                            <button
                                className="btn btn-primary"
                                onClick={onConfirm}
                                disabled={!canConfirm || isLoading}
                            >
                                <Play size={16} />
                                {isLoading ? 'Starting...' : 'Start Training'}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

interface ResumeTrainingWizardProps {
    isOpen: boolean;
    checkpoints: CheckpointInfo[];
    selectedCheckpoint: string;
    onCheckpointChange: (checkpoint: string) => void;
    additionalEpochs: number;
    onAdditionalEpochsChange: (epochs: number) => void;
    onClose: () => void;
    onConfirm: () => void;
    isLoading: boolean;
    error: string | null;
}

function ResumeTrainingWizard({
    isOpen,
    checkpoints,
    selectedCheckpoint,
    onCheckpointChange,
    additionalEpochs,
    onAdditionalEpochsChange,
    onClose,
    onConfirm,
    isLoading,
    error,
}: ResumeTrainingWizardProps) {
    const steps = ['Checkpoint', 'Schedule'];
    const [currentPage, setCurrentPage] = useState(0);

    useEffect(() => {
        if (isOpen) {
            setCurrentPage(0);
        }
    }, [isOpen]);

    const selectedInfo = useMemo(
        () => checkpoints.find((cp) => cp.name === selectedCheckpoint) || null,
        [checkpoints, selectedCheckpoint]
    );

    if (!isOpen) return null;

    const isLastPage = currentPage === steps.length - 1;
    const canConfirm = Boolean(selectedCheckpoint);

    return (
        <div className="training-modal-backdrop" onClick={onClose}>
            <div className="training-wizard-modal" onClick={(event) => event.stopPropagation()}>
                <div className="training-wizard-header">
                    <div>
                        <h3>Resume Training Wizard</h3>
                        <p>Checkpoint: <strong>{selectedCheckpoint || 'Select a checkpoint'}</strong></p>
                    </div>
                    <button className="training-wizard-close" onClick={onClose} aria-label="Close wizard">
                        <X size={18} />
                    </button>
                </div>
                <WizardSteps steps={steps} current={currentPage} />

                <div className="training-wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <RotateCcw size={16} />
                                <span>Select Checkpoint</span>
                            </div>
                            <div className="wizard-grid">
                                <div className="form-group span-2">
                                    <label className="form-label">Checkpoint</label>
                                    <select
                                        className="form-select"
                                        value={selectedCheckpoint}
                                        onChange={(e) => onCheckpointChange(e.target.value)}
                                    >
                                        <option value="">-- Select a checkpoint --</option>
                                        {checkpoints.map((cp) => (
                                            <option key={cp.name} value={cp.name}>
                                                {cp.name} - Epoch {cp.epochs} - Loss: {cp.loss.toFixed(4)}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                            {selectedInfo && (
                                <div className="wizard-summary">
                                    <div>
                                        <span>Epochs</span>
                                        <strong>{selectedInfo.epochs}</strong>
                                    </div>
                                    <div>
                                        <span>Loss</span>
                                        <strong>{selectedInfo.loss.toFixed(4)}</strong>
                                    </div>
                                    <div>
                                        <span>Val Loss</span>
                                        <strong>{selectedInfo.val_loss.toFixed(4)}</strong>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Activity size={16} />
                                <span>Training Schedule</span>
                            </div>
                            <div className="wizard-grid">
                                <div className="form-group">
                                    <label className="form-label">Additional Epochs</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={additionalEpochs}
                                        min={1}
                                        onChange={(e) => onAdditionalEpochsChange(parseInt(e.target.value, 10))}
                                    />
                                </div>
                            </div>
                            {selectedInfo && (
                                <div className="wizard-summary">
                                    <div>
                                        <span>Starting Epoch</span>
                                        <strong>{selectedInfo.epochs}</strong>
                                    </div>
                                    <div>
                                        <span>Total Epochs</span>
                                        <strong>{selectedInfo.epochs + additionalEpochs}</strong>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="training-wizard-footer">
                    {error && <span className="wizard-error">{error}</span>}
                    <div className="wizard-actions">
                        <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
                            Cancel
                        </button>
                        {currentPage > 0 && (
                            <button
                                className="btn btn-secondary"
                                onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 0))}
                                disabled={isLoading}
                            >
                                <ChevronLeft size={16} />
                                Back
                            </button>
                        )}
                        {!isLastPage && (
                            <button
                                className="btn btn-primary"
                                onClick={() => setCurrentPage((prev) => Math.min(prev + 1, steps.length - 1))}
                                disabled={isLoading}
                            >
                                Next
                                <ChevronRight size={16} />
                            </button>
                        )}
                        {isLastPage && (
                            <button
                                className="btn btn-primary"
                                onClick={onConfirm}
                                disabled={!canConfirm || isLoading}
                            >
                                <RotateCcw size={16} />
                                {isLoading ? 'Resuming...' : 'Resume Training'}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function TrainingPage() {
    const {
        state,
        updateConfig,
        setSelectedCheckpoint,
        setAdditionalEpochs,
        setDashboardState,
        setChartData,
        setAvailableMetrics,
        setEpochBoundaries,
    } = useTrainingPageState();

    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<DatasetInfo | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [newTrainingError, setNewTrainingError] = useState<string | null>(null);
    const [resumeTrainingError, setResumeTrainingError] = useState<string | null>(null);
    const [metadataModal, setMetadataModal] = useState<MetadataModalState | null>(null);
    const [isNewWizardOpen, setIsNewWizardOpen] = useState(false);
    const [isResumeWizardOpen, setIsResumeWizardOpen] = useState(false);
    const pollerRef = useRef<{ stop: () => void } | null>(null);

    const stopPolling = useCallback(() => {
        if (pollerRef.current) {
            pollerRef.current.stop();
            pollerRef.current = null;
        }
    }, []);

    const applyJobStatus = useCallback((status: JobStatusResponse | null) => {
        if (!status) return;

        const result = (status.result ?? {}) as Record<string, unknown>;
        const currentEpoch = typeof result.current_epoch === 'number' ? result.current_epoch : undefined;
        const totalEpochs = typeof result.total_epochs === 'number' ? result.total_epochs : undefined;
        const loss = typeof result.loss === 'number' ? result.loss : undefined;
        const valLoss = typeof result.val_loss === 'number' ? result.val_loss : undefined;
        const accuracy = typeof result.accuracy === 'number' ? result.accuracy : undefined;
        const valAccuracy = typeof result.val_accuracy === 'number' ? result.val_accuracy : undefined;
        const progressPercent = typeof result.progress_percent === 'number' ? result.progress_percent : status.progress;
        const elapsedSeconds = typeof result.elapsed_seconds === 'number' ? result.elapsed_seconds : undefined;

        setDashboardState((prev) => ({
            ...prev,
            isTraining: status.status === 'running' || status.status === 'pending',
            currentEpoch: currentEpoch ?? prev.currentEpoch,
            totalEpochs: totalEpochs ?? prev.totalEpochs,
            loss: loss ?? prev.loss,
            valLoss: valLoss ?? prev.valLoss,
            accuracy: accuracy ?? prev.accuracy,
            valAccuracy: valAccuracy ?? prev.valAccuracy,
            progressPercent: progressPercent ?? prev.progressPercent,
            elapsedSeconds: elapsedSeconds ?? prev.elapsedSeconds,
        }));

        if (Array.isArray(result.chart_data)) {
            setChartData(result.chart_data as ChartDataPoint[]);
        }
        if (Array.isArray(result.available_metrics)) {
            setAvailableMetrics(result.available_metrics as string[]);
        }
        if (Array.isArray(result.epoch_boundaries)) {
            setEpochBoundaries(result.epoch_boundaries as number[]);
        }
    }, [setAvailableMetrics, setChartData, setDashboardState, setEpochBoundaries]);

    const startPolling = useCallback((jobId: string) => {
        stopPolling();
        pollerRef.current = pollJobStatus(
            getTrainingJobStatus,
            jobId,
            (status) => applyJobStatus(status),
            (status) => applyJobStatus(status),
            (pollError) => {
                console.error('Training poll error:', pollError);
                stopPolling();
            },
            2000
        );
    }, [applyJobStatus, stopPolling]);

    const fetchDatasets = useCallback(async () => {
        const { result, error } = await getDatasetNames();
        if (error) {
            console.error('Failed to fetch datasets:', error);
            return;
        }
        if (result) {
            setDatasets(result.datasets);
            setSelectedDataset((prev) => {
                if (result.datasets.length === 0) return null;
                if (prev && result.datasets.some((ds) => ds.name === prev.name)) {
                    return result.datasets.find((ds) => ds.name === prev.name) || result.datasets[0];
                }
                return result.datasets[0];
            });
        }
    }, []);

    const fetchCheckpoints = useCallback(async () => {
        const { result, error: fetchError } = await getCheckpoints();
        if (fetchError) {
            console.error('Failed to fetch checkpoints:', fetchError);
            return;
        }
        if (result) {
            setCheckpoints(result.checkpoints);
            if (result.checkpoints.length > 0) {
                const exists = result.checkpoints.some((cp) => cp.name === state.selectedCheckpoint);
                if (!exists) {
                    setSelectedCheckpoint(result.checkpoints[0].name);
                }
            } else {
                setSelectedCheckpoint('');
            }
        }
    }, [setSelectedCheckpoint, state.selectedCheckpoint]);

    useEffect(() => {
        const checkTrainingStatus = async () => {
            const { result } = await getTrainingStatus();
            if (result) {
                setDashboardState((prev) => ({
                    ...prev,
                    isTraining: result.is_training,
                    currentEpoch: result.current_epoch,
                    totalEpochs: result.total_epochs,
                    loss: result.loss,
                    valLoss: result.val_loss,
                    accuracy: result.accuracy,
                    valAccuracy: result.val_accuracy,
                    progressPercent: result.progress_percent,
                    elapsedSeconds: result.elapsed_seconds,
                }));
            }
            if (result?.is_training && result.job_id) {
                startPolling(result.job_id);
            }
        };
        checkTrainingStatus();
        return () => {
            stopPolling();
        };
    }, [setDashboardState, startPolling, stopPolling]);

    useEffect(() => {
        fetchDatasets();
        fetchCheckpoints();
    }, [fetchCheckpoints, fetchDatasets]);

    const handleConfigChange = (key: keyof TrainingConfig, value: TrainingConfig[keyof TrainingConfig]) => {
        updateConfig(key, value);
    };

    const handleStartTraining = async () => {
        setIsLoading(true);
        setNewTrainingError(null);
        setChartData([]);
        setAvailableMetrics([]);
        setEpochBoundaries([]);

        const config: StartTrainingConfig = {
            epochs: state.config.epochs,
            batch_size: state.config.batchSize,
            training_seed: state.config.trainSeed,
            num_encoders: state.config.numEncoders,
            num_decoders: state.config.numDecoders,
            embedding_dims: state.config.embeddingDims,
            attention_heads: state.config.attnHeads,
            train_temp: state.config.trainTemp,
            freeze_img_encoder: state.config.freezeImgEncoder,
            use_img_augmentation: state.config.useImgAugment,
            shuffle_with_buffer: state.config.shuffleWithBuffer,
            shuffle_size: state.config.shuffleBufferSize,
            save_checkpoints: state.config.saveCheckpoints,
            use_tensorboard: state.config.runTensorboard,
            use_mixed_precision: state.config.mixedPrecision,
            use_device_GPU: true,
            device_ID: 0,
            plot_training_metrics: state.config.realTimePlot,
            use_scheduler: state.config.useScheduler,
            target_LR: state.config.targetLR,
            warmup_steps: state.config.warmupSteps,
        };

        const { result: startResult, error: trainError } = await startTraining(config);
        setIsLoading(false);

        if (trainError) {
            setNewTrainingError(trainError);
            console.error('Training failed:', trainError);
            return;
        }
        if (startResult) {
            setIsNewWizardOpen(false);
            startPolling(startResult.job_id);
        }
    };

    const handleResumeTraining = async () => {
        if (!state.selectedCheckpoint) return;

        setIsLoading(true);
        setResumeTrainingError(null);

        const { result: startResult, error: resumeError } = await resumeTraining(
            state.selectedCheckpoint,
            state.additionalEpochs
        );
        setIsLoading(false);

        if (resumeError) {
            setResumeTrainingError(resumeError);
            console.error('Resume training failed:', resumeError);
            return;
        }
        if (startResult) {
            setIsResumeWizardOpen(false);
            startPolling(startResult.job_id);
        }
    };

    const handleStopTraining = async () => {
        const { error: stopError } = await stopTraining();
        if (stopError) {
            console.error('Stop training failed:', stopError);
        }
    };

    const handleShowDatasetMetadata = async (dataset: DatasetInfo) => {
        setMetadataModal({
            title: 'Dataset Processing Metadata',
            subtitle: dataset.name,
        });

        const { result, error } = await getProcessingMetadata(dataset.name);
        if (error || !result) {
            setMetadataModal({
                title: 'Dataset Processing Metadata',
                subtitle: dataset.name,
                error: error || 'No metadata found',
            });
            return;
        }

        setMetadataModal({
            title: 'Dataset Processing Metadata',
            subtitle: result.dataset_name,
            sections: [
                {
                    title: 'Processing Parameters',
                    entries: buildEntries(result.metadata),
                },
            ],
        });
    };

    const handleShowCheckpointMetadata = async (checkpoint: CheckpointInfo) => {
        setMetadataModal({
            title: 'Checkpoint Metadata',
            subtitle: checkpoint.name,
        });

        const { result, error } = await getCheckpointMetadata(checkpoint.name);
        if (error || !result) {
            setMetadataModal({
                title: 'Checkpoint Metadata',
                subtitle: checkpoint.name,
                error: error || 'No metadata found',
            });
            return;
        }

        setMetadataModal({
            title: 'Checkpoint Metadata',
            subtitle: result.checkpoint,
            sections: [
                { title: 'Training Configuration', entries: buildEntries(result.configuration) },
                { title: 'Dataset Metadata', entries: buildEntries(result.metadata) },
                { title: 'Session Summary', entries: buildEntries(result.session) },
            ],
        });
    };

    const handleDeleteDataset = async (dataset: DatasetInfo) => {
        const confirmed = window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`);
        if (!confirmed) return;

        const { error } = await deleteDataset(dataset.name);
        if (error) {
            console.error('Failed to delete dataset:', error);
            return;
        }

        await fetchDatasets();
    };

    const handleDeleteCheckpoint = async (checkpoint: CheckpointInfo) => {
        const confirmed = window.confirm(`Delete checkpoint "${checkpoint.name}"? This cannot be undone.`);
        if (!confirmed) return;

        const { error } = await deleteCheckpoint(checkpoint.name);
        if (error) {
            console.error('Failed to delete checkpoint:', error);
            return;
        }

        await fetchCheckpoints();
    };

    const selectedCheckpointInfo = useMemo(
        () => checkpoints.find((cp) => cp.name === state.selectedCheckpoint) || null,
        [checkpoints, state.selectedCheckpoint]
    );

    return (
        <div className="training-container">
            <div className="header">
                <h1>XREPORT Transformer</h1>
                <p>Configure and monitor your training sessions</p>
            </div>

            <div className="training-panels">
                <div className="training-panel">
                    <div className="panel-left">
                        <div className="panel-header">
                            <div>
                                <h3>New Training Session</h3>
                                <p>Select a processed dataset to configure your next run.</p>
                            </div>
                            <button
                                className="panel-refresh"
                                onClick={fetchDatasets}
                                type="button"
                                aria-label="Refresh datasets"
                            >
                                <RefreshCw size={16} />
                            </button>
                        </div>

                        <div className="panel-list">
                            {datasets.length === 0 && (
                                <div className="panel-empty">No datasets available yet.</div>
                            )}
                            {datasets.map((dataset) => {
                                const isSelected = selectedDataset?.name === dataset.name;
                                return (
                                    <div
                                        key={dataset.name}
                                        className={`panel-row ${isSelected ? 'selected' : ''}`}
                                        onClick={() => setSelectedDataset(dataset)}
                                        role="button"
                                        tabIndex={0}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                setSelectedDataset(dataset);
                                            }
                                        }}
                                    >
                                        <div className="panel-row-main">
                                            <span className="panel-row-title">{dataset.name}</span>
                                            <span className="panel-row-count">{dataset.row_count.toLocaleString()} rows</span>
                                        </div>
                                        <div className="panel-row-actions" onClick={(event) => event.stopPropagation()}>
                                            <button
                                                type="button"
                                                className="icon-button"
                                                title="Show metadata"
                                                onClick={() => handleShowDatasetMetadata(dataset)}
                                            >
                                                <Info size={15} />
                                            </button>
                                            <button
                                                type="button"
                                                className="icon-button danger"
                                                title="Delete dataset"
                                                onClick={() => handleDeleteDataset(dataset)}
                                            >
                                                <Trash2 size={15} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>



                    <div className="panel-right">
                        <div className="panel-card">
                            <div className="panel-card-header">
                                <div className="panel-card-title-row">
                                    <Play size={18} />
                                    <h4>Initialize Training</h4>
                                </div>
                                <p>Launch the configuration wizard to set up your training run.</p>
                            </div>
                            <div className="panel-card-summary">
                                <span>Selected Dataset</span>
                                <strong>{selectedDataset?.name || 'None selected'}</strong>
                                <span>Samples</span>
                                <strong>{selectedDataset ? selectedDataset.row_count.toLocaleString() : 'N/A'}</strong>
                            </div>
                            <button
                                className="btn btn-primary"
                                type="button"
                                onClick={() => {
                                    setNewTrainingError(null);
                                    setIsNewWizardOpen(true);
                                }}
                                disabled={!selectedDataset}
                            >
                                <Play size={16} />
                                Configure Training
                            </button>
                        </div>
                    </div>
                </div>

                <div className="training-panel">
                    <div className="panel-left">
                        <div className="panel-header">
                            <div>
                                <h3>Resume Training</h3>
                                <p>Pick a checkpoint to continue training from a saved state.</p>
                            </div>
                            <button
                                className="panel-refresh"
                                onClick={fetchCheckpoints}
                                type="button"
                                aria-label="Refresh checkpoints"
                            >
                                <RefreshCw size={16} />
                            </button>
                        </div>
                        <div className="panel-list">
                            {checkpoints.length === 0 && (
                                <div className="panel-empty">No checkpoints available yet.</div>
                            )}
                            {checkpoints.map((checkpoint) => {
                                const isSelected = state.selectedCheckpoint === checkpoint.name;
                                return (
                                    <div
                                        key={checkpoint.name}
                                        className={`panel-row ${isSelected ? 'selected' : ''}`}
                                        onClick={() => setSelectedCheckpoint(checkpoint.name)}
                                        role="button"
                                        tabIndex={0}
                                        onKeyDown={(event) => {
                                            if (event.key === 'Enter' || event.key === ' ') {
                                                setSelectedCheckpoint(checkpoint.name);
                                            }
                                        }}
                                    >
                                        <div className="panel-row-main">
                                            <span className="panel-row-title">{checkpoint.name}</span>
                                            <span className="panel-row-meta">
                                                {checkpoint.epochs} epochs · loss {checkpoint.loss.toFixed(4)}
                                            </span>
                                        </div>
                                        <div className="panel-row-actions" onClick={(event) => event.stopPropagation()}>
                                            <button
                                                type="button"
                                                className="icon-button"
                                                title="Show metadata"
                                                onClick={() => handleShowCheckpointMetadata(checkpoint)}
                                            >
                                                <Info size={15} />
                                            </button>
                                            <button
                                                type="button"
                                                className="icon-button danger"
                                                title="Delete checkpoint"
                                                onClick={() => handleDeleteCheckpoint(checkpoint)}
                                            >
                                                <Trash2 size={15} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>



                    <div className="panel-right">
                        <div className="panel-card">
                            <div className="panel-card-header">
                                <div className="panel-card-title-row">
                                    <RotateCcw size={18} />
                                    <h4>Resume Session</h4>
                                </div>
                                <p>Continue training with a previous checkpoint.</p>
                            </div>
                            <div className="panel-card-summary">
                                <span>Selected Checkpoint</span>
                                <strong>{state.selectedCheckpoint || 'None selected'}</strong>
                                <span>Epochs</span>
                                <strong>{selectedCheckpointInfo ? selectedCheckpointInfo.epochs : 'N/A'}</strong>
                            </div>
                            <button
                                className="btn btn-primary"
                                type="button"
                                onClick={() => {
                                    setResumeTrainingError(null);
                                    setIsResumeWizardOpen(true);
                                }}
                                disabled={!state.selectedCheckpoint}
                            >
                                <RotateCcw size={16} />
                                Resume Training
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <TrainingDashboard
                onStopTraining={handleStopTraining}
                dashboardState={state.dashboardState}
            />

            <NewTrainingWizard
                isOpen={isNewWizardOpen}
                config={state.config}
                onConfigChange={handleConfigChange}
                onClose={() => setIsNewWizardOpen(false)}
                onConfirm={handleStartTraining}
                isLoading={isLoading}
                selectedDatasetLabel={selectedDataset?.name ?? ''}
                error={newTrainingError}
            />

            <ResumeTrainingWizard
                isOpen={isResumeWizardOpen}
                checkpoints={checkpoints}
                selectedCheckpoint={state.selectedCheckpoint}
                onCheckpointChange={setSelectedCheckpoint}
                additionalEpochs={state.additionalEpochs}
                onAdditionalEpochsChange={setAdditionalEpochs}
                onClose={() => setIsResumeWizardOpen(false)}
                onConfirm={handleResumeTraining}
                isLoading={isLoading}
                error={resumeTrainingError}
            />

            <MetadataModal state={metadataModal} onClose={() => setMetadataModal(null)} />
        </div>
    );
}
