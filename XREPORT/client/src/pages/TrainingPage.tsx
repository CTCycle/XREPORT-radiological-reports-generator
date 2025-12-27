import { useState, useEffect } from 'react';
import {
    Play, Settings, Activity, Cpu, ChevronDown, RotateCcw
} from 'lucide-react';
import './TrainingPage.css';
import { useTrainingPageState } from '../AppStateContext';
import TrainingDashboard from '../components/TrainingDashboard';
import {
    startTraining,
    resumeTraining,
    stopTraining,
    getCheckpoints,
    CheckpointInfo,
    StartTrainingConfig,
} from '../services/trainingService';

export default function TrainingPage() {
    const {
        state,
        updateConfig,
        setNewSessionExpanded,
        setResumeSessionExpanded,
        setSelectedCheckpoint,
        setAdditionalEpochs
    } = useTrainingPageState();

    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [shouldConnectWs, setShouldConnectWs] = useState(false);

    // Fetch checkpoints on mount
    useEffect(() => {
        const fetchCheckpoints = async () => {
            const { result, error: fetchError } = await getCheckpoints();
            if (result) {
                setCheckpoints(result.checkpoints);
            } else if (fetchError) {
                console.error('Failed to fetch checkpoints:', fetchError);
            }
        };
        fetchCheckpoints();
    }, []);

    const handleConfigChange = (key: string, value: number | boolean) => {
        updateConfig(key as keyof typeof state.config, value);
    };

    const handleStartTraining = async () => {
        setIsLoading(true);
        setError(null);
        setShouldConnectWs(true); // Connect WebSocket when training starts

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

        const { error: trainError } = await startTraining(config);
        setIsLoading(false);

        if (trainError) {
            setError(trainError);
            console.error('Training failed:', trainError);
        }
    };

    const handleResumeTraining = async () => {
        if (!state.selectedCheckpoint) return;

        setIsLoading(true);
        setError(null);
        setShouldConnectWs(true); // Connect WebSocket when resume starts

        const { error: resumeError } = await resumeTraining(
            state.selectedCheckpoint,
            state.additionalEpochs
        );
        setIsLoading(false);

        if (resumeError) {
            setError(resumeError);
            console.error('Resume training failed:', resumeError);
        }
    };

    const handleStopTraining = async () => {
        const { error: stopError } = await stopTraining();
        if (stopError) {
            console.error('Stop training failed:', stopError);
        }
    };

    return (
        <div className="training-container">
            <div className="header">
                <h1>XREPORT Transformer</h1>
                <p>Configure and monitor your training sessions</p>
            </div>

            <div className="layout-rows">
                {/* Row 3: Training Session Accordions */}
                <div className="accordion-row">
                    {/* New Training Session Accordion */}
                    <div className="accordion">
                        <div
                            className="accordion-header"
                            onClick={() => setNewSessionExpanded(!state.newSessionExpanded)}
                        >
                            <div className="accordion-header-left">
                                <Play size={18} />
                                <span className="">New Training Session</span>
                            </div>
                            <ChevronDown
                                size={20}
                                className={`accordion-chevron ${state.newSessionExpanded ? 'expanded' : ''}`}
                            />
                        </div>
                        {state.newSessionExpanded && <div className="accordion-divider" />}
                        <div className={`accordion-content ${state.newSessionExpanded ? 'expanded' : ''}`}>
                            <div className="accordion-content-inner">
                                <div className="row-training-controls">
                                    {/* Left Column: Model Architecture */}
                                    <div className="section-column">
                                        <div className="section">
                                            <div className="section-title">
                                                <Cpu size={18} />
                                                <span>Model Architecture</span>
                                            </div>
                                            <div className="inputs-grid config-group-spacing">
                                                <div className="form-group">
                                                    <label className="form-label">Encoders</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={state.config.numEncoders}
                                                        onChange={(e) => handleConfigChange('numEncoders', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Decoders</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={state.config.numDecoders}
                                                        onChange={(e) => handleConfigChange('numDecoders', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Embedding Dims</label>
                                                    <input
                                                        type="number"
                                                        step="8"
                                                        className="form-input"
                                                        value={state.config.embeddingDims}
                                                        onChange={(e) => handleConfigChange('embeddingDims', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Attention Heads</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={state.config.attnHeads}
                                                        onChange={(e) => handleConfigChange('attnHeads', parseInt(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Temperature</label>
                                                    <input
                                                        type="number"
                                                        step="0.05"
                                                        className="form-input"
                                                        value={state.config.trainTemp}
                                                        onChange={(e) => handleConfigChange('trainTemp', parseFloat(e.target.value))}
                                                    />
                                                </div>
                                            </div>
                                            <div className="toggles-grid">
                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={state.config.freezeImgEncoder}
                                                            onChange={(e) => handleConfigChange('freezeImgEncoder', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Freeze Encoder</span>
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Right Column: Dataset Config */}
                                    <div className="section-column">
                                        <div className="section">
                                            <div className="section-title">
                                                <Settings size={18} />
                                                <span>Dataset Config</span>
                                            </div>
                                            <div className="toggles-grid config-group-spacing">
                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={state.config.useImgAugment}
                                                            onChange={(e) => handleConfigChange('useImgAugment', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Image Augmentation</span>
                                                    </label>
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-checkbox">
                                                        <input
                                                            type="checkbox"
                                                            checked={state.config.shuffleWithBuffer}
                                                            onChange={(e) => handleConfigChange('shuffleWithBuffer', e.target.checked)}
                                                        />
                                                        <div className="checkbox-visual" />
                                                        <span className="checkbox-label">Shuffle Buffered</span>
                                                    </label>
                                                </div>
                                            </div>
                                            {state.config.shuffleWithBuffer && (
                                                <div className="inputs-grid">
                                                    <div className="form-group">
                                                        <label className="form-label">Buffer Size</label>
                                                        <input
                                                            type="number"
                                                            step="10"
                                                            className="form-input"
                                                            value={state.config.shuffleBufferSize}
                                                            onChange={(e) => handleConfigChange('shuffleBufferSize', parseInt(e.target.value))}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Full Width Row: Training Parameters */}
                                <div style={{ marginTop: '1rem' }}>
                                    <div className="section">
                                        <div className="section-title">
                                            <Activity size={18} />
                                            <span>Training Parameters</span>
                                        </div>
                                        <div className="inputs-grid config-group-spacing">
                                            <div className="form-group">
                                                <label className="form-label">Epochs</label>
                                                <input
                                                    type="number"
                                                    className="form-input"
                                                    value={state.config.epochs}
                                                    onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                                                />
                                            </div>
                                            <div className="form-group">
                                                <label className="form-label">Batch Size</label>
                                                <input
                                                    type="number"
                                                    className="form-input"
                                                    value={state.config.batchSize}
                                                    onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                                                />
                                            </div>
                                            <div className="form-group">
                                                <label className="form-label">Training Seed</label>
                                                <input
                                                    type="number"
                                                    className="form-input"
                                                    value={state.config.trainSeed}
                                                    onChange={(e) => handleConfigChange('trainSeed', parseInt(e.target.value))}
                                                />
                                            </div>
                                        </div>

                                        <div className="toggles-grid">
                                            <div className="form-group">
                                                <label className="form-checkbox">
                                                    <input
                                                        type="checkbox"
                                                        checked={state.config.saveCheckpoints}
                                                        onChange={(e) => handleConfigChange('saveCheckpoints', e.target.checked)}
                                                    />
                                                    <div className="checkbox-visual" />
                                                    <span className="checkbox-label">Save Checkpoints</span>
                                                </label>
                                            </div>

                                            <div className="form-group">
                                                <label className="form-checkbox">
                                                    <input
                                                        type="checkbox"
                                                        checked={state.config.runTensorboard}
                                                        onChange={(e) => handleConfigChange('runTensorboard', e.target.checked)}
                                                    />
                                                    <div className="checkbox-visual" />
                                                    <span className="checkbox-label">Tensorboard</span>
                                                </label>
                                            </div>

                                            <div className="form-group">
                                                <label className="form-checkbox">
                                                    <input
                                                        type="checkbox"
                                                        checked={state.config.mixedPrecision}
                                                        onChange={(e) => handleConfigChange('mixedPrecision', e.target.checked)}
                                                    />
                                                    <div className="checkbox-visual" />
                                                    <span className="checkbox-label">Mixed Precision</span>
                                                </label>
                                            </div>

                                            <div className="form-group">
                                                <label className="form-checkbox">
                                                    <input
                                                        type="checkbox"
                                                        checked={state.config.useScheduler}
                                                        onChange={(e) => handleConfigChange('useScheduler', e.target.checked)}
                                                    />
                                                    <div className="checkbox-visual" />
                                                    <span className="checkbox-label">LR Scheduler</span>
                                                </label>
                                            </div>
                                        </div>
                                        {/* Scheduler configuration row */}
                                        {state.config.useScheduler && (
                                            <div className="scheduler-config-row">
                                                <div className="form-group">
                                                    <label className="form-label">Target Learning Rate</label>
                                                    <input
                                                        type="number"
                                                        step="0.0001"
                                                        className="form-input"
                                                        value={state.config.targetLR}
                                                        onChange={(e) => handleConfigChange('targetLR', parseFloat(e.target.value))}
                                                    />
                                                </div>
                                                <div className="form-group">
                                                    <label className="form-label">Warmup Steps</label>
                                                    <input
                                                        type="number"
                                                        className="form-input"
                                                        value={state.config.warmupSteps}
                                                        onChange={(e) => handleConfigChange('warmupSteps', parseInt(e.target.value))}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Actions Bar inside accordion */}
                            <div className="actions-bar" style={{ marginTop: '1rem' }}>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleStartTraining}
                                    disabled={isLoading}
                                >
                                    <Play size={16} />
                                    {isLoading ? 'Starting...' : 'Start Training'}
                                </button>
                                {error && (
                                    <span style={{ color: '#ef4444', marginLeft: '1rem', fontSize: '0.85rem' }}>
                                        {error}
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Resume Training Session Accordion */}
                <div className="accordion">
                    <div
                        className="accordion-header"
                        onClick={() => setResumeSessionExpanded(!state.resumeSessionExpanded)}
                    >
                        <div className="accordion-header-left">
                            <RotateCcw size={18} />
                            <span>Resume Training Session</span>
                        </div>
                        <ChevronDown
                            size={20}
                            className={`accordion-chevron ${state.resumeSessionExpanded ? 'expanded' : ''}`}
                        />
                    </div>
                    {state.resumeSessionExpanded && <div className="accordion-divider" />}
                    <div className={`accordion-content ${state.resumeSessionExpanded ? 'expanded' : ''}`}>
                        <div className="accordion-content-inner">
                            <div className="resume-session-grid">
                                <div className="form-group">
                                    <label className="form-label">Select Checkpoint</label>
                                    <select
                                        className="form-select"
                                        value={state.selectedCheckpoint}
                                        onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                    >
                                        <option value="">-- Select a checkpoint --</option>
                                        {checkpoints.map((cp) => (
                                            <option key={cp.name} value={cp.name}>
                                                {cp.name} - Epoch {cp.epochs} - Loss: {cp.loss.toFixed(4)}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Additional Epochs</label>
                                    <input
                                        type="number"
                                        className="form-input"
                                        value={state.additionalEpochs}
                                        onChange={(e) => setAdditionalEpochs(parseInt(e.target.value))}
                                        min={1}
                                    />
                                </div>
                                <button
                                    className="btn btn-primary"
                                    disabled={!state.selectedCheckpoint || isLoading}
                                    onClick={handleResumeTraining}
                                >
                                    <RotateCcw size={16} />
                                    {isLoading ? 'Resuming...' : 'Resume Training'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Training Dashboard */}
            <TrainingDashboard onStopTraining={handleStopTraining} shouldConnect={shouldConnectWs} />
        </div>
    );
}
