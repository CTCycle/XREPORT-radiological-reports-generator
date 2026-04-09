import { useCallback, useState } from 'react';
import {
    Activity,
    ChevronLeft,
    ChevronRight,
    Cpu,
    Info,
    Monitor,
    Play,
    Settings,
} from 'lucide-react';
import { TrainingConfig } from '../types';
import WizardSteps from './WizardSteps';
import FormCheckbox from './shared/FormCheckbox';
import TrainingWizardModal from './shared/TrainingWizardModal';
import { useResetOnOpen } from '../hooks/useResetOnOpen';
import '../pages/TrainingPage.css';

interface NewTrainingWizardProps {
    isOpen: boolean;
    config: TrainingConfig;
    onConfigChange: (key: keyof TrainingConfig, value: TrainingConfig[keyof TrainingConfig]) => void;
    onClose: () => void;
    onConfirm: (checkpointName: string) => void;
    isLoading: boolean;
    selectedDatasetLabel: string;
    error: string | null;
}

export default function NewTrainingWizard({
    isOpen,
    config,
    onConfigChange,
    onClose,
    onConfirm,
    isLoading,
    selectedDatasetLabel,
    error,
}: NewTrainingWizardProps) {
    const steps = ['Model', 'Dataset', 'Training', 'Device', 'Summary'];
    const jitBackendOptions = ['inductor', 'eager', 'aot_eager', 'nvprims_nvfuser'];
    const selectedGpuId = Math.max(0, config.gpuId);
    const baseGpuDeviceOptions = [0, 1, 2, 3];
    const gpuDeviceOptions = baseGpuDeviceOptions.includes(selectedGpuId)
        ? baseGpuDeviceOptions
        : [...baseGpuDeviceOptions, selectedGpuId].sort((left, right) => left - right);
    const [currentPage, setCurrentPage] = useState(0);
    const [checkpointName, setCheckpointName] = useState('');

    const resetWizard = useCallback(() => {
        setCurrentPage(0);
        setCheckpointName('');
    }, []);

    useResetOnOpen(isOpen, resetWizard);

    if (!isOpen) return null;

    const canConfirm = Boolean(selectedDatasetLabel);
    const isLastPage = currentPage === steps.length - 1;

    const parseIntOrFallback = (rawValue: string, fallback: number, min: number) => {
        const parsedValue = Number.parseInt(rawValue, 10);
        if (Number.isNaN(parsedValue)) {
            return fallback;
        }
        return Math.max(min, parsedValue);
    };

    const handleConfirm = () => {
        let finalName = checkpointName.trim();
        if (!finalName) {
            const timestamp = new Date().toISOString().replace(/[-:.]/g, '').slice(0, 15);
            finalName = `XREPORT_e${config.numEncoders}_d${config.numDecoders}_${timestamp}`;
        }
        onConfirm(finalName);
    };

    return (
        <TrainingWizardModal
            title="New Training Wizard"
            subtitle={<p>Dataset: <strong>{selectedDatasetLabel || 'No dataset selected'}</strong></p>}
            onClose={onClose}
            steps={<WizardSteps steps={steps} current={currentPage} />}
            body={(
                <>
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Cpu size={16} />
                                <span>Model Architecture</span>
                            </div>
                            <div className="wizard-2col-panel">
                                <div className="wizard-col">
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
                                </div>
                                <div className="wizard-col">
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
                                    <div className="form-group">
                                        <label className="form-label form-label-placeholder">Encoders</label>
                                        <FormCheckbox
                                            checked={config.freezeImgEncoder}
                                            label="Freeze Encoder"
                                            onChange={(checked) => onConfigChange('freezeImgEncoder', checked)}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 1 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Settings size={16} />
                                <span>Dataset Configuration</span>
                            </div>
                            <div className="wizard-2col-panel">
                                <div className="wizard-col">
                                    <div className="form-group">
                                        <label className="form-label">Image Augmentation</label>
                                        <FormCheckbox
                                            checked={config.useImgAugment}
                                            label="Enable Augmentation"
                                            onChange={(checked) => onConfigChange('useImgAugment', checked)}
                                        />
                                    </div>
                                </div>
                                <div className="wizard-col">
                                    <div className="form-group">
                                        <label className="form-label form-label-transparent">Shuffle</label>
                                        <div className="wizard-inline-row">
                                            <FormCheckbox
                                                checked={config.shuffleWithBuffer}
                                                label="Shuffle"
                                                className="form-checkbox-no-margin"
                                                onChange={(checked) => onConfigChange('shuffleWithBuffer', checked)}
                                            />
                                            {config.shuffleWithBuffer && (
                                                <input
                                                    type="number"
                                                    step="10"
                                                    className="form-input shuffle-buffer-input"
                                                    placeholder="Buffer"
                                                    value={config.shuffleBufferSize}
                                                    onChange={(e) => onConfigChange('shuffleBufferSize', parseInt(e.target.value, 10))}
                                                />
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 2 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Activity size={16} />
                                <span>Training Parameters</span>
                            </div>
                            <div className="wizard-2col-panel">
                                <div className="wizard-col">
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
                                </div>
                                <div className="wizard-col">
                                    <div className="form-group">
                                        <label className="form-label form-label-placeholder">Save Checkpoints</label>
                                        <FormCheckbox
                                            checked={config.saveCheckpoints}
                                            label="Save Checkpoints"
                                            onChange={(checked) => onConfigChange('saveCheckpoints', checked)}
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="wizard-separator" />

                            <div className="wizard-2col-panel">
                                <div className="wizard-col">
                                    <div className="form-group">
                                        <FormCheckbox
                                            checked={config.useScheduler}
                                            label="Use LR Scheduler"
                                            onChange={(checked) => onConfigChange('useScheduler', checked)}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="form-label">Target Learning Rate</label>
                                        <input
                                            type="number"
                                            step="0.0001"
                                            className="form-input"
                                            value={config.targetLR}
                                            onChange={(e) => onConfigChange('targetLR', parseFloat(e.target.value))}
                                            disabled={!config.useScheduler}
                                        />
                                    </div>
                                </div>
                                <div className="wizard-col">
                                    <div className="form-group">
                                        <label className="form-label">Warmup Steps</label>
                                        <input
                                            type="number"
                                            className="form-input"
                                            value={config.warmupSteps}
                                            onChange={(e) => onConfigChange('warmupSteps', parseInt(e.target.value, 10))}
                                            disabled={!config.useScheduler}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 3 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Monitor size={16} />
                                <span>Device Configuration</span>
                            </div>
                            <div className="wizard-device-panel">
                                <div className="wizard-device-panel-header">
                                    <Monitor size={14} />
                                    <span>Device Controls</span>
                                </div>
                                <p className="wizard-device-panel-description">
                                    Configure data loading and runtime acceleration options.
                                </p>
                                <div className="wizard-2col-panel wizard-device-layout">
                                    <div className="wizard-col">
                                        <div className="form-group">
                                            <FormCheckbox
                                                checked={config.pinMemory}
                                                label="Pin Memory"
                                                onChange={(checked) => onConfigChange('pinMemory', checked)}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <FormCheckbox
                                                checked={config.useMixedPrecision}
                                                label="Mixed Precision"
                                                onChange={(checked) => onConfigChange('useMixedPrecision', checked)}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Dataloader Workers</label>
                                            <input
                                                type="number"
                                                min={0}
                                                className="form-input"
                                                value={config.dataloaderWorkers}
                                                onChange={(e) => onConfigChange(
                                                    'dataloaderWorkers',
                                                    parseIntOrFallback(e.target.value, config.dataloaderWorkers, 0),
                                                )}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <label className="form-label">Prefetch Factor</label>
                                            <input
                                                type="number"
                                                min={1}
                                                className="form-input"
                                                value={config.prefetchFactor}
                                                onChange={(e) => onConfigChange(
                                                    'prefetchFactor',
                                                    parseIntOrFallback(e.target.value, config.prefetchFactor, 1),
                                                )}
                                                disabled={config.dataloaderWorkers === 0}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <FormCheckbox
                                                checked={config.persistentWorkers}
                                                label="Persistent Workers"
                                                disabled={config.dataloaderWorkers === 0}
                                                onChange={(checked) => onConfigChange('persistentWorkers', checked)}
                                            />
                                        </div>
                                        <div className="form-group">
                                            <FormCheckbox
                                                checked={config.realTimePlot}
                                                label="Plot Training Metrics"
                                                onChange={(checked) => onConfigChange('realTimePlot', checked)}
                                            />
                                        </div>
                                    </div>
                                    <div className="wizard-col">
                                        <div className="wizard-device-card">
                                            <h5>Torch Compile</h5>
                                            <p>Enable torch.compile to optimize runtime graph execution.</p>
                                            <div className="wizard-device-card-controls">
                                                <FormCheckbox
                                                    checked={config.jitCompile}
                                                    label="Torch Compile"
                                                    className="form-checkbox-no-margin"
                                                    onChange={(checked) => onConfigChange('jitCompile', checked)}
                                                />
                                                <div className="form-group">
                                                    <label className="form-label">Backend</label>
                                                    <select
                                                        className="form-select"
                                                        value={config.jitBackend}
                                                        onChange={(e) => onConfigChange('jitBackend', e.target.value)}
                                                        disabled={!config.jitCompile}
                                                    >
                                                        {jitBackendOptions.map((backend) => (
                                                            <option key={backend} value={backend}>
                                                                {backend}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="wizard-device-card">
                                            <h5>Enable GPU</h5>
                                            <p>Run training on CUDA and choose the target GPU device index.</p>
                                            <div className="wizard-device-card-controls">
                                                <FormCheckbox
                                                    checked={config.useGpu}
                                                    label="Enable GPU"
                                                    className="form-checkbox-no-margin"
                                                    onChange={(checked) => onConfigChange('useGpu', checked)}
                                                />
                                                <div className="form-group">
                                                    <label className="form-label">Device</label>
                                                    <select
                                                        className="form-select"
                                                        value={selectedGpuId}
                                                        onChange={(e) => onConfigChange(
                                                            'gpuId',
                                                            parseIntOrFallback(e.target.value, selectedGpuId, 0),
                                                        )}
                                                        disabled={!config.useGpu}
                                                    >
                                                        {gpuDeviceOptions.map((gpuOption) => (
                                                            <option key={gpuOption} value={gpuOption}>
                                                                {gpuOption}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentPage === 4 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Info size={16} />
                                <span>Training Summary</span>
                            </div>

                            <div className="form-group checkpoint-name-group">
                                <label className="form-label">Training Checkpoint Name</label>
                                <input
                                    type="text"
                                    className="form-input highlight-input"
                                    placeholder="e.g. My_Experiment_v1 (Optional)"
                                    value={checkpointName}
                                    onChange={(e) => setCheckpointName(e.target.value)}
                                />
                                <span className="form-help">Unique name for this training run. If empty, a default name will be generated.</span>
                            </div>

                            <div className="wizard-summary-content">
                                <div className="wizard-summary">
                                    <div className="summary-section">
                                        <h5>Model</h5>
                                        <div className="summary-item"><label>Encoders:</label> <span>{config.numEncoders}</span></div>
                                        <div className="summary-item"><label>Decoders:</label> <span>{config.numDecoders}</span></div>
                                        <div className="summary-item"><label>Embedding:</label> <span>{config.embeddingDims}</span></div>
                                        <div className="summary-item"><label>Heads:</label> <span>{config.attnHeads}</span></div>
                                    </div>
                                    <div className="summary-section">
                                        <h5>Training</h5>
                                        <div className="summary-item"><label>Epochs:</label> <span>{config.epochs}</span></div>
                                        <div className="summary-item"><label>Batch Size:</label> <span>{config.batchSize}</span></div>
                                        <div className="summary-item"><label>LR Scheduler:</label> <span>{config.useScheduler ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Target LR:</label> <span>{config.targetLR}</span></div>
                                    </div>
                                    <div className="summary-section">
                                        <h5>Options</h5>
                                        <div className="summary-item"><label>Augment:</label> <span>{config.useImgAugment ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Shuffle:</label> <span>{config.shuffleWithBuffer ? `Yes (${config.shuffleBufferSize})` : 'No'}</span></div>
                                        <div className="summary-item"><label>Checkpoints:</label> <span>{config.saveCheckpoints ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Freeze:</label> <span>{config.freezeImgEncoder ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>GPU:</label> <span>{config.useGpu ? `cuda:${config.gpuId}` : 'CPU'}</span></div>
                                        <div className="summary-item"><label>Workers:</label> <span>{config.dataloaderWorkers}</span></div>
                                        <div className="summary-item"><label>Prefetch:</label> <span>{config.dataloaderWorkers > 0 ? config.prefetchFactor : 'N/A'}</span></div>
                                        <div className="summary-item"><label>Pin Memory:</label> <span>{config.pinMemory ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Persistent:</label> <span>{config.persistentWorkers ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Mixed Precision:</label> <span>{config.useMixedPrecision ? 'Yes' : 'No'}</span></div>
                                        <div className="summary-item"><label>Torch Compile:</label> <span>{config.jitCompile ? `Yes (${config.jitBackend})` : 'No'}</span></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}
            footer={(
                <>
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
                                onClick={handleConfirm}
                                disabled={!canConfirm || isLoading}
                            >
                                <Play size={16} />
                                {isLoading ? 'Starting...' : 'Start Training'}
                            </button>
                        )}
                    </div>
                </>
            )}
        />
    );
}
