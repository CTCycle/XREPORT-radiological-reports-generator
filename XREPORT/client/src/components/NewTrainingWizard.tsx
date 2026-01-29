import React, { useState, useEffect } from 'react';
import {
    Activity,
    ChevronLeft,
    ChevronRight,
    Cpu,
    Info,
    Play,
    Settings,
    X,
} from 'lucide-react';
import { TrainingConfig } from '../types';
import WizardSteps from './WizardSteps';
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
    const steps = ['Model', 'Dataset', 'Training', 'Summary'];
    const [currentPage, setCurrentPage] = useState(0);
    const [checkpointName, setCheckpointName] = useState('');

    useEffect(() => {
        if (isOpen) {
            setCurrentPage(0);
            setCheckpointName('');
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const canConfirm = Boolean(selectedDatasetLabel);
    const isLastPage = currentPage === steps.length - 1;

    const handleConfirm = () => {
        let finalName = checkpointName.trim();
        if (!finalName) {
            const timestamp = new Date().toISOString().replace(/[-:.]/g, '').slice(0, 15);
            finalName = `XREPORT_e${config.numEncoders}_d${config.numDecoders}_${timestamp}`;
        }
        onConfirm(finalName);
    };

    return (
        <div className="training-modal-backdrop">
            <div className="training-wizard-modal">
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
                            <div className="wizard-page-layout">
                                <div className="wizard-left-col">
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
                                <div className="wizard-right-col">
                                    <div className="form-group check-group-right">
                                        {/* Spacer to align checkbox with input box, matching left col label height */}
                                        <label className="form-label" style={{ opacity: 0, userSelect: 'none' }}>Encoders</label>
                                        <label className="form-checkbox">
                                            <input
                                                type="checkbox"
                                                checked={config.freezeImgEncoder}
                                                onChange={(e) => onConfigChange('freezeImgEncoder', e.target.checked)}
                                            />
                                            <div className="checkbox-visual" />
                                            <span className="checkbox-label">Freeze Encoder</span>
                                        </label>
                                        <span className="feature-explanation">
                                            Prevents encoder weights from being updated during training.
                                        </span>
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
                            <div className="wizard-aligned-grid">
                                <div className="form-group grid-col-left">
                                    <label className="form-label">Image Augmentation</label>
                                    <label className="form-checkbox">
                                        <input
                                            type="checkbox"
                                            checked={config.useImgAugment}
                                            onChange={(e) => onConfigChange('useImgAugment', e.target.checked)}
                                        />
                                        <div className="checkbox-visual" />
                                        <span className="checkbox-label">Enable Augmentation</span>
                                    </label>
                                </div>
                                <div className="shuffle-container grid-col-right form-group">
                                    <label className="form-label" style={{ userSelect: 'none' }}>Shuffle</label>
                                    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'stretch' }}>
                                        <label className="form-checkbox" style={{ marginBottom: 0 }}>
                                            <input
                                                type="checkbox"
                                                checked={config.shuffleWithBuffer}
                                                onChange={(e) => onConfigChange('shuffleWithBuffer', e.target.checked)}
                                            />
                                            <div className="checkbox-visual" />
                                            <span className="checkbox-label">Shuffle Buffered</span>
                                        </label>
                                        {config.shuffleWithBuffer && (
                                            <input
                                                type="number"
                                                step="10"
                                                className="form-input"
                                                placeholder="Size"
                                                style={{ flex: 1, minWidth: '80px' }}
                                                value={config.shuffleBufferSize}
                                                onChange={(e) => onConfigChange('shuffleBufferSize', parseInt(e.target.value, 10))}
                                            />
                                        )}
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
                            <div className="wizard-page-layout">
                                <div className="wizard-left-col">
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
                                <div className="wizard-right-col">
                                    <div className="form-group">
                                        <label className="form-label" style={{ opacity: 0, userSelect: 'none' }}>Save Checkpoints</label>
                                        <label className="form-checkbox">
                                            <input
                                                type="checkbox"
                                                checked={config.saveCheckpoints}
                                                onChange={(e) => onConfigChange('saveCheckpoints', e.target.checked)}
                                            />
                                            <div className="checkbox-visual" />
                                            <span className="checkbox-label">Save Checkpoints</span>
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <div className="wizard-separator" />

                            <div className="wizard-lr-card">
                                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-end' }}>
                                    <div style={{ paddingBottom: '1px' }}>
                                        <label className="form-checkbox">
                                            <input
                                                type="checkbox"
                                                checked={config.useScheduler}
                                                onChange={(e) => onConfigChange('useScheduler', e.target.checked)}
                                            />
                                            <div className="checkbox-visual" />
                                            <span className="checkbox-label">Use LR Scheduler</span>
                                        </label>
                                    </div>
                                    <div className={`lr-inputs ${!config.useScheduler ? 'disabled-section' : ''}`} style={{ display: 'flex', gap: '1.5rem', flex: 1 }}>
                                        <div className="form-group" style={{ flex: 1 }}>
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
                                        <div className="form-group" style={{ flex: 1 }}>
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
                        </div>
                    )}

                    {currentPage === 3 && (
                        <div className="wizard-page">
                            <div className="wizard-section-title">
                                <Info size={16} />
                                <span>Training Summary</span>
                            </div>
                            <div className="wizard-summary-content">
                                <div className="summary-scrollable">
                                    <div className="summary-section">
                                        <h5>Model Architecture</h5>
                                        <div className="summary-grid">
                                            <div className="summary-item"><label>Encoders:</label> <span>{config.numEncoders}</span></div>
                                            <div className="summary-item"><label>Decoders:</label> <span>{config.numDecoders}</span></div>
                                            <div className="summary-item"><label>Embedding:</label> <span>{config.embeddingDims}</span></div>
                                            <div className="summary-item"><label>Heads:</label> <span>{config.attnHeads}</span></div>
                                            <div className="summary-item"><label>Freeze Encoder:</label> <span>{config.freezeImgEncoder ? 'Yes' : 'No'}</span></div>
                                        </div>
                                    </div>
                                    <div className="summary-section">
                                        <h5>Dataset & Training</h5>
                                        <div className="summary-grid">
                                            <div className="summary-item"><label>Dataset:</label> <span>{selectedDatasetLabel}</span></div>

                                            <div className="summary-item"><label>Epochs:</label> <span>{config.epochs}</span></div>
                                            <div className="summary-item"><label>Batch Size:</label> <span>{config.batchSize}</span></div>
                                        </div>
                                    </div>
                                    <div className="summary-section">
                                        <h5>Preprocessing & Opts</h5>
                                        <div className="summary-grid">
                                            <div className="summary-item"><label>Augment:</label> <span>{config.useImgAugment ? 'Yes' : 'No'}</span></div>
                                            <div className="summary-item"><label>Shuffle:</label> <span>{config.shuffleWithBuffer ? `Yes (${config.shuffleBufferSize})` : 'No'}</span></div>
                                            <div className="summary-item"><label>Checkpoints:</label> <span>{config.saveCheckpoints ? 'Yes' : 'No'}</span></div>
                                            <div className="summary-item"><label>LR Scheduler:</label> <span>{config.useScheduler ? 'Yes' : 'No'}</span></div>
                                        </div>
                                    </div>
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
                            </div>
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
                                onClick={handleConfirm}
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
