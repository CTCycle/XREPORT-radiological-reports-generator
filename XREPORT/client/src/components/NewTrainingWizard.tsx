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

    const isLastPage = currentPage === steps.length - 1;
    const canConfirm = Boolean(selectedDatasetLabel) && checkpointName.trim().length > 0;

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
                            <div className="wizard-grid">
                                <div className="form-group">
                                    <label className="form-label">Validation Split</label>
                                    <input
                                        type="number"
                                        step="0.01"
                                        min="0.05"
                                        max="0.5"
                                        className="form-input"
                                        value={config.validationSize}
                                        onChange={(e) => onConfigChange('validationSize', parseFloat(e.target.value))}
                                    />
                                    <span className="form-help">0.05 - 0.5 (default: 0.2)</span>
                                </div>
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
                                <div className="wizard-grid wizard-grid-single">
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
                                        <h5>Data & Schedule</h5>
                                        <div className="summary-grid">
                                            <div className="summary-item"><label>Dataset:</label> <span>{selectedDatasetLabel}</span></div>
                                            <div className="summary-item"><label>Validation Split:</label> <span>{config.validationSize}</span></div>
                                            <div className="summary-item"><label>Epochs:</label> <span>{config.epochs}</span></div>
                                            <div className="summary-item"><label>Batch Size:</label> <span>{config.batchSize}</span></div>
                                            <div className="summary-item"><label>Scheduler:</label> <span>{config.useScheduler ? 'Yes' : 'No'}</span></div>
                                        </div>
                                    </div>
                                </div>

                                <div className="form-group checkpoint-name-group">
                                    <label className="form-label">Training Checkpoint Name</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        placeholder="e.g. My_Experiment_v1"
                                        value={checkpointName}
                                        onChange={(e) => setCheckpointName(e.target.value)}
                                        required
                                    />
                                    <span className="form-help">Unique name for this training run. Required.</span>
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
                                onClick={() => onConfirm(checkpointName)}
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
