import React, { useState, useEffect, useMemo } from 'react';
import {
    Activity,
    ChevronLeft,
    ChevronRight,
    RotateCcw,
    X,
} from 'lucide-react';
import { CheckpointInfo } from '../services/trainingService';
import WizardSteps from './WizardSteps';
import '../pages/TrainingPage.css';

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

export default function ResumeTrainingWizard({
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
        <div className="training-modal-backdrop">
            <div className="training-wizard-modal">
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
