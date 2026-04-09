import { useMemo } from 'react';
import {
    Activity,
    RotateCcw,
} from 'lucide-react';
import { CheckpointInfo } from '../services/trainingService';
import TrainingWizardModal from './shared/TrainingWizardModal';
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
    additionalEpochs,
    onAdditionalEpochsChange,
    onClose,
    onConfirm,
    isLoading,
    error,
}: ResumeTrainingWizardProps) {
    const selectedInfo = useMemo(
        () => checkpoints.find((cp) => cp.name === selectedCheckpoint) || null,
        [checkpoints, selectedCheckpoint]
    );

    if (!isOpen) return null;

    const canConfirm = Boolean(selectedCheckpoint);

    return (
        <TrainingWizardModal
            title="Resume Training"
            subtitle={<p>Configure the continuation for: <strong>{selectedCheckpoint}</strong></p>}
            onClose={onClose}
            body={(
                <>
                    <div className="wizard-page">
                        <div className="wizard-section-title">
                            <Activity size={16} />
                            <span>Training Schedule</span>
                        </div>
                        <div className="wizard-compact-grid">
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
                </>
            )}
            footer={(
                <>
                    {error && <span className="wizard-error">{error}</span>}
                    <div className="wizard-actions">
                        <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
                            Cancel
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={onConfirm}
                            disabled={!canConfirm || isLoading}
                        >
                            <RotateCcw size={16} />
                            {isLoading ? 'Resuming...' : 'Resume Training'}
                        </button>
                    </div>
                </>
            )}
        />
    );
}
