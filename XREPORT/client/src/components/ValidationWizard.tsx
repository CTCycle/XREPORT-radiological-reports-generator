import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { DatasetInfo } from '../services/trainingService';
import './ValidationWizard.css';

export type ValidationMetric = 'pixels_distribution' | 'text_statistics' | 'image_statistics';

interface ValidationWizardProps {
    isOpen: boolean;
    row: DatasetInfo | null;
    initialSelected?: ValidationMetric[];
    onClose: () => void;
    onConfirm: (config: { metrics: ValidationMetric[]; row: DatasetInfo | null; sampleFraction: number }) => void;
}

const METRICS: Array<{ id: ValidationMetric; name: string; description: string }> = [
    {
        id: 'pixels_distribution',
        name: 'Pixel intensity histogram',
        description: 'Visualize intensity spread across the dataset to spot exposure issues.',
    },
    {
        id: 'text_statistics',
        name: 'Text statistics',
        description: 'Summaries of word counts, vocabulary size, and report length distribution.',
    },
    {
        id: 'image_statistics',
        name: 'Image statistics',
        description: 'Per-image dimensions, mean/std values, and noise indicators.',
    },
];

export default function ValidationWizard({
    isOpen,
    row,
    initialSelected,
    onClose,
    onConfirm,
}: ValidationWizardProps) {
    const [selectedMetrics, setSelectedMetrics] = useState<ValidationMetric[]>([]);
    const [validateFullDataset, setValidateFullDataset] = useState(true);
    const [validationFraction, setValidationFraction] = useState<string>('0.5');

    useEffect(() => {
        if (!isOpen) return;
        setSelectedMetrics(initialSelected ?? []);
        setValidateFullDataset(true);
        setValidationFraction('0.5');
    }, [isOpen, initialSelected, row]);

    const datasetLabel = useMemo(() => {
        if (!row) return 'Select a dataset';
        return row.name;
    }, [row]);

    const toggleMetric = (metric: ValidationMetric) => {
        setSelectedMetrics(prev => {
            if (prev.includes(metric)) {
                return prev.filter(item => item !== metric);
            }
            return [...prev, metric];
        });
    };

    const handleConfirm = () => {
        const fractionValue = validateFullDataset ? 1.0 : parseFloat(validationFraction);
        const fraction = Number.isFinite(fractionValue)
            ? Math.min(Math.max(fractionValue, 0.01), 1.0)
            : 1.0;
        onConfirm({
            metrics: selectedMetrics,
            row,
            sampleFraction: fraction,
        });
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="wizard-modal" onClick={(e) => e.stopPropagation()}>
                <div className="wizard-header">
                    <div>
                        <h3>Validation Wizard</h3>
                        <p className="wizard-subtitle">Dataset: <strong>{datasetLabel}</strong></p>
                    </div>
                    <button className="wizard-close" onClick={onClose} aria-label="Close validation wizard">
                        <X size={18} />
                    </button>
                </div>

                <div className="wizard-config-bar">
                    <div className="config-option" onClick={() => setValidateFullDataset(!validateFullDataset)}>
                        <div className={`toggle-switch ${validateFullDataset ? 'checked' : ''}`}>
                            <div className="toggle-slider"></div>
                        </div>
                        <span className="toggle-label">Validate full dataset</span>
                    </div>

                    <div className={`config-input-group ${validateFullDataset ? 'disabled' : ''}`}>
                        <span>Fraction:</span>
                        <input
                            type="number"
                            min="0.01"
                            max="1.0"
                            step="0.01"
                            value={validationFraction}
                            disabled={validateFullDataset}
                            onChange={(e) => setValidationFraction(e.target.value)}
                        />
                    </div>
                </div>

                <div className="wizard-body">
                    <div className="wizard-page">
                        <div className="wizard-step-title">Metrics selection</div>
                        <div className="wizard-metrics-grid">
                            {METRICS.map(metric => {
                                const isSelected = selectedMetrics.includes(metric.id);
                                return (
                                    <button
                                        key={metric.id}
                                        type="button"
                                        className={`wizard-metric ${isSelected ? 'selected' : ''}`}
                                        onClick={() => toggleMetric(metric.id)}
                                    >
                                        <div className="wizard-metric-title">{metric.name}</div>
                                        <div className="wizard-metric-desc">{metric.description}</div>
                                        <div className="wizard-metric-state">
                                            {isSelected ? 'Selected' : 'Select'}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="wizard-footer">
                    <div className="wizard-footer-actions">
                        <button className="btn btn-secondary" onClick={onClose}>
                            Cancel
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleConfirm}
                            disabled={selectedMetrics.length === 0}
                        >
                            Confirm
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
