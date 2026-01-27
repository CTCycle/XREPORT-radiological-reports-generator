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
    onConfirm: (config: { metrics: ValidationMetric[]; row: DatasetInfo | null }) => void;
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

const STEPS = [
    {
        id: 'metrics',
        title: 'Metrics selection',
    },
];

export default function ValidationWizard({
    isOpen,
    row,
    initialSelected,
    onClose,
    onConfirm,
}: ValidationWizardProps) {
    const [currentPage, setCurrentPage] = useState(0);
    const [selectedMetrics, setSelectedMetrics] = useState<ValidationMetric[]>([]);

    useEffect(() => {
        if (!isOpen) return;
        setCurrentPage(0);
        setSelectedMetrics(initialSelected ?? []);
    }, [isOpen, initialSelected, row]);

    const isLastPage = currentPage >= STEPS.length - 1;
    const currentStep = STEPS[currentPage];

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

    const handlePrevious = () => {
        setCurrentPage(prev => Math.max(0, prev - 1));
    };

    const handleNext = () => {
        if (!isLastPage) {
            setCurrentPage(prev => Math.min(STEPS.length - 1, prev + 1));
            return;
        }
        onConfirm({ metrics: selectedMetrics, row });
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

                <div className="wizard-page-indicator">
                    {STEPS.map((step, index) => (
                        <span
                            key={step.id}
                            className={`wizard-dot ${currentPage === index ? 'active' : ''}`}
                        >
                            {index + 1}
                        </span>
                    ))}
                </div>

                <div className="wizard-body">
                    {currentPage === 0 && (
                        <div className="wizard-page">
                            <div className="wizard-card">
                                <div className="wizard-step-title">{currentStep.title}</div>
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
                    )}
                </div>

                <div className="wizard-footer">
                    <button
                        className="btn btn-secondary"
                        onClick={handlePrevious}
                        disabled={currentPage === 0}
                    >
                        Previous
                    </button>
                    <div className="wizard-footer-actions">
                        <button className="btn btn-secondary" onClick={onClose}>
                            Cancel
                        </button>
                        <button className="btn btn-primary" onClick={handleNext}>
                            {isLastPage ? 'Confirm' : 'Next'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
