import React, { useState, useMemo } from 'react';
import {
    X,
    ArrowLeft,
    ArrowRight,
    Check,
    BarChart3,
    Target,
    Activity,
    SlidersHorizontal
} from 'lucide-react';
import './EvaluationWizard.css';

// --- Types & Constants ---

export interface EvaluationWizardProps {
    isOpen: boolean;
    onClose: () => void;
    checkpointName: string;
}

interface MetricCatalogItem {
    id: string;
    title: string;
    description: string;
    icon: React.ReactNode;
    defaultConfig?: Record<string, any>;
}

const METRICS_CATALOG: MetricCatalogItem[] = [
    {
        id: 'evaluation_report',
        title: 'Evaluation Report',
        description: 'Standard validation including Loss and Accuracy metrics on the validation dataset.',
        icon: <Target size={24} />,
        defaultConfig: {
            dataFraction: 1.0, // 100% of validation data
        }
    },
    {
        id: 'bleu_score',
        title: 'BLEU Score',
        description: 'Bilingual Evaluation Understudy score to measure text generation quality.',
        icon: <BarChart3 size={24} />,
        defaultConfig: {
            dataFraction: 0.1, // Default to smaller subset for expensive BLEU
        }
    }
];

// --- Sub-components ---


// Step 2: Configuration
const MetricConfigStep: React.FC<{
    metricId: string;
    config: Record<string, any>;
    onUpdateConfig: (key: string, value: any) => void;
}> = ({ metricId, config, onUpdateConfig }) => {
    const metric = METRICS_CATALOG.find(m => m.id === metricId);
    if (!metric) return null;

    return (
        <div className="config-container">
            <div className="config-header">
                <div className="config-header-icon">
                    {metric.icon}
                </div>
                <h3>Configure {metric.title}</h3>
                <p>{metric.description}</p>
            </div>

            <div className="config-form">
                {/* Visual Separator */}
                <div className="wizard-separator"></div>

                {/* Common: Data Fraction */}
                <div className="form-group">
                    <label>
                        Data Fraction
                        <span className="fraction-value">
                            {Math.round((config.dataFraction || 0) * 100)}%
                        </span>
                    </label>
                    <div className="range-control">
                        <span style={{ fontSize: '0.8rem', color: '#888' }}>0%</span>
                        <input
                            type="range"
                            min="0.01"
                            max="1.0"
                            step="0.01"
                            value={config.dataFraction || 0.1}
                            onChange={(e) => onUpdateConfig('dataFraction', parseFloat(e.target.value))}
                        />
                        <span style={{ fontSize: '0.8rem', color: '#888' }}>100%</span>
                    </div>
                    <p className="param-description">
                        Percentage of the validation dataset to use for this evaluation.
                        {metric.id === 'bleu_score' && " Lower values recommended for faster BLEU calculation."}
                    </p>
                </div>


            </div>
        </div>
    );
};

// Step 3: Summary
const SummaryStep: React.FC<{
    checkpointName: string;
    configs: Record<string, any>;
    metricsOrder: string[]; // Order of metrics to display
}> = ({ checkpointName, configs, metricsOrder }) => {
    return (
        <div className="summary-container">
            <h3 className="wizard-step-title-text">Review & Confirm</h3>

            <div className="summary-card">
                <div className="summary-row">
                    <span className="summary-label">Checkpoint</span>
                    <span className="summary-checkpoint-value">{checkpointName}</span>
                </div>
            </div>

            <h4 className="config-section-title">Selected Metrics Configuration</h4>

            {metricsOrder.map(metricId => {
                const metric = METRICS_CATALOG.find(m => m.id === metricId);
                const config = configs[metricId];
                if (!metric || !config) return null;

                return (
                    <div key={metricId} className="summary-card compact">
                        <div className="summary-metric-header">
                            {metric.icon}
                            <span>{metric.title}</span>
                        </div>
                        <div className="config-summary-list">
                            <div className="config-item">
                                <span className="item-label">Data Fraction</span>
                                <span>{Math.round(config.dataFraction * 100)}%</span>
                            </div>

                        </div>
                    </div>
                );
            })}
        </div>
    );
};


// --- Main Wizard Component ---

export default function EvaluationWizard({ isOpen, onClose, checkpointName }: EvaluationWizardProps) {
    // State
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
    const [metricConfigs, setMetricConfigs] = useState<Record<string, any>>({});

    // Reset state when opening/closing (optional, but good for wizards)
    // For now we just rely on parent unmounting or key changes if needed, 
    // but typically a proper 'isOpen' effect handles this. 
    // We'll skip complex reset logic for simplicity unless requested.

    // Derived Logic
    // Step 0: Selection
    // Step 1..N: Config for each selected metric
    // Step N+1: Summary

    // Sort selected metrics based on CATALOG order for deterministic flow
    const orderedSelectedMetrics = useMemo(() => {
        return METRICS_CATALOG
            .filter(m => selectedMetrics.includes(m.id))
            .map(m => m.id);
    }, [selectedMetrics]);

    const totalSteps = 1 + orderedSelectedMetrics.length + 1; // Select + Configs + Summary

    const isFirstStep = currentStepIndex === 0;
    const isLastStep = currentStepIndex === totalSteps - 1;
    const currentMetricIdForConfig = (!isFirstStep && !isLastStep)
        ? orderedSelectedMetrics[currentStepIndex - 1]
        : null;

    // --- Handlers ---

    const toggleMetric = (id: string) => {
        const next = [...selectedMetrics];
        const index = next.indexOf(id);
        if (index > -1) {
            next.splice(index, 1);
        } else {
            next.push(id);
            // Initialize default config if not present
            if (!metricConfigs[id]) {
                const metric = METRICS_CATALOG.find(m => m.id === id);
                setMetricConfigs(prev => ({
                    ...prev,
                    [id]: { ...(metric?.defaultConfig || {}) }
                }));
            }
        }
        setSelectedMetrics(next);
    };

    const updateConfig = (metricId: string, key: string, value: any) => {
        setMetricConfigs(prev => ({
            ...prev,
            [metricId]: {
                ...prev[metricId],
                [key]: value
            }
        }));
    };

    const handleNext = () => {
        if (isLastStep) return;
        setCurrentStepIndex(prev => prev + 1);
    };

    const handleBack = () => {
        if (isFirstStep) return;
        setCurrentStepIndex(prev => prev - 1);
    };

    const handleConfirm = () => {
        // SCENARIO: Stub execution
        const payload = {
            checkpoint: checkpointName,
            metrics: orderedSelectedMetrics,
            configurations: orderedSelectedMetrics.reduce((acc, id) => ({
                ...acc,
                [id]: metricConfigs[id]
            }), {})
        };

        console.log('[EvaluationWizard] CONFIRM ACTION', payload);
        alert(`Evaluation Request Logged to Console!\n\nMetrics: ${orderedSelectedMetrics.join(', ')}`);

        onClose();
    };

    // --- Render ---

    if (!isOpen) return null;

    return (
        <div className="wizard-overlay">
            <div className="wizard-container">
                {/* Header */}
                <div className="wizard-header">
                    <h2>
                        <Activity size={20} />
                        Evaluation Wizard
                    </h2>
                    <button
                        className="btn-wizard-close"
                        onClick={onClose}
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="wizard-content">
                    {/* Progress Dots */}
                    <div className="wizard-steps">
                        {Array.from({ length: totalSteps }).map((_, idx) => (
                            <div
                                key={idx}
                                className={`step-indicator ${idx === currentStepIndex ? 'active' :
                                    idx < currentStepIndex ? 'completed' : ''
                                    }`}
                            />
                        ))}
                    </div>

                    {/* Step Body */}
                    {/* Step Body */}
                    {isFirstStep && (
                        <div className="metrics-step">
                            <h3 className="wizard-step-title-text">Select Evaluation Metrics</h3>
                            <div className="metrics-grid">
                                {METRICS_CATALOG.map((metric) => {
                                    const isSelected = selectedMetrics.includes(metric.id);
                                    return (
                                        <button
                                            key={metric.id}
                                            type="button"
                                            className={`metric-card ${isSelected ? 'selected' : ''}`}
                                            onClick={() => toggleMetric(metric.id)}
                                        >
                                            <div className="metric-icon">{metric.icon}</div>
                                            <div className="metric-check"><Check size={20} strokeWidth={3} /></div>
                                            <h3>{metric.title}</h3>
                                            <p>{metric.description}</p>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {currentMetricIdForConfig && (
                        <MetricConfigStep
                            metricId={currentMetricIdForConfig}
                            config={metricConfigs[currentMetricIdForConfig] || {}}
                            onUpdateConfig={(key, val) => updateConfig(currentMetricIdForConfig, key, val)}
                        />
                    )}

                    {isLastStep && (
                        <SummaryStep
                            checkpointName={checkpointName}
                            configs={metricConfigs}
                            metricsOrder={orderedSelectedMetrics}
                        />
                    )}
                </div>

                {/* Footer Controls */}
                <div className="wizard-footer">
                    <button
                        className="btn-wizard btn-wizard-secondary"
                        onClick={handleBack}
                        disabled={isFirstStep}
                    >
                        <ArrowLeft size={16} />
                        Back
                    </button>

                    {isLastStep ? (
                        <button
                            className="btn-wizard btn-wizard-primary"
                            onClick={handleConfirm}
                        >
                            <Check size={16} />
                            Start Evaluation
                        </button>
                    ) : (
                        <button
                            className="btn-wizard btn-wizard-primary"
                            onClick={handleNext}
                            disabled={isFirstStep && selectedMetrics.length === 0}
                        >
                            Next
                            <ArrowRight size={16} />
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
