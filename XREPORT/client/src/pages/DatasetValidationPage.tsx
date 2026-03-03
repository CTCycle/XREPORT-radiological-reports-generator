import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Settings } from 'lucide-react';
import ValidationDashboard from '../components/ValidationDashboard';
import {
    runValidation,
    pollValidationJobStatus,
    parseValidationResponse,
    ValidationResponse,
} from '../services/validationService';
import { useManagedPoller } from '../hooks/useManagedPoller';
import './DatasetPage.css'; // Reusing styles for now

type ValidationConfig = {
    evalSampleSize: number;
    imgStats: boolean;
    textStats: boolean;
    pixDist: boolean;
};

export default function DatasetValidationPage() {
    const { datasetName } = useParams<{ datasetName: string }>();
    const navigate = useNavigate();

    const [config, setConfig] = useState<ValidationConfig>({
        evalSampleSize: 0.2, // Default 20%
        imgStats: true,
        textStats: true,
        pixDist: true,
    });

    const [isValidating, setIsValidating] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);
    const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null);
    const { startPolling: startValidationPolling, stopPolling: stopValidationPolling } = useManagedPoller();

    const handleConfigChange = <K extends keyof ValidationConfig>(key: K, value: ValidationConfig[K]) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    const handleRunValidation = async () => {
        const metrics: string[] = [];
        if (config.imgStats) metrics.push('image_statistics');
        if (config.textStats) metrics.push('text_statistics');
        if (config.pixDist) metrics.push('pixels_distribution');

        if (metrics.length === 0) {
            setValidationError('Please select at least one validation metric');
            return;
        }

        setIsValidating(true);
        setValidationError(null);
        setValidationResult(null);

        const { result: jobResult, error: startError } = await runValidation({
            dataset_name: datasetName || 'default',
            metrics,
            sample_size: config.evalSampleSize,
        });

        if (startError || !jobResult) {
            setIsValidating(false);
            setValidationError(startError || 'Failed to start validation job');
            return;
        }

        const pollInterval = (jobResult.poll_interval ?? 2) * 1000;
        startValidationPolling(() => pollValidationJobStatus(
            jobResult.job_id,
            () => {
                // Progress updates are shown by the dashboard loading state in this page.
            },
            (status) => {
                stopValidationPolling();
                setIsValidating(false);
                if (status.status === 'completed' && status.result) {
                    setValidationResult(parseValidationResponse(status.result));
                    return;
                }
                if (status.status === 'failed') {
                    setValidationError(status.error || 'Validation failed');
                    return;
                }
                if (status.status === 'cancelled') {
                    setValidationError('Validation was cancelled');
                }
            },
            (pollError) => {
                stopValidationPolling();
                setIsValidating(false);
                setValidationError(pollError);
            },
            pollInterval
        ));
    };

    return (
        <div className="dataset-container">
            <div className="header">
                <button className="btn-icon-text validation-back-button" onClick={() => navigate('/dataset')}>
                    <ArrowLeft size={20} />
                    <span>Back to Datasets</span>
                </button>
                <h1>Validation Wizard</h1>
                <p>Validate and analyze: <strong>{datasetName}</strong></p>
            </div>

            <div className="section">
                <div className="section-title">
                    <Settings size={18} />
                    <span>Configuration</span>
                </div>

                <div className="validation-config-panel validation-config-layout">
                    <div className="validation-config-row">
                        <div className="form-group validation-config-min-width">
                            <label className="form-label">Evaluation Sample Size (0-1)</label>
                            <input
                                type="number"
                                step="0.05"
                                min="0.01"
                                max="1.0"
                                className="form-input"
                                value={config.evalSampleSize}
                                onChange={(e) => handleConfigChange('evalSampleSize', parseFloat(e.target.value))}
                            />
                        </div>

                        <div className="eval-checkboxes validation-checkboxes-row">
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.imgStats}
                                    onChange={(e) => handleConfigChange('imgStats', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Image statistics</span>
                            </label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.textStats}
                                    onChange={(e) => handleConfigChange('textStats', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Text statistics</span>
                            </label>
                            <label className="form-checkbox">
                                <input
                                    type="checkbox"
                                    checked={config.pixDist}
                                    onChange={(e) => handleConfigChange('pixDist', e.target.checked)}
                                />
                                <div className="checkbox-visual" />
                                <span className="checkbox-label">Pixel intensity dist.</span>
                            </label>
                        </div>
                    </div>

                    <div className="form-actions">
                        <button
                            className="btn btn-primary validation-run-button"
                            onClick={handleRunValidation}
                            disabled={isValidating}
                        >
                            {isValidating ? (
                                <>Running Validation...</>
                            ) : (
                                <><Play size={16} /> Run Validation</>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Validation Results */}
            {(isValidating || validationResult || validationError) && (
                <div className="validation-results-spacing">
                    <ValidationDashboard
                        isLoading={isValidating}
                        validationResult={validationResult}
                        error={validationError}
                    />
                </div>
            )}
        </div>
    );
}
