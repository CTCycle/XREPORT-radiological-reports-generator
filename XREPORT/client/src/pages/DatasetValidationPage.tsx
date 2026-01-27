import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Settings } from 'lucide-react';
import ValidationDashboard from '../components/ValidationDashboard';
import { runValidation, getValidationJobStatus } from '../services/validationService';
import './DatasetPage.css'; // Reusing styles for now

export default function DatasetValidationPage() {
    const { datasetName } = useParams<{ datasetName: string }>();
    const navigate = useNavigate();

    const [config, setConfig] = useState({
        evalSampleSize: 0.2, // Default 20%
        imgStats: true,
        textStats: true,
        pixDist: true,
    });

    const [isValidating, setIsValidating] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);
    // Reuse the same validation result structure as before
    const [validationResult, setValidationResult] = useState<any | null>(null);

    const handleConfigChange = (key: string, value: any) => {
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

        // Poll for job completion
        const pollInterval = 2000;
        const poll = async () => {
            const { result: status, error: pollError } = await getValidationJobStatus(jobResult.job_id);

            if (pollError) {
                setIsValidating(false);
                setValidationError(pollError);
                return;
            }

            if (!status) {
                setIsValidating(false);
                setValidationError('Failed to get job status');
                return;
            }

            if (status.status === 'completed' && status.result) {
                setIsValidating(false);
                const res = status.result as Record<string, unknown>;
                setValidationResult({
                    success: res.success as boolean ?? true,
                    message: res.message as string ?? 'Validation completed',
                    pixel_distribution: res.pixel_distribution,
                    image_statistics: res.image_statistics,
                    text_statistics: res.text_statistics,
                });
            } else if (status.status === 'failed') {
                setIsValidating(false);
                setValidationError(status.error || 'Validation failed');
            } else if (status.status === 'cancelled') {
                setIsValidating(false);
                setValidationError('Validation was cancelled');
            } else {
                setTimeout(poll, pollInterval);
            }
        };
        poll();
    };

    return (
        <div className="dataset-container">
            <div className="header">
                <button className="btn-icon-text" onClick={() => navigate('/dataset')} style={{ marginBottom: '1rem', border: 'none', background: 'none', padding: 0, color: 'var(--text-secondary)', cursor: 'pointer' }}>
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

                <div className="validation-config-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                        <div className="form-group" style={{ minWidth: '200px' }}>
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

                        <div className="eval-checkboxes" style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
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
                            className="btn btn-primary"
                            onClick={handleRunValidation}
                            disabled={isValidating}
                            style={{ alignSelf: 'flex-start' }}
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
                <div style={{ marginTop: '2rem' }}>
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
