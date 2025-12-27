import { useEffect } from 'react';
import {
    FolderUp, FileSpreadsheet, Database, Sliders, BarChart2,
    Loader, CheckCircle, AlertCircle
} from 'lucide-react';
import './DatasetPage.css';
import {
    uploadDataset,
    loadDataset,
    validateImagePath,
    processDataset,
    getDatasetStatus,
    getDatasetNames,
} from '../services/trainingService';
import { runValidation } from '../services/validationService';
import FolderBrowser from '../components/FolderBrowser';
import ValidationDashboard from '../components/ValidationDashboard';
import { useDatasetPageState } from '../AppStateContext';

export default function DatasetPage() {
    const {
        state,
        updateConfig,
        setImageFolderPath,
        setImageFolderName,
        setImageValidation,
        setDatasetFile,
        setDatasetUpload,
        setLoadResult,
        setIsLoading,
        setUploadError,
        setFolderBrowserOpen,
        setIsProcessing,
        setProcessingResult,
        setDbStatus,
        setDatasetNames,
        setSelectedDataset,
        setIsValidating,
        setValidationResult,
        setValidationError,
    } = useDatasetPageState();

    // Fetch database status and dataset names on component mount
    useEffect(() => {
        const fetchData = async () => {
            // Fetch database status
            const { result: statusResult } = await getDatasetStatus();
            if (statusResult) {
                setDbStatus(statusResult);
            }
            // Fetch dataset names
            const { result: namesResult } = await getDatasetNames();
            if (namesResult) {
                setDatasetNames(namesResult);
                // Auto-select first dataset if available and none selected
                if (namesResult.dataset_names.length > 0 && !state.selectedDataset) {
                    setSelectedDataset(namesResult.dataset_names[0]);
                }
            }
        };
        fetchData();
    }, [setDbStatus, setDatasetNames, setSelectedDataset, state.selectedDataset]);

    // Determine if at least one dataset exists (for LED indicator)
    const hasDatasets = (state.datasetNames?.count ?? 0) > 0;
    // Determine if data is available for processing
    const hasDataForProcessing = state.loadResult?.success || state.dbStatus?.has_data;

    const handleConfigChange = (key: string, value: number | string | boolean) => {
        updateConfig(key as keyof typeof state.config, value);
    };

    const handleBuildDataset = async () => {
        // Validation: Must have data either from fresh load OR existing in database
        if (!hasDataForProcessing) {
            setUploadError("No data available. Please load a dataset or ensure data exists in the database.");
            return;
        }

        setIsProcessing(true);
        setUploadError(null);
        setProcessingResult(null);

        const { result, error } = await processDataset({
            sample_size: state.config.sampleSize,
            validation_size: state.config.validationSize,
            tokenizer: state.config.tokenizer,
            max_report_size: state.config.maxReportSize,
        });

        setIsProcessing(false);

        if (error) {
            setUploadError(error);
        } else if (result) {
            setProcessingResult(result);
        }
    };

    const handleViewEvaluation = async () => {
        // Build metrics list from config
        const metrics: string[] = [];
        if (state.config.imgStats) metrics.push('image_statistics');
        if (state.config.textStats) metrics.push('text_statistics');
        if (state.config.pixDist) metrics.push('pixels_distribution');

        if (metrics.length === 0) {
            setValidationError('Please select at least one validation metric');
            return;
        }

        setIsValidating(true);
        setValidationError(null);
        setValidationResult(null);

        const { result, error } = await runValidation({
            metrics,
            sample_size: state.config.evalSampleSize,
        });

        setIsValidating(false);

        if (error) {
            setValidationError(error);
        } else if (result) {
            setValidationResult(result);
        }
    };

    const handleFolderSelect = async (path: string, _imageCount: number) => {
        setImageFolderPath(path);
        // Extract folder name from path
        const parts = path.split(/[\\/]/);
        const folderName = parts[parts.length - 1] || parts[parts.length - 2] || path;
        setImageFolderName(folderName);
        setUploadError(null);

        // Validate the selected folder on the server
        const { result, error } = await validateImagePath(path);
        if (error) {
            setUploadError(error);
            setImageValidation(null);
        } else if (result) {
            setImageValidation(result);
        }
    };

    const handleFileUpload = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv, .xlsx';
        input.onchange = async (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) {
                setDatasetFile(file);
                setDatasetUpload(null);
                setUploadError(null);

                const { result, error } = await uploadDataset(file);
                if (error) {
                    setUploadError(error);
                } else if (result) {
                    setDatasetUpload(result);
                }
            }
        };
        input.click();
    };

    const handleLoadDataset = async () => {
        if (!state.imageValidation?.valid) {
            setUploadError('Please select an image folder first');
            return;
        }
        if (!state.datasetUpload?.success) {
            setUploadError('Please upload a dataset file first');
            return;
        }

        setIsLoading(true);
        setUploadError(null);
        setLoadResult(null);

        // Send the full folder path - backend uses it directly
        const { result, error } = await loadDataset({
            image_folder_path: state.imageFolderPath,
            sample_size: state.config.sampleSize,
        });

        setIsLoading(false);

        if (error) {
            setUploadError(error);
        } else if (result) {
            setLoadResult(result);
            // Refresh dataset status and names after successful load
            const { result: statusResult } = await getDatasetStatus();
            if (statusResult) {
                setDbStatus(statusResult);
            }
            const { result: namesResult } = await getDatasetNames();
            if (namesResult) {
                setDatasetNames(namesResult);
                // Auto-select the first dataset if none is selected
                if (namesResult.dataset_names.length > 0 && !state.selectedDataset) {
                    setSelectedDataset(namesResult.dataset_names[0]);
                }
            }
        }
    };

    return (
        <div className="dataset-container">
            <div className="header">
                <h1>Dataset Management</h1>
                <p>Upload, process, and evaluate your datasets</p>
            </div>

            <div className="layout-rows">
                {/* Row 1: Data Upload Only */}
                <div className="layout-row row-datasource">
                    <div className="section">
                        <div className="section-title">
                            <Database size={18} />
                            <span>Data Source</span>
                        </div>
                        <div className="upload-row-content">
                            <div className="upload-grid">
                                {/* Image Folder Picker */}
                                <div className="upload-card" onClick={() => setFolderBrowserOpen(true)}>
                                    <FolderUp className="upload-icon" />
                                    <div className="upload-text">Upload Image Folder</div>
                                    <div className="upload-subtext">
                                        {state.imageFolderName ? state.imageFolderName : 'Select directory'}
                                    </div>
                                    {state.imageValidation && (
                                        <div className={`upload-status ${state.imageValidation.valid ? 'success' : 'error'}`}>
                                            {state.imageValidation.valid ? (
                                                <><CheckCircle size={14} /> {state.imageValidation.image_count} images</>
                                            ) : (
                                                <><AlertCircle size={14} /> {state.imageValidation.message}</>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Dataset File Upload */}
                                <div className="upload-card" onClick={handleFileUpload}>
                                    <FileSpreadsheet className="upload-icon" />
                                    <div className="upload-text">Upload Data File</div>
                                    <div className="upload-subtext">
                                        {state.datasetFile ? state.datasetFile.name : 'Select .csv or .xlsx'}
                                    </div>
                                    {state.datasetUpload?.success && (
                                        <div className="upload-status success">
                                            <CheckCircle size={14} /> {state.datasetUpload.row_count} rows, {state.datasetUpload.column_count} cols
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="load-dataset-section">
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={handleLoadDataset}
                                    disabled={state.isLoading}
                                >
                                    {state.isLoading ? (
                                        <><Loader size={14} className="spin" /> Loading...</>
                                    ) : (
                                        'Load Dataset'
                                    )}
                                </button>

                                {state.uploadError && (
                                    <div className="upload-status error">
                                        <AlertCircle size={14} /> {state.uploadError}
                                    </div>
                                )}

                                {state.loadResult?.success && (
                                    <div className="upload-status success">
                                        <CheckCircle size={14} /> Loaded {state.loadResult.matched_records} records ({state.loadResult.total_images} images)
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Row 2: Dataset Processing & Evaluation */}
                <div className="layout-row row-processing-analysis">
                    {/* Processing */}
                    <div className="section">
                        <div className="section-title">
                            <Sliders size={18} />
                            <span>Dataset Processing</span>
                        </div>
                        <div className="config-grid">
                            <div className="form-group">
                                <label className="form-label">Sample Size (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    min="0.01"
                                    max="1.0"
                                    className="form-input"
                                    value={state.config.sampleSize}
                                    onChange={(e) => handleConfigChange('sampleSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Val Split (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    max="1.0"
                                    className="form-input"
                                    value={state.config.validationSize}
                                    onChange={(e) => handleConfigChange('validationSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Max Report Size</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={state.config.maxReportSize}
                                    onChange={(e) => handleConfigChange('maxReportSize', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Tokenizer</label>
                                <select
                                    className="form-select"
                                    value={state.config.tokenizer}
                                    onChange={(e) => handleConfigChange('tokenizer', e.target.value)}
                                >
                                    <option value="distilbert-base-uncased">distilbert-base-uncased</option>
                                    <option value="bert-base-uncased">bert-base-uncased</option>
                                    <option value="roberta-base">roberta-base</option>
                                </select>
                            </div>
                            <div className="form-group span-4">
                                <div className="build-dataset-row">
                                    <span
                                        className={`status-led ${hasDatasets ? 'led-green' : 'led-red'}`}
                                        title={hasDatasets
                                            ? `${state.datasetNames?.count} dataset(s) available`
                                            : 'No datasets in database'
                                        }
                                    />
                                    <select
                                        className="form-select"
                                        value={state.selectedDataset}
                                        onChange={(e) => setSelectedDataset(e.target.value)}
                                        disabled={!hasDatasets}
                                        style={{ flex: 1 }}
                                    >
                                        {!hasDatasets && (
                                            <option value="">No datasets available</option>
                                        )}
                                        {state.datasetNames?.dataset_names.map((name) => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleBuildDataset}
                                        disabled={state.isProcessing}
                                        style={{ justifyContent: 'center' }}
                                    >
                                        {state.isProcessing ? (
                                            <><Loader size={16} className="spin" /> Processing Dataset...</>
                                        ) : (
                                            <><Sliders size={16} /> Build Dataset</>
                                        )}
                                    </button>
                                </div>
                                {state.dbStatus?.has_data && !state.loadResult?.success && (
                                    <div className="upload-status info" style={{ marginTop: '0.5rem' }}>
                                        Using existing data: {state.dbStatus.row_count.toLocaleString()} records in database
                                    </div>
                                )}
                                {state.processingResult?.success && (
                                    <div className="upload-status success" style={{ marginTop: '0.5rem' }}>
                                        <CheckCircle size={14} /> Processed: {state.processingResult.train_samples} train, {state.processingResult.validation_samples} val samples
                                    </div>
                                )}
                                {state.processingResult === undefined && state.uploadError && state.uploadError.includes("Tokenization") && (
                                    <div className="upload-status error" style={{ marginTop: '0.5rem' }}>
                                        <AlertCircle size={14} /> {state.uploadError}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Evaluation */}
                    <div className="section">
                        <div className="section-title">
                            <BarChart2 size={18} />
                            <span>Dataset Evaluation</span>
                        </div>
                        <div className="config-grid" style={{ gridTemplateColumns: '1fr' }}>
                            <div className="form-group">
                                <label className="form-label">Evaluation Sample (0-1)</label>
                                <input
                                    type="number"
                                    step="0.05"
                                    min="0.01"
                                    max="1.0"
                                    className="form-input"
                                    value={state.config.evalSampleSize}
                                    onChange={(e) => handleConfigChange('evalSampleSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="eval-checkboxes">
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={state.config.imgStats}
                                        onChange={(e) => handleConfigChange('imgStats', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Image statistics</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={state.config.textStats}
                                        onChange={(e) => handleConfigChange('textStats', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Text statistics</span>
                                </label>
                                <label className="form-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={state.config.pixDist}
                                        onChange={(e) => handleConfigChange('pixDist', e.target.checked)}
                                    />
                                    <div className="checkbox-visual" />
                                    <span className="checkbox-label">Pixel intensity dist.</span>
                                </label>
                            </div>
                            <button
                                className="btn btn-secondary btn-sm"
                                style={{ marginTop: '0.5rem' }}
                                onClick={handleViewEvaluation}
                                disabled={state.isValidating}
                            >
                                {state.isValidating ? (
                                    <><Loader size={14} className="spin" /> Validating...</>
                                ) : (
                                    'View Evaluation'
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Folder Browser Modal */}
            <FolderBrowser
                isOpen={state.folderBrowserOpen}
                onClose={() => setFolderBrowserOpen(false)}
                onSelect={handleFolderSelect}
            />

            {/* Validation Dashboard */}
            {(state.isValidating || state.validationResult || state.validationError) && (
                <ValidationDashboard
                    isLoading={state.isValidating}
                    validationResult={state.validationResult}
                    error={state.validationError}
                />
            )}
        </div>
    );
}

