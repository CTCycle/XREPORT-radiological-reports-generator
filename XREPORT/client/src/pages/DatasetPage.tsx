import {
    FolderUp, FileSpreadsheet, Database, Sliders, BarChart2,
    Loader, CheckCircle, AlertCircle
} from 'lucide-react';
import './DatasetPage.css';
import {
    uploadDataset,
    loadDataset,
    validateImagePath,
} from '../services/trainingService';
import FolderBrowser from '../components/FolderBrowser';
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
        setFolderBrowserOpen
    } = useDatasetPageState();

    const handleConfigChange = (key: string, value: number | string | boolean) => {
        updateConfig(key as keyof typeof state.config, value);
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
            seed: state.config.seed,
        });

        setIsLoading(false);

        if (error) {
            setUploadError(error);
        } else if (result) {
            setLoadResult(result);
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
                                <label className="form-label">Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={state.config.seed}
                                    onChange={(e) => handleConfigChange('seed', parseInt(e.target.value))}
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
                                <label className="form-label">Split Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={state.config.splitSeed}
                                    onChange={(e) => handleConfigChange('splitSeed', parseInt(e.target.value))}
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
                            <div className="form-group span-3">
                                <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.25rem' }}>
                                    Build Dataset
                                </button>
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
                            <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.25rem' }}>
                                View Evaluation
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
        </div>
    );
}

