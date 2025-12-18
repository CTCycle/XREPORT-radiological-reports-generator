import { useState } from 'react';
import {
    FolderUp, FileSpreadsheet, Database, Sliders, BarChart2,
    Loader, CheckCircle, AlertCircle
} from 'lucide-react';
import './DatasetPage.css';
import {
    uploadDataset,
    loadDataset,
    validateImagePath,
    ImagePathResponse,
    DatasetUploadResponse,
    LoadDatasetResponse
} from '../services/trainingService';
import FolderBrowser from '../components/FolderBrowser';

interface DatasetProcessingConfig {
    seed: number;
    sampleSize: number;
    validationSize: number;
    splitSeed: number;
    maxReportSize: number;
    tokenizer: string;

    // Dataset Evaluation
    imgStats: boolean;
    textStats: boolean;
    pixDist: boolean;
}

export default function DatasetPage() {
    const [config, setConfig] = useState<DatasetProcessingConfig>({
        // Dataset Processing
        seed: 42,
        sampleSize: 1.0,
        validationSize: 0.2,
        splitSeed: 42,
        maxReportSize: 200,
        tokenizer: 'distilbert-base-uncased',

        // Dataset Evaluation
        imgStats: false,
        textStats: false,
        pixDist: false
    });

    // Upload state
    const [imageFolderPath, setImageFolderPath] = useState('');
    const [imageFolderName, setImageFolderName] = useState('');
    const [imageValidation, setImageValidation] = useState<ImagePathResponse | null>(null);
    const [datasetFile, setDatasetFile] = useState<File | null>(null);
    const [datasetUpload, setDatasetUpload] = useState<DatasetUploadResponse | null>(null);
    const [loadResult, setLoadResult] = useState<LoadDatasetResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);

    // Folder browser modal state
    const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);

    const handleConfigChange = (key: string, value: any) => {
        setConfig(prev => ({ ...prev, [key]: value }));
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
        if (!imageValidation?.valid) {
            setUploadError('Please select an image folder first');
            return;
        }
        if (!datasetUpload?.success) {
            setUploadError('Please upload a dataset file first');
            return;
        }

        setIsLoading(true);
        setUploadError(null);
        setLoadResult(null);

        // Send the full folder path - backend uses it directly
        const { result, error } = await loadDataset({
            image_folder_path: imageFolderPath,
            sample_size: config.sampleSize,
            seed: config.seed,
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
                                        {imageFolderName ? imageFolderName : 'Select directory'}
                                    </div>
                                    {imageValidation && (
                                        <div className={`upload-status ${imageValidation.valid ? 'success' : 'error'}`}>
                                            {imageValidation.valid ? (
                                                <><CheckCircle size={14} /> {imageValidation.image_count} images</>
                                            ) : (
                                                <><AlertCircle size={14} /> {imageValidation.message}</>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Dataset File Upload */}
                                <div className="upload-card" onClick={handleFileUpload}>
                                    <FileSpreadsheet className="upload-icon" />
                                    <div className="upload-text">Upload Data File</div>
                                    <div className="upload-subtext">
                                        {datasetFile ? datasetFile.name : 'Select .csv or .xlsx'}
                                    </div>
                                    {datasetUpload?.success && (
                                        <div className="upload-status success">
                                            <CheckCircle size={14} /> {datasetUpload.row_count} rows, {datasetUpload.column_count} cols
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="load-dataset-section">
                                <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={handleLoadDataset}
                                    disabled={isLoading}
                                >
                                    {isLoading ? (
                                        <><Loader size={14} className="spin" /> Loading...</>
                                    ) : (
                                        'Load Dataset'
                                    )}
                                </button>

                                {uploadError && (
                                    <div className="upload-status error">
                                        <AlertCircle size={14} /> {uploadError}
                                    </div>
                                )}

                                {loadResult?.success && (
                                    <div className="upload-status success">
                                        <CheckCircle size={14} /> Loaded {loadResult.matched_records} records ({loadResult.total_images} images)
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
                                    value={config.sampleSize}
                                    onChange={(e) => handleConfigChange('sampleSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.seed}
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
                                    value={config.validationSize}
                                    onChange={(e) => handleConfigChange('validationSize', parseFloat(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Split Seed</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.splitSeed}
                                    onChange={(e) => handleConfigChange('splitSeed', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Max Report Size</label>
                                <input
                                    type="number"
                                    className="form-input"
                                    value={config.maxReportSize}
                                    onChange={(e) => handleConfigChange('maxReportSize', parseInt(e.target.value))}
                                />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Tokenizer</label>
                                <select
                                    className="form-select"
                                    value={config.tokenizer}
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
                            <button className="btn btn-secondary btn-sm" style={{ marginTop: '0.25rem' }}>
                                View Evaluation
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Folder Browser Modal */}
            <FolderBrowser
                isOpen={folderBrowserOpen}
                onClose={() => setFolderBrowserOpen(false)}
                onSelect={handleFolderSelect}
            />
        </div>
    );
}
