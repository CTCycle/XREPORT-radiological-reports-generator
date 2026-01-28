import { useEffect, useMemo, useRef, useState } from 'react';
import {
    FolderUp, FileSpreadsheet, Database, Sliders,
    Loader, CheckCircle, AlertCircle, BarChart2, RefreshCw, Trash2, Eye
} from 'lucide-react';
import './DatasetPage.css';
import {
    uploadDataset,
    loadDataset,
    validateImagePath,
    processDataset,
    getDatasetStatus,
    getDatasetNames,
    getPreparationJobStatus,
    DatasetInfo,
} from '../services/trainingService';
import FolderBrowser from '../components/FolderBrowser';
import { useDatasetPageState } from '../AppStateContext';
import ValidationWizard, { ValidationMetric } from '../components/ValidationWizard';
import ValidationReportModal from '../components/ValidationReportModal';
import {
    runValidation,
    getValidationReport,
    pollValidationJobStatus,
    ValidationReport,
    ValidationResponse,
} from '../services/validationService';
import ImageViewerModal from '../components/ImageViewerModal';
import { deleteDataset } from '../services/trainingService';

const VALIDATION_JOB_STORAGE_KEY = 'xreport.validation.jobs';

type StoredValidationJob = {
    jobId: string;
    metrics: ValidationMetric[];
    sampleSize: number;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress?: number;
};

function loadStoredValidationJobs(): Record<string, StoredValidationJob> {
    try {
        const raw = localStorage.getItem(VALIDATION_JOB_STORAGE_KEY);
        if (!raw) {
            return {};
        }
        const parsed = JSON.parse(raw) as Record<string, StoredValidationJob>;
        if (!parsed || typeof parsed !== 'object') {
            return {};
        }
        return parsed;
    } catch {
        return {};
    }
}

function persistStoredValidationJobs(jobs: Record<string, StoredValidationJob>) {
    try {
        localStorage.setItem(VALIDATION_JOB_STORAGE_KEY, JSON.stringify(jobs));
    } catch {
        // Ignore storage errors (private mode or quota limits)
    }
}

export default function DatasetPage() {
    const [validationWizardOpen, setValidationWizardOpen] = useState(false);
    const [validationRow, setValidationRow] = useState<DatasetInfo | null>(null);
    const [reportModalOpen, setReportModalOpen] = useState(false);
    const [reportDataset, setReportDataset] = useState<DatasetInfo | null>(null);
    const [reportLoading, setReportLoading] = useState(false);
    const [reportError, setReportError] = useState<string | null>(null);
    const [reportResult, setReportResult] = useState<ValidationResponse | null>(null);
    const [reportProgress, setReportProgress] = useState<number | null>(null);
    const [reportStatus, setReportStatus] = useState<
        'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null
    >(null);
    const [reportMetadata, setReportMetadata] = useState<{
        date?: string | null;
        sampleSize?: number | null;
        metrics?: string[];
    } | null>(null);
    const [validationJobs, setValidationJobs] = useState<Record<string, StoredValidationJob>>({});
    const validationPollers = useRef<Record<string, { stop: () => void }>>({});
    const reportDatasetRef = useRef<string | null>(null);

    // Image Viewer State
    const [viewerOpen, setViewerOpen] = useState(false);
    const [viewerDataset, setViewerDataset] = useState<string | null>(null);

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
        setSelectedDatasets,
    } = useDatasetPageState();

    useEffect(() => {
        reportDatasetRef.current = reportDataset?.name ?? null;
    }, [reportDataset]);

    const updateValidationJobs = (
        updater: (prev: Record<string, StoredValidationJob>) => Record<string, StoredValidationJob>
    ) => {
        setValidationJobs(prev => {
            const next = updater(prev);
            persistStoredValidationJobs(next);
            return next;
        });
    };

    const removeValidationJob = (datasetName: string) => {
        updateValidationJobs(prev => {
            if (!prev[datasetName]) {
                return prev;
            }
            const next = { ...prev };
            delete next[datasetName];
            return next;
        });
    };

    const stopValidationPolling = (jobId: string) => {
        const poller = validationPollers.current[jobId];
        if (poller) {
            poller.stop();
            delete validationPollers.current[jobId];
        }
    };

    const startValidationPolling = (datasetName: string, jobId: string, jobMeta: StoredValidationJob) => {
        if (validationPollers.current[jobId]) {
            return;
        }

        const poller = pollValidationJobStatus(
            jobId,
            (status) => {
                updateValidationJobs(prev => {
                    const current = prev[datasetName] ?? jobMeta;
                    return {
                        ...prev,
                        [datasetName]: {
                            ...current,
                            status: status.status,
                            progress: status.progress,
                        },
                    };
                });

                if (reportDatasetRef.current === datasetName) {
                    setReportProgress(status.progress);
                    setReportStatus(status.status);
                    setReportLoading(status.status === 'running' || status.status === 'pending');
                }
            },
            (status) => {
                stopValidationPolling(jobId);
                removeValidationJob(datasetName);

                if (reportDatasetRef.current === datasetName) {
                    setReportStatus(status.status);
                    setReportProgress(status.progress ?? 100);
                    setReportLoading(false);
                }

                if (status.status === 'completed' && status.result) {
                    const res = status.result as Record<string, unknown>;
                    if (reportDatasetRef.current === datasetName) {
                        setReportResult({
                            success: (res.success as boolean) ?? true,
                            message: (res.message as string) ?? 'Validation completed',
                            pixel_distribution: res.pixel_distribution as ValidationResponse['pixel_distribution'],
                            image_statistics: res.image_statistics as ValidationResponse['image_statistics'],
                            text_statistics: res.text_statistics as ValidationResponse['text_statistics'],
                        });
                        setReportError(null);
                    }
                    void (async () => {
                        const { result: namesResult } = await getDatasetNames();
                        if (namesResult) {
                            setDatasetNames(namesResult);
                        }
                    })();
                } else if (status.status === 'failed') {
                    if (reportDatasetRef.current === datasetName) {
                        setReportError(status.error || 'Validation failed');
                    }
                } else if (status.status === 'cancelled') {
                    if (reportDatasetRef.current === datasetName) {
                        setReportError('Validation was cancelled');
                    }
                }
            },
            (error) => {
                stopValidationPolling(jobId);
                removeValidationJob(datasetName);
                if (reportDatasetRef.current === datasetName) {
                    setReportLoading(false);
                    setReportStatus('failed');
                    setReportError(error);
                }
            },
            2000
        );

        validationPollers.current[jobId] = poller;
    };

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
                if (namesResult.datasets.length > 0 && (state.selectedDatasets || []).length === 0) {
                    if (setSelectedDatasets) setSelectedDatasets([namesResult.datasets[0].name]);
                }
            }
        };
        fetchData();
    }, [setDbStatus, setDatasetNames, setSelectedDatasets, state.selectedDatasets]);

    useEffect(() => {
        const storedJobs = loadStoredValidationJobs();
        const entries = Object.entries(storedJobs);
        if (entries.length === 0) {
            return;
        }
        setValidationJobs(storedJobs);
        entries.forEach(([datasetName, job]) => {
            startValidationPolling(datasetName, job.jobId, job);
        });
    }, []);

    useEffect(() => {
        return () => {
            Object.values(validationPollers.current).forEach(poller => poller.stop());
            validationPollers.current = {};
        };
    }, []);

    // Determine if at least one dataset exists (for LED indicator)
    const hasDatasets = (state.datasetNames?.count ?? 0) > 0;
    // Determine if data is available for processing
    const hasDataForProcessing = state.loadResult?.success || (state.dbStatus?.has_data && state.selectedDatasets.length > 0);

    const handleConfigChange = (key: string, value: number | string | boolean) => {
        updateConfig(key as keyof typeof state.config, value);
    };

    const handleBuildDataset = async () => {
        // Validation: Must have data either from fresh load OR existing in database
        if (!hasDataForProcessing) {
            setUploadError("No data available. Please load a dataset or ensure data exists in the database.");
            return;
        }
        const selectedNames = state.selectedDatasets || [];
        if (selectedNames.length !== 1) {
            setUploadError('Select exactly one dataset to process.');
            return;
        }

        setIsProcessing(true);
        setUploadError(null);
        setProcessingResult(null);

        const { result: jobResult, error: startError } = await processDataset({
            dataset_name: selectedNames[0],
            sample_size: state.config.sampleSize,
            validation_size: state.config.validationSize,
            tokenizer: state.config.tokenizer,
            max_report_size: state.config.maxReportSize,
        });

        if (startError || !jobResult) {
            setIsProcessing(false);
            setUploadError(startError || 'Failed to start processing job');
            return;
        }

        // Poll for job completion
        const pollInterval = 2000;
        const poll = async () => {
            const { result: status, error: pollError } = await getPreparationJobStatus(jobResult.job_id);

            if (pollError) {
                setIsProcessing(false);
                setUploadError(pollError);
                return;
            }

            if (!status) {
                setIsProcessing(false);
                setUploadError('Failed to get job status');
                return;
            }

            if (status.status === 'completed' && status.result) {
                setIsProcessing(false);
                setProcessingResult({
                    success: true,
                    total_samples: (status.result as Record<string, number>).total_samples ?? 0,
                    train_samples: (status.result as Record<string, number>).train_samples ?? 0,
                    validation_samples: (status.result as Record<string, number>).validation_samples ?? 0,
                    vocabulary_size: (status.result as Record<string, number>).vocabulary_size ?? 0,
                    message: 'Dataset processed successfully',
                });
            } else if (status.status === 'failed') {
                setIsProcessing(false);
                setUploadError(status.error || 'Processing failed');
            } else if (status.status === 'cancelled') {
                setIsProcessing(false);
                setUploadError('Processing was cancelled');
            } else {
                // Still running, poll again
                setTimeout(poll, pollInterval);
            }
        };
        poll();
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

    const currentValidationSelection = useMemo<ValidationMetric[]>(() => {
        const selection: ValidationMetric[] = [];
        if (state.config.pixDist) selection.push('pixels_distribution');
        if (state.config.textStats) selection.push('text_statistics');
        if (state.config.imgStats) selection.push('image_statistics');
        return selection;
    }, [state.config.pixDist, state.config.textStats, state.config.imgStats]);

    const handleValidationConfirm = async (config: { metrics: ValidationMetric[]; row: DatasetInfo | null; sampleFraction: number }) => {
        updateConfig('pixDist', config.metrics.includes('pixels_distribution'));
        updateConfig('textStats', config.metrics.includes('text_statistics'));
        updateConfig('imgStats', config.metrics.includes('image_statistics'));
        updateConfig('evalSampleSize', config.sampleFraction);
        setValidationRow(config.row);
        setValidationWizardOpen(false);

        if (!config.row) {
            return;
        }
        const selectedRow = config.row;

        if (config.metrics.length === 0) {
            setReportError('Please select at least one validation metric.');
            setReportResult(null);
            setReportProgress(null);
            setReportStatus(null);
            setReportLoading(false);
            setReportDataset(selectedRow);
            setReportMetadata({ sampleSize: config.sampleFraction, metrics: [] });
            setReportModalOpen(true);
            return;
        }

        setReportDataset(selectedRow);
        setReportMetadata({ sampleSize: config.sampleFraction, metrics: config.metrics });
        setReportResult(null);
        setReportError(null);
        setReportProgress(0);
        setReportStatus('pending');
        setReportLoading(true);
        setReportModalOpen(true);

        const { result: jobResult, error: startError } = await runValidation({
            dataset_name: selectedRow.name,
            metrics: config.metrics,
            sample_size: config.sampleFraction,
        });

        if (startError || !jobResult) {
            setReportLoading(false);
            setReportError(startError || 'Failed to start validation job');
            return;
        }

        const jobMeta: StoredValidationJob = {
            jobId: jobResult.job_id,
            metrics: config.metrics,
            sampleSize: config.sampleFraction,
            status: jobResult.status,
            progress: 0,
        };
        updateValidationJobs(prev => ({
            ...prev,
            [selectedRow.name]: jobMeta,
        }));
        setReportProgress(0);
        setReportStatus(jobResult.status);
        setReportLoading(true);
        startValidationPolling(selectedRow.name, jobResult.job_id, jobMeta);
    };

    const handleVisualizeReport = async (dataset: DatasetInfo) => {
        setReportDataset(dataset);
        setReportResult(null);
        setReportError(null);
        setReportMetadata(null);
        setReportProgress(null);
        setReportStatus(null);
        setReportLoading(true);
        setReportModalOpen(true);

        const activeJob = validationJobs[dataset.name];
        if (activeJob && activeJob.status !== 'completed' && activeJob.status !== 'failed' && activeJob.status !== 'cancelled') {
            setReportMetadata({
                sampleSize: activeJob.sampleSize,
                metrics: activeJob.metrics,
            });
            setReportProgress(activeJob.progress ?? 0);
            setReportStatus(activeJob.status ?? 'running');
            setReportLoading(activeJob.status === 'running' || activeJob.status === 'pending');
            startValidationPolling(dataset.name, activeJob.jobId, activeJob);
            return;
        }

        const { result, error } = await getValidationReport(dataset.name);
        if (error || !result) {
            setReportLoading(false);
            setReportError(error || 'Failed to load validation report');
            return;
        }

        const report = result as ValidationReport;
        setReportMetadata({
            date: report.date,
            sampleSize: report.sample_size ?? null,
            metrics: report.metrics,
        });
        setReportResult({
            success: true,
            message: 'Validation report loaded',
            pixel_distribution: report.pixel_distribution,
            image_statistics: report.image_statistics,
            text_statistics: report.text_statistics,
        });
        setReportProgress(100);
        setReportStatus('completed');
        setReportLoading(false);
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
                if (namesResult.datasets.length > 0 && (state.selectedDatasets || []).length === 0) {
                    if (setSelectedDatasets) setSelectedDatasets([namesResult.datasets[0].name]);
                }
            }
        }
    };

    const handleDeleteDataset = async (datasetName: string) => {
        if (!confirm(`Are you sure you want to delete dataset "${datasetName}"?`)) {
            return;
        }

        const { error } = await deleteDataset(datasetName);
        if (error) {
            setUploadError(error); // Or a toast if available
        } else {
            // Refresh list
            const { result: namesResult } = await getDatasetNames();
            if (namesResult) {
                setDatasetNames(namesResult);
                if (setSelectedDatasets) {
                    // Remove deleted dataset from selection
                    setSelectedDatasets((state.selectedDatasets || []).filter(n => n !== datasetName));
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
                                    <div className="upload-hint">DICOM, PNG, JPG</div>
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
                                    <div className="upload-hint">Reports & Metadata</div>
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

                {/* Row 2: Dataset Processing */}
                <div className="layout-row">
                    <div className="section">
                        <div className="section-title">
                            <Sliders size={18} />
                            <span>Dataset Processing</span>
                        </div>

                        {/* Dataset Processing Split View */}
                        <div className="dataset-processing-split">
                            {/* Left: Dataset Grid */}
                            <div className="dataset-grid-section">
                                <div className="dataset-table-container">
                                    <div className="dataset-table-header-row">
                                        <span className="dataset-table-title">Available Datasets</span>
                                        <button
                                            className="btn-icon-small"
                                            style={{ marginLeft: 'auto' }}
                                            onClick={async (e) => {
                                                e.preventDefault();
                                                const { result } = await getDatasetNames();
                                                if (result) setDatasetNames(result);
                                            }}
                                            title="Refresh datasets"
                                        >
                                            <RefreshCw size={16} />
                                        </button>
                                    </div>
                                    <div className="dataset-table">
                                        <div className="dataset-table-header">
                                            <span style={{ textAlign: 'center' }}>Actions</span>
                                            <span>Name</span>
                                            <span>Source</span>
                                            <span style={{ textAlign: 'right' }}>Rows</span>
                                        </div>
                                        <div className="dataset-table-body">
                                            {!hasDatasets && (
                                                <div className="dataset-table-empty">
                                                    Please upload at least one dataset
                                                </div>
                                            )}
                                            {state.datasetNames?.datasets.map((dataset) => {
                                                const isSelected = (state.selectedDatasets || []).includes(dataset.name);
                                                const activeJob = validationJobs[dataset.name];
                                                const hasActiveJob = Boolean(
                                                    activeJob &&
                                                    activeJob.status !== 'completed' &&
                                                    activeJob.status !== 'failed' &&
                                                    activeJob.status !== 'cancelled'
                                                );
                                                const canViewReport = dataset.has_validation_report || hasActiveJob;
                                                return (
                                                    <div
                                                        key={dataset.name}
                                                        className={`dataset-table-row ${isSelected ? 'selected' : ''}`}
                                                        onClick={() => {
                                                            // Prevent row selection when clicking the validation icon if it propagates
                                                            let newSelection: string[];
                                                            const currentSelection = state.selectedDatasets || [];
                                                            if (isSelected) {
                                                                // Deselect
                                                                newSelection = currentSelection.filter(n => n !== dataset.name);
                                                            } else {
                                                                // Select (append)
                                                                newSelection = [...currentSelection, dataset.name];
                                                            }
                                                            if (setSelectedDatasets) {
                                                                setSelectedDatasets(newSelection);
                                                            }
                                                        }}
                                                    >
                                                        <div className="dataset-actions" onClick={(e) => {
                                                            e.stopPropagation(); // Don't toggle selection
                                                        }}>
                                                            <button
                                                                type="button"
                                                                className="btn-icon-small"
                                                                title="Delete Dataset"
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    handleDeleteDataset(dataset.name);
                                                                }}
                                                            >
                                                                <Trash2 size={16} />
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="btn-icon-small"
                                                                title="View Images"
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    setViewerDataset(dataset.name);
                                                                    setViewerOpen(true);
                                                                }}
                                                            >
                                                                <Eye size={16} />
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="btn-icon-small"
                                                                title="Run Validation"
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    setValidationRow(dataset);
                                                                    setValidationWizardOpen(true);
                                                                }}
                                                            >
                                                                <CheckCircle size={16} />
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="btn-icon-small"
                                                                title={
                                                                    hasActiveJob
                                                                        ? 'Validation in progress'
                                                                        : dataset.has_validation_report
                                                                            ? 'Visualize Report'
                                                                            : 'No report available'
                                                                }
                                                                disabled={!canViewReport}
                                                                onClick={(e) => {
                                                                    e.preventDefault();
                                                                    e.stopPropagation();
                                                                    if (!canViewReport) return;
                                                                    handleVisualizeReport(dataset);
                                                                }}
                                                            >
                                                                <BarChart2 size={16} />
                                                            </button>
                                                        </div>
                                                        <span className="dataset-name">{dataset.name}</span>
                                                        <span className="dataset-path" title={dataset.folder_path}>{dataset.folder_path}</span>
                                                        <span className="dataset-rows">{dataset.row_count.toLocaleString()}</span>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Vertical Separator */}
                            <div className="vertical-separator"></div>

                            {/* Right: Processing Configuration */}
                            <div className="processing-config-section">
                                <div className="processing-content">
                                    <h4 className="config-panel-title">Configurations</h4>
                                    <div className="config-grid-compact-vertical">
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
                                    </div>

                                    {/* Action Button at Bottom Right */}
                                    <div className="processing-actions">
                                        <button
                                            className="btn btn-primary"
                                            onClick={handleBuildDataset}
                                            disabled={state.isProcessing}
                                        >
                                            {state.isProcessing ? (
                                                <><Loader size={16} className="spin" /> Processing...</>
                                            ) : (
                                                <><Sliders size={16} /> Build Dataset</>
                                            )}
                                        </button>
                                    </div>

                                    {state.processingResult?.success && (
                                        <div className="upload-status success text-right">
                                            <CheckCircle size={14} /> Processed: {state.processingResult.train_samples} train, {state.processingResult.validation_samples} val
                                        </div>
                                    )}
                                    {state.processingResult === undefined && state.uploadError && state.uploadError.includes("Tokenization") && (
                                        <div className="upload-status error text-right">
                                            <AlertCircle size={14} /> {state.uploadError}
                                        </div>
                                    )}
                                </div>
                            </div>
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

            <ValidationWizard
                isOpen={validationWizardOpen}
                row={validationRow}
                initialSelected={currentValidationSelection}
                onClose={() => setValidationWizardOpen(false)}
                onConfirm={handleValidationConfirm}
            />
            {/* Validation Report Modal */}
            <ValidationReportModal
                isOpen={reportModalOpen}
                datasetName={reportDataset?.name ?? null}
                isLoading={reportLoading}
                validationResult={reportResult}
                error={reportError}
                progress={reportProgress}
                status={reportStatus}
                metadata={reportMetadata}
                onClose={() => setReportModalOpen(false)}
            />

            {/* Image Viewer Modal */}
            <ImageViewerModal
                isOpen={viewerOpen}
                datasetName={viewerDataset}
                onClose={() => {
                    setViewerOpen(false);
                    setViewerDataset(null);
                }}
            />
        </div>
    );
}
