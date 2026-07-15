import { useRef, DragEvent, ChangeEvent, useEffect, useMemo } from 'react';
import {
    ImagePlus, ChevronLeft, ChevronRight, Trash, Trash2, FileText,
    Sparkles, ArrowRight, Copy, Check, Loader2
} from 'lucide-react';
import './InferencePage.css';
import { useInferencePageState } from '../AppStateContext';
import type { GenerationProfile } from '../types/inferenceApi';
import { useAsyncJob } from '../hooks/useAsyncJob';
import { asRecord, readString, readStringArray } from '../common/parsers';
import {
    getInferenceModels,
    generateReports,
    getInferenceJobStatus,
} from '../services/inferenceService';

const MAX_IMAGES = 16;

function isGenerationProfile(value: string): value is GenerationProfile {
    return value === 'deterministic' || value === 'concise' || value === 'detailed';
}

function parseGenerationProfile(value: string, fallback: GenerationProfile): GenerationProfile {
    return isGenerationProfile(value) ? value : fallback;
}

function readStringMap(value: unknown): Record<string, string> | undefined {
    const record = asRecord(value);
    if (!record) {
        return undefined;
    }

    const entries = Object.entries(record);
    if (entries.some(([, entryValue]) => readString(entryValue) === undefined)) {
        return undefined;
    }

    return Object.fromEntries(entries.map(([key, entryValue]) => [key, readString(entryValue) ?? '']));
}

function toReportsByIndex(
    result: unknown,
    images: File[],
): Record<number, string> {
    const payload = asRecord(result);
    if (!payload) {
        return {};
    }

    const reports = readStringMap(payload.reports);
    const reportsOrdered = readStringArray(payload.reports_ordered);
    const reportFilenames = readStringArray(payload.report_filenames);
    const reportsByIndex: Record<number, string> = {};

    if (reportsOrdered && reportsOrdered.length > 0) {
        reportsOrdered.forEach((report, index) => {
            if (report) {
                reportsByIndex[index] = report;
            }
        });
        return reportsByIndex;
    }

    if (!reports) {
        return reportsByIndex;
    }

    if (reportFilenames && reportFilenames.length > 0) {
        reportFilenames.forEach((filename, index) => {
            const report = reports[filename];
            if (report !== undefined) {
                reportsByIndex[index] = report;
            }
        });
    } else {
        images.forEach((image, index) => {
            const report = reports[image.name];
            if (report !== undefined) {
                reportsByIndex[index] = report;
            }
        });
    }

    if (Object.keys(reportsByIndex).length === 0) {
        Object.values(reports).forEach((report, index) => {
            reportsByIndex[index] = report;
        });
    }

    return reportsByIndex;
}

export default function InferencePage() {
    const {
        state,
        setImages,
        setCurrentIndex,
        setGeneratedReport,
        setIsGenerating,
        setIsCopied,
        clearImages,
        setSelectedModelRef,
        setGenerationProfile,
        setClinicalContext,
        setModelAvailability,
        setIsLoadingModels,
        setReports,
        setStreamingTokens,
        setCurrentStreamingIndex,
    } = useInferencePageState();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const generationJob = useAsyncJob({
        startJob: ({ images, modelRef, generationProfile, clinicalContext }: { images: File[]; modelRef: string; generationProfile: GenerationProfile; clinicalContext: string }) =>
            generateReports(images, modelRef, generationProfile, clinicalContext),
        getStatus: getInferenceJobStatus,
        onUpdate: (status) => {
            const reportsByIndex = toReportsByIndex(status.result, state.images);
            if (Object.keys(reportsByIndex).length > 0) {
                setReports(reportsByIndex);
                if (reportsByIndex[state.currentIndex] !== undefined) {
                    setGeneratedReport(reportsByIndex[state.currentIndex]);
                }
            }
        },
        onComplete: (status) => {
            if (status.status === 'failed') {
                console.error('Generation failed:', status.error);
            }
            setIsGenerating(false);
        },
    });
    // Fetch the local-only model catalog on mount.
    useEffect(() => {
        const fetchModels = async () => {
            setIsLoadingModels(true);
            const { result, error } = await getInferenceModels();
            if (result) {
                setModelAvailability(result.models);
                const readyModels = result.models.filter(model => model.status === 'ready');
                if (readyModels.length > 0 && !state.selectedModelRef) {
                    setSelectedModelRef(readyModels[0].model_ref);
                }
            } else if (error) {
                console.error('Failed to fetch inference models:', error);
            }
            setIsLoadingModels(false);
        };
        fetchModels();
    }, []);

    // Handle file selection
    const handleFileSelect = (files: FileList | null) => {
        if (!files || files.length === 0) return;

        const imageFiles = Array.from(files).filter(file =>
            file.type.startsWith('image/')
        );

        if (imageFiles.length === 0) return;

        const availableSlots = MAX_IMAGES - state.images.length;
        if (availableSlots <= 0) return;

        // Enforce max 16 images, append when possible
        const limitedImages = imageFiles.slice(0, availableSlots);

        if (limitedImages.length > 0) {
            const hasExistingImages = state.images.length > 0;
            const nextImages = hasExistingImages
                ? [...state.images, ...limitedImages]
                : limitedImages;
            setImages(nextImages);
            if (!hasExistingImages) {
                setCurrentIndex(0);
            }
            setGeneratedReport('');
            setReports({});
            setStreamingTokens('');
            setCurrentStreamingIndex(-1);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    // Drag and drop handlers
    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        handleFileSelect(e.dataTransfer.files);
    };

    const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
        handleFileSelect(e.target.files);
    };

    const openFileDialog = () => {
        fileInputRef.current?.click();
    };

    // Navigation with synchronized report display
    const goToIndex = (newIndex: number) => {
        setCurrentIndex(newIndex);
        const reportForIndex = state.reports[newIndex];
        setGeneratedReport(reportForIndex ?? '');
    };

    const goToPrevious = () => {
        const newIndex = Math.max(0, state.currentIndex - 1);
        goToIndex(newIndex);
    };

    const goToNext = () => {
        const newIndex = Math.min(state.images.length - 1, state.currentIndex + 1);
        goToIndex(newIndex);
    };

    const reportForIndex = state.reports[state.currentIndex];

    useEffect(() => {
        if (state.isGenerating && state.currentStreamingIndex === state.currentIndex) {
            return;
        }
        if (reportForIndex !== undefined && reportForIndex !== state.generatedReport) {
            setGeneratedReport(reportForIndex);
            return;
        }
        if (reportForIndex === undefined && state.generatedReport) {
            setGeneratedReport('');
        }
    }, [
        reportForIndex,
        state.currentIndex,
        state.currentStreamingIndex,
        state.generatedReport,
        state.isGenerating,
        setGeneratedReport,
    ]);

    // Clear images
    const handleClearImages = () => {
        clearImages();
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleRemoveCurrentImage = () => {
        if (state.images.length === 0) return;

        const removeIndex = state.currentIndex;
        const nextImages = state.images.filter((_, idx) => idx !== removeIndex);
        const nextReports: Record<number, string> = {};

        Object.entries(state.reports).forEach(([key, value]) => {
            const index = Number(key);
            if (Number.isNaN(index)) return;
            if (index < removeIndex) {
                nextReports[index] = value;
            } else if (index > removeIndex) {
                nextReports[index - 1] = value;
            }
        });

        setImages(nextImages);
        setReports(nextReports);
        setStreamingTokens('');
        setCurrentStreamingIndex(-1);

        if (nextImages.length === 0) {
            setCurrentIndex(0);
            setGeneratedReport('');
            return;
        }

        const nextIndex = Math.min(removeIndex, nextImages.length - 1);
        setCurrentIndex(nextIndex);
        if (nextReports[nextIndex]) {
            setGeneratedReport(nextReports[nextIndex]);
        } else {
            setGeneratedReport('');
        }
    };

    // Generate reports
    const handleGenerateReport = async () => {
        if (state.images.length === 0 || !state.selectedModelRef) return;

        setIsGenerating(true);
        setStreamingTokens('');
        setReports({});
        setGeneratedReport('');
        setCurrentStreamingIndex(-1);

        // Call generate endpoint - now returns job_id
        const jobResult = await generationJob.start({
            images: state.images,
            modelRef: state.selectedModelRef,
            generationProfile: state.generationProfile,
            clinicalContext: state.clinicalContext,
        });

        if (!jobResult) {
            console.error('Generation failed:', generationJob.error);
            setIsGenerating(false);
            return;
        }
    };

    // Copy report to clipboard
    const copyReport = async () => {
        if (!state.generatedReport) return;

        try {
            await navigator.clipboard.writeText(state.generatedReport);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    // Get current image URL
    const currentImage = state.images[state.currentIndex] ?? null;
    const currentImageUrl = useMemo(() => {
        if (!currentImage) {
            return null;
        }
        return URL.createObjectURL(currentImage);
    }, [currentImage]);

    useEffect(() => {
        return () => {
            if (currentImageUrl) {
                URL.revokeObjectURL(currentImageUrl);
            }
        };
    }, [currentImageUrl]);

    // Determine what to display in report panel
    const displayContent = state.isGenerating && state.currentStreamingIndex === state.currentIndex
        ? state.streamingTokens
        : reportForIndex ?? state.generatedReport ?? '';

    return (
        <div className="inference-container">
            {/* Control Panel */}
            <div className="inference-header">
                <h1>Inference</h1>
                <p>
                    Generate detailed radiological reports from X-ray scans using advanced AI.
                    Upload medical images, verify the selected research model in the settings bar,
                    and click 'Generate Report' to obtain a structured analysis of the findings.
                </p>
                <p><strong>Research use only. Generated drafts are not clinically approved and require qualified review.</strong></p>
            </div>

            <div className="inference-main">
                {/* Left Column - Image Canvas */}
                <div className="inference-panel">
                    <div className="panel-header">
                        <ImagePlus size={18} />
                        <h2>X-Ray Images</h2>
                        {state.images.length > 0 && (
                            <span className="image-limit-badge">
                                {state.images.length} / {MAX_IMAGES}
                            </span>
                        )}
                    </div>
                    <div className="panel-content">
                        {state.images.length === 0 ? (
                            <div
                                className="image-dropzone"
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                onClick={openFileDialog}
                            >
                                <ImagePlus className="dropzone-icon" />
                                <div className="dropzone-text">
                                    <div className="dropzone-title">
                                        Drop images here or click to upload
                                    </div>
                                    <div className="dropzone-subtitle">
                                        Maximum {MAX_IMAGES} X-ray images
                                    </div>
                                </div>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    multiple
                                    onChange={handleInputChange}
                                    className="visually-hidden-input"
                                />
                            </div>
                        ) : (
                            <div className="image-viewer">
                                <div className="image-display">
                                    {currentImageUrl && (
                                        <img
                                            src={currentImageUrl}
                                            alt={`X-ray ${state.currentIndex + 1}`}
                                        />
                                    )}

                                    {state.images.length > 1 && (
                                        <>
                                            <button
                                                type="button"
                                                className="nav-arrow prev"
                                                onClick={goToPrevious}
                                                disabled={state.currentIndex === 0}
                                                aria-label="Previous image"
                                            >
                                                <ChevronLeft size={20} />
                                            </button>
                                            <button
                                                type="button"
                                                className="nav-arrow next"
                                                onClick={goToNext}
                                                disabled={state.currentIndex === state.images.length - 1}
                                                aria-label="Next image"
                                            >
                                                <ChevronRight size={20} />
                                            </button>
                                            <div className="image-counter">
                                                {state.currentIndex + 1} / {state.images.length}
                                            </div>
                                        </>
                                    )}
                                </div>

                                <div className="image-controls">
                                    <button
                                        type="button"
                                        className="btn-icon"
                                        onClick={openFileDialog}
                                        aria-label="Add more images"
                                        title="Add more images"
                                        disabled={state.images.length >= MAX_IMAGES}
                                    >
                                        <ImagePlus />
                                    </button>
                                    <button
                                        type="button"
                                        className="btn-icon"
                                        onClick={handleRemoveCurrentImage}
                                        aria-label="Remove current image"
                                        title="Remove current image"
                                        disabled={state.images.length === 0 || state.isGenerating}
                                    >
                                        <Trash />
                                    </button>
                                    <button
                                        type="button"
                                        className="btn-icon"
                                        onClick={handleClearImages}
                                        aria-label="Clear all images"
                                        title="Clear all images"
                                        disabled={state.images.length === 0 || state.isGenerating}
                                    >
                                        <Trash2 />
                                    </button>
                                </div>

                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    multiple
                                    onChange={handleInputChange}
                                    className="visually-hidden-input"
                                />
                            </div>
                        )}
                    </div>
                    <div className="panel-footer">
                        <div className="panel-control-item">
                            <label htmlFor="model-select">Model:</label>
                            <select
                                id="model-select"
                                value={state.selectedModelRef}
                                onChange={(e) => setSelectedModelRef(e.target.value)}
                                disabled={state.isLoadingModels || state.isGenerating}
                            >
                                {state.modelAvailability.length === 0 && (
                                    <option value="">No models discovered</option>
                                )}
                                {state.modelAvailability.map((model) => (
                                    <option key={model.model_ref} value={model.model_ref} disabled={model.status !== 'ready'}>
                                        {model.display_name} — {model.status}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="panel-control-item">
                            <label htmlFor="profile-select">Profile:</label>
                            <select
                                id="profile-select"
                                value={state.generationProfile}
                                onChange={(e) => setGenerationProfile(parseGenerationProfile(e.target.value, state.generationProfile))}
                                disabled={state.isGenerating}
                            >
                                <option value="deterministic">Deterministic</option>
                                <option value="concise">Concise</option>
                                <option value="detailed">Detailed</option>
                            </select>
                        </div>
                        <div className="panel-control-item">
                            <label htmlFor="clinical-context">Clinical context:</label>
                            <textarea
                                id="clinical-context"
                                value={state.clinicalContext}
                                onChange={(e) => setClinicalContext(e.target.value)}
                                disabled={state.isGenerating || !state.modelAvailability.find(model => model.model_ref === state.selectedModelRef)?.capabilities.clinical_context}
                                placeholder="Not supported by the selected model"
                            />
                        </div>
                    </div>
                </div>

                {/* Center Column - Flow Connector */}
                <div className="flow-panel">
                    <div className="flow-connector">
                        <div className="flow-arrow">
                            <ArrowRight size={24} />
                            <ArrowRight size={24} />
                            <ArrowRight size={24} />
                        </div>

                        <button
                            type="button"
                            className={`btn-generate ${state.isGenerating ? 'generating' : ''}`}
                            onClick={handleGenerateReport}
                            disabled={state.images.length === 0 || state.isGenerating || !state.selectedModelRef}
                        >
                            {state.isGenerating ? (
                                <Loader2 className="loading-spinner" />
                            ) : (
                                <Sparkles />
                            )}
                            <span>
                                {state.isGenerating ? 'Generating...' : 'Generate Report'}
                            </span>
                        </button>

                        <div className="flow-arrow">
                            <ArrowRight size={24} />
                            <ArrowRight size={24} />
                            <ArrowRight size={24} />
                        </div>
                    </div>
                </div>

                {/* Right Column - Report Viewer */}
                <div className="inference-panel">
                    <div className="panel-header">
                        <div className="report-header">
                            <div className="report-header-title">
                                <FileText size={18} />
                                <h2>Generated Report</h2>
                                {state.isGenerating && state.currentStreamingIndex === state.currentIndex && (
                                    <span className="streaming-indicator">
                                        <Loader2 size={14} className="loading-spinner" />
                                        Streaming...
                                    </span>
                                )}
                            </div>
                            <div className="report-header-actions">
                                {state.images.length > 1 && (
                                    <div className="report-nav">
                                        <button
                                            type="button"
                                            className="btn-icon report-nav-btn"
                                            onClick={goToPrevious}
                                            aria-label="Previous report"
                                            title="Previous report"
                                            disabled={state.currentIndex === 0}
                                        >
                                            <ChevronLeft size={16} />
                                        </button>
                                        <span className="report-index">
                                            {state.currentIndex + 1} / {state.images.length}
                                        </span>
                                        <button
                                            type="button"
                                            className="btn-icon report-nav-btn"
                                            onClick={goToNext}
                                            aria-label="Next report"
                                            title="Next report"
                                            disabled={state.currentIndex === state.images.length - 1}
                                        >
                                            <ChevronRight size={16} />
                                        </button>
                                    </div>
                                )}
                                {displayContent && (
                                    <div className="report-actions">
                                        <button
                                            type="button"
                                            className="btn-icon"
                                            onClick={copyReport}
                                            aria-label="Copy report to clipboard"
                                            title="Copy to clipboard"
                                            disabled={state.isGenerating}
                                        >
                                            {state.isCopied ? <Check /> : <Copy />}
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="panel-content">
                        {displayContent ? (
                            <div className="report-content">
                                <div
                                    className={`markdown-report ${state.isGenerating ? 'streaming' : ''}`}
                                    dangerouslySetInnerHTML={{
                                        __html: formatMarkdown(displayContent)
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="report-empty">
                                <FileText />
                                <p>
                                    {state.images.length === 0
                                        ? 'Upload X-ray images to generate a report'
                                        : !state.selectedModelRef
                                            ? 'Select an available model to generate reports'
                                            : 'Click "Generate Report" to analyze the images'}
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

        </div>
    );
}

// Simple markdown formatter (basic implementation)
function formatMarkdown(text: string): string {
    return text
        // Headers
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Horizontal rule
        .replace(/^---$/gm, '<hr />')
        // Unordered list items
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        // Wrap consecutive list items in ul
        .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
        // Line breaks for paragraphs
        .replace(/\n\n/g, '</p><p>')
        // Wrap in paragraph
        .replace(/^(.+)$/gm, (match) => {
            if (match.startsWith('<')) return match;
            return match;
        });
}
