import { useRef, DragEvent, ChangeEvent, useEffect } from 'react';
import {
    ImagePlus, ChevronLeft, ChevronRight, Trash, Trash2, FileText,
    Sparkles, ArrowRight, Copy, Check, Loader2
} from 'lucide-react';
import './InferencePage.css';
import { useInferencePageState } from '../AppStateContext';
import { GenerationMode } from '../types';
import {
    getInferenceCheckpoints,
    generateReports,
    getInferenceJobStatus
} from '../services/inferenceService';

const MAX_IMAGES = 16;

export default function InferencePage() {
    const {
        state,
        setImages,
        setCurrentIndex,
        setGeneratedReport,
        setIsGenerating,
        setIsCopied,
        clearImages,
        setSelectedCheckpoint,
        setGenerationMode,
        setCheckpoints,
        setIsLoadingCheckpoints,
        setReports,
        setStreamingTokens,
        setCurrentStreamingIndex,
    } = useInferencePageState();

    const fileInputRef = useRef<HTMLInputElement>(null);
    // Fetch checkpoints on mount
    useEffect(() => {
        const fetchCheckpoints = async () => {
            setIsLoadingCheckpoints(true);
            const { result, error } = await getInferenceCheckpoints();
            if (result && result.success) {
                const names = result.checkpoints.map(cp => cp.name);
                setCheckpoints(names);
                if (names.length > 0 && !state.selectedCheckpoint) {
                    setSelectedCheckpoint(names[0]);
                }
            } else if (error) {
                console.error('Failed to fetch checkpoints:', error);
            }
            setIsLoadingCheckpoints(false);
        };
        fetchCheckpoints();
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
        if (state.images.length === 0 || !state.selectedCheckpoint) return;

        setIsGenerating(true);
        setStreamingTokens('');
        setReports({});
        setGeneratedReport('');
        setCurrentStreamingIndex(-1);

        // Call generate endpoint - now returns job_id
        const { result: jobResult, error: startError } = await generateReports(
            state.images,
            state.selectedCheckpoint,
            state.generationMode
        );

        if (startError || !jobResult) {
            console.error('Generation failed:', startError);
            setIsGenerating(false);
            return;
        }

        // Poll for job completion
        const pollInterval = 2000;
        const poll = async () => {
            const { result: status, error: pollError } = await getInferenceJobStatus(jobResult.job_id);

            if (pollError) {
                console.error('Poll error:', pollError);
                setIsGenerating(false);
                return;
            }

            if (!status) {
                setIsGenerating(false);
                return;
            }

            if (status.result) {
                const resultPayload = status.result as Record<string, unknown>;
                const reports = resultPayload.reports as Record<string, string> | undefined;
                const reportsOrdered = resultPayload.reports_ordered as string[] | undefined;
                const reportFilenames = resultPayload.report_filenames as string[] | undefined;

                const reportsByIndex: Record<number, string> = {};

                if (reportsOrdered && reportsOrdered.length > 0) {
                    reportsOrdered.forEach((report, idx) => {
                        if (report) {
                            reportsByIndex[idx] = report;
                        }
                    });
                } else if (reports) {
                    if (reportFilenames && reportFilenames.length > 0) {
                        reportFilenames.forEach((filename, idx) => {
                            const report = reports[filename];
                            if (report !== undefined) {
                                reportsByIndex[idx] = report;
                            }
                        });
                    } else {
                        state.images.forEach((image, idx) => {
                            const report = reports[image.name];
                            if (report !== undefined) {
                                reportsByIndex[idx] = report;
                            }
                        });
                    }

                    if (Object.keys(reportsByIndex).length === 0) {
                        Object.values(reports).forEach((report, idx) => {
                            reportsByIndex[idx] = report;
                        });
                    }
                }

                if (Object.keys(reportsByIndex).length > 0) {
                    setReports(reportsByIndex);
                    if (reportsByIndex[state.currentIndex] !== undefined) {
                        setGeneratedReport(reportsByIndex[state.currentIndex]);
                    }
                }
            }

            if (status.status === 'completed') {
                setIsGenerating(false);
            } else if (status.status === 'failed') {
                console.error('Generation failed:', status.error);
                setIsGenerating(false);
            } else if (status.status === 'cancelled') {
                setIsGenerating(false);
            } else {
                // Still running, poll again
                setTimeout(poll, pollInterval);
            }
        };
        poll();
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
    const currentImageUrl = state.images.length > 0
        ? URL.createObjectURL(state.images[state.currentIndex])
        : null;

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
                    Simply upload your medical images, verify the selected model checkpoint in the settings bar,
                    and click 'Generate Report' to obtain a structured analysis of the findings.
                </p>
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
                                    style={{ display: 'none' }}
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
                                                className="nav-arrow prev"
                                                onClick={goToPrevious}
                                                disabled={state.currentIndex === 0}
                                            >
                                                <ChevronLeft size={20} />
                                            </button>
                                            <button
                                                className="nav-arrow next"
                                                onClick={goToNext}
                                                disabled={state.currentIndex === state.images.length - 1}
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
                                        className="btn-icon"
                                        onClick={openFileDialog}
                                        title="Add more images"
                                        disabled={state.images.length >= MAX_IMAGES}
                                    >
                                        <ImagePlus />
                                    </button>
                                    <button
                                        className="btn-icon"
                                        onClick={handleRemoveCurrentImage}
                                        title="Remove current image"
                                        disabled={state.images.length === 0 || state.isGenerating}
                                    >
                                        <Trash />
                                    </button>
                                    <button
                                        className="btn-icon"
                                        onClick={handleClearImages}
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
                                    style={{ display: 'none' }}
                                />
                            </div>
                        )}
                    </div>
                    <div className="panel-footer">
                        <div className="panel-control-item">
                            <label htmlFor="checkpoint-select">Checkpoint:</label>
                            <select
                                id="checkpoint-select"
                                value={state.selectedCheckpoint}
                                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                disabled={state.isLoadingCheckpoints || state.isGenerating}
                            >
                                {state.checkpoints.length === 0 && (
                                    <option value="">No checkpoints</option>
                                )}
                                {state.checkpoints.map((cp) => (
                                    <option key={cp} value={cp}>{cp}</option>
                                ))}
                            </select>
                        </div>
                        <div className="panel-control-item">
                            <label htmlFor="mode-select">Mode:</label>
                            <select
                                id="mode-select"
                                value={state.generationMode}
                                onChange={(e) => setGenerationMode(e.target.value as GenerationMode)}
                                disabled={state.isGenerating}
                            >
                                <option value="greedy_search">Greedy</option>
                                <option value="beam_search">Beam</option>
                            </select>
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
                            className={`btn-generate ${state.isGenerating ? 'generating' : ''}`}
                            onClick={handleGenerateReport}
                            disabled={state.images.length === 0 || state.isGenerating || !state.selectedCheckpoint}
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
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
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
                                            className="btn-icon report-nav-btn"
                                            onClick={goToPrevious}
                                            title="Previous report"
                                            disabled={state.currentIndex === 0}
                                        >
                                            <ChevronLeft size={16} />
                                        </button>
                                        <span className="report-index">
                                            {state.currentIndex + 1} / {state.images.length}
                                        </span>
                                        <button
                                            className="btn-icon report-nav-btn"
                                            onClick={goToNext}
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
                                            className="btn-icon"
                                            onClick={copyReport}
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
                                        : !state.selectedCheckpoint
                                            ? 'Select a checkpoint to generate reports'
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
