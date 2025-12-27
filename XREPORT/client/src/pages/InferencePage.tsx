import { useRef, DragEvent, ChangeEvent, useEffect, useCallback } from 'react';
import {
    ImagePlus, ChevronLeft, ChevronRight, Trash2, FileText,
    Sparkles, ArrowRight, Copy, Check, Loader2, Settings2
} from 'lucide-react';
import './InferencePage.css';
import { useInferencePageState } from '../AppStateContext';
import { GenerationMode } from '../types';
import {
    getInferenceCheckpoints,
    generateReports,
    connectInferenceWebSocket,
    disconnectInferenceWebSocket,
    InferenceStreamMessage
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
        appendStreamingToken,
        setCurrentStreamingIndex
    } = useInferencePageState();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

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

    // Handle WebSocket messages
    const handleWebSocketMessage = useCallback((message: InferenceStreamMessage) => {
        if (message.type === 'token') {
            const imgIdx = message.image_index ?? 0;
            if (imgIdx === state.currentStreamingIndex) {
                appendStreamingToken(message.token ?? '');
            }
        } else if (message.type === 'start') {
            setStreamingTokens('');
            setCurrentStreamingIndex(0);
        } else if (message.type === 'complete' && message.reports) {
            // Store all reports by index
            const reportsByIndex: Record<number, string> = {};
            Object.values(message.reports).forEach((report, idx) => {
                reportsByIndex[idx] = report;
            });
            setReports(reportsByIndex);
            setIsGenerating(false);
            setCurrentStreamingIndex(-1);
            // Set the generated report for the current index
            if (reportsByIndex[state.currentIndex]) {
                setGeneratedReport(reportsByIndex[state.currentIndex]);
            }
        } else if (message.type === 'error') {
            console.error('Generation error:', message.message);
            setIsGenerating(false);
            setCurrentStreamingIndex(-1);
        }
    }, [state.currentStreamingIndex, state.currentIndex, appendStreamingToken, setStreamingTokens, setCurrentStreamingIndex, setReports, setIsGenerating, setGeneratedReport]);

    // Handle file selection
    const handleFileSelect = (files: FileList | null) => {
        if (!files || files.length === 0) return;

        const imageFiles = Array.from(files).filter(file =>
            file.type.startsWith('image/')
        );

        // Enforce max 16 images
        const limitedImages = imageFiles.slice(0, MAX_IMAGES);

        if (limitedImages.length > 0) {
            setImages(limitedImages);
            setCurrentIndex(0);
            setGeneratedReport('');
            setReports({});
            setStreamingTokens('');
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
    const goToPrevious = () => {
        const newIndex = Math.max(0, state.currentIndex - 1);
        setCurrentIndex(newIndex);
        // Update displayed report for new index
        if (state.reports[newIndex]) {
            setGeneratedReport(state.reports[newIndex]);
        } else {
            setGeneratedReport('');
        }
    };

    const goToNext = () => {
        const newIndex = Math.min(state.images.length - 1, state.currentIndex + 1);
        setCurrentIndex(newIndex);
        // Update displayed report for new index
        if (state.reports[newIndex]) {
            setGeneratedReport(state.reports[newIndex]);
        } else {
            setGeneratedReport('');
        }
    };

    // Clear images
    const handleClearImages = () => {
        clearImages();
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Generate reports
    const handleGenerateReport = async () => {
        if (state.images.length === 0 || !state.selectedCheckpoint) return;

        setIsGenerating(true);
        setStreamingTokens('');
        setReports({});
        setCurrentStreamingIndex(0);

        // Connect WebSocket for streaming
        wsRef.current = connectInferenceWebSocket(
            handleWebSocketMessage,
            () => setIsGenerating(false),
            () => setIsGenerating(false)
        );

        // Call generate endpoint
        const { result: _result, error } = await generateReports(
            state.images,
            state.selectedCheckpoint,
            state.generationMode
        );

        if (error) {
            console.error('Generation failed:', error);
            setIsGenerating(false);
        }

        // Clean up WebSocket after a delay
        setTimeout(() => {
            disconnectInferenceWebSocket(wsRef.current);
            wsRef.current = null;
        }, 1000);
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
        : state.generatedReport || state.reports[state.currentIndex] || '';

    return (
        <div className="inference-container">
            {/* Control Panel */}
            <div className="inference-control-panel">
                <div className="control-panel-content">
                    <div className="control-panel-item">
                        <Settings2 size={16} />
                        <span className="control-label">Settings</span>
                    </div>
                    <div className="control-panel-item">
                        <label htmlFor="checkpoint-select">Checkpoint:</label>
                        <select
                            id="checkpoint-select"
                            value={state.selectedCheckpoint}
                            onChange={(e) => setSelectedCheckpoint(e.target.value)}
                            disabled={state.isLoadingCheckpoints || state.isGenerating}
                        >
                            {state.checkpoints.length === 0 && (
                                <option value="">No checkpoints available</option>
                            )}
                            {state.checkpoints.map((cp) => (
                                <option key={cp} value={cp}>{cp}</option>
                            ))}
                        </select>
                    </div>
                    <div className="control-panel-item">
                        <label htmlFor="mode-select">Mode:</label>
                        <select
                            id="mode-select"
                            value={state.generationMode}
                            onChange={(e) => setGenerationMode(e.target.value as GenerationMode)}
                            disabled={state.isGenerating}
                        >
                            <option value="greedy_search">Greedy Search</option>
                            <option value="beam_search">Beam Search</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="inference-header">
                <h1>Inference</h1>
                <p>Generate radiological reports from X-ray scans using AI</p>
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
                                        onClick={handleClearImages}
                                        title="Clear all images"
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
