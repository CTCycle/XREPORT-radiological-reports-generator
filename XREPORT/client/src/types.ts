import { ImagePathResponse, DatasetUploadResponse, LoadDatasetResponse, ProcessDatasetResponse, DatasetStatusResponse, DatasetNamesResponse } from './services/trainingService';
import { ValidationResponse } from './services/validationService';
import { TableInfo } from './services/databaseBrowser';

// ============================================================================
// Dataset Page State
// ============================================================================
export interface DatasetProcessingConfig {
    sampleSize: number;
    validationSize: number;
    maxReportSize: number;
    tokenizer: string;
    // Evaluation settings (separate from processing)
    evalSampleSize: number;
    imgStats: boolean;
    textStats: boolean;
    pixDist: boolean;
}

export interface DatasetPageState {
    config: DatasetProcessingConfig;
    imageFolderPath: string;
    imageFolderName: string;
    imageValidation: ImagePathResponse | null;
    datasetFile: File | null;
    datasetUpload: DatasetUploadResponse | null;
    loadResult: LoadDatasetResponse | null;
    isLoading: boolean;
    uploadError: string | null;
    folderBrowserOpen: boolean;
    isProcessing: boolean;
    processingResult: ProcessDatasetResponse | null;
    dbStatus: DatasetStatusResponse | null;
    datasetNames: DatasetNamesResponse | null;
    selectedDataset: string;
    // Validation state
    isValidating: boolean;
    validationResult: ValidationResponse | null;
    validationError: string | null;
}

// ============================================================================
// Training Page State
// ============================================================================
export interface TrainingConfig {
    numEncoders: number;
    numDecoders: number;
    embeddingDims: number;
    attnHeads: number;
    freezeImgEncoder: boolean;
    trainTemp: number;
    useImgAugment: boolean;
    shuffleWithBuffer: boolean;
    shuffleBufferSize: number;
    epochs: number;
    batchSize: number;
    trainSeed: number;
    saveCheckpoints: boolean;
    checkpointFreq: number;
    mixedPrecision: boolean;
    runTensorboard: boolean;
    useScheduler: boolean;
    targetLR: number;
    warmupSteps: number;
    realTimePlot: boolean;
    useGpu: boolean;
    gpuId: number;
}

export interface ChartDataPoint {
    batch: number;
    loss?: number;
    val_loss?: number;
    MaskedAccuracy?: number;
    val_MaskedAccuracy?: number;
    [key: string]: number | undefined;
}

export interface TrainingDashboardState {
    isTraining: boolean;
    currentEpoch: number;
    totalEpochs: number;
    loss: number;
    valLoss: number;
    accuracy: number;
    valAccuracy: number;
    progressPercent: number;
    elapsedSeconds: number;
    chartData: ChartDataPoint[];
    availableMetrics: string[];
    epochBoundaries: number[];
    shouldConnectWs: boolean;
}

export interface TrainingPageState {
    config: TrainingConfig;
    newSessionExpanded: boolean;
    resumeSessionExpanded: boolean;
    selectedCheckpoint: string;
    additionalEpochs: number;
    dashboardState: TrainingDashboardState;
}

// ============================================================================
// Inference Page State
// ============================================================================
export type GenerationMode = 'greedy_search' | 'beam_search';

export interface InferencePageState {
    images: File[];
    currentIndex: number;
    generatedReport: string;
    isGenerating: boolean;
    isCopied: boolean;
    // Checkpoint and generation settings
    selectedCheckpoint: string;
    generationMode: GenerationMode;
    checkpoints: string[];
    isLoadingCheckpoints: boolean;
    // Per-image reports for synchronized navigation
    reports: Record<number, string>;
    // Streaming state
    streamingTokens: string;
    currentStreamingIndex: number;
}

// ============================================================================
// Database Browser Page State
// ============================================================================
export interface DatabaseBrowserPageState {
    tables: TableInfo[];
    selectedTable: string;
    rows: Record<string, unknown>[];
    columns: string[];
    rowCount: number;
    columnCount: number;
    displayName: string;
    limit: number;
    offset: number;
    loading: boolean;
    loadingMore: boolean;
    hasMore: boolean;
    error: string | null;
    tablesLoaded: boolean;
    dataLoaded: boolean;
}
