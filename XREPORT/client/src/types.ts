import { ImagePathResponse, DatasetUploadResponse, LoadDatasetResponse, ProcessDatasetResponse, DatasetStatusResponse } from './services/trainingService';
import { TableInfo } from './services/databaseBrowser';

// ============================================================================
// Dataset Page State
// ============================================================================
export interface DatasetProcessingConfig {
    seed: number;
    sampleSize: number;
    validationSize: number;
    splitSeed: number;
    maxReportSize: number;
    tokenizer: string;
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
    useJIT: boolean;
    jitBackend: string;
    useScheduler: boolean;
    targetLR: number;
    warmupSteps: number;
    realTimePlot: boolean;
    useGpu: boolean;
    gpuId: number;
}

export interface TrainingPageState {
    config: TrainingConfig;
    newSessionExpanded: boolean;
    resumeSessionExpanded: boolean;
    selectedCheckpoint: string;
    additionalEpochs: number;
}

// ============================================================================
// Inference Page State
// ============================================================================
export interface InferencePageState {
    images: File[];
    currentIndex: number;
    generatedReport: string;
    isGenerating: boolean;
    isCopied: boolean;
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
    error: string | null;
    tablesLoaded: boolean;
    dataLoaded: boolean;
}
