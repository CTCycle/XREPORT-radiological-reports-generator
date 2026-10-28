import type { JobLifecycleStatus } from '@xreport/shared/app/types/jobs';
import type {
    DatasetNamesResponse,
    DatasetStatusResponse,
    DatasetUploadResponse,
    ImagePathResponse,
    LoadDatasetResponse,
    ProcessDatasetResponse,
} from './trainingApi';

// ============================================================================
// Dataset Page State
// ============================================================================
export interface DatasetProcessingConfig {
    datasetName?: string;
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
    isProcessing: boolean;
    processingResult: ProcessDatasetResponse | null;
    dbStatus: DatasetStatusResponse | null;
    datasetNames: DatasetNamesResponse | null;
    selectedDatasets: string[];
    // Validation state
    isValidating: boolean;
    validationError: string | null;
}

// ============================================================================
// Training Page State
// ============================================================================
export interface TrainingConfig {
    architecturePreset: string;
    useImgAugment: boolean;
    epochs: number;
    batchSize: number;
    gradientAccumulationSteps: number;
    saveCheckpoints: boolean;
    useScheduler: boolean;
    targetLR: number;
    warmupSteps: number;
    realTimePlot: boolean;
    dataloaderWorkers: number;
    prefetchFactor: number;
    pinMemory: boolean;
    persistentWorkers: boolean;
    useMixedPrecision: boolean;
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
    sessionStatus?: JobLifecycleStatus;
    currentEpoch: number;
    totalEpochs: number;
    loss: number;
    valLoss: number;
    accuracy: number;
    valAccuracy: number;
    progressPercent: number;
    elapsedSeconds: number;
    workerPhase?: string;
    workerPhaseStatus?: string;
    workerPhaseElapsedSeconds?: number;
    workerElapsedSeconds?: number;
    chartData: ChartDataPoint[];
    availableMetrics: string[];
    epochBoundaries: number[];
    logEntries: string[];
}

export interface TrainingPageState {
    config: TrainingConfig;
    newSessionExpanded: boolean;
    resumeSessionExpanded: boolean;
    selectedCheckpoint: string;
    additionalEpochs: number;
    dashboardState: TrainingDashboardState;
}
