import type { JobLifecycleStatus } from './types/jobs';
import {
    DatasetNamesResponse,
    DatasetStatusResponse,
    DatasetUploadResponse,
    ImagePathResponse,
    LoadDatasetResponse,
    ProcessDatasetResponse,
} from './types/trainingApi';
import { ValidationResponse } from './types/validationApi';
import type { DraftSections } from './common/report-sections';
import type { GenerationProfile, InferenceHistoryDetail, ModelAvailability } from './types/inferenceApi';

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
    validationResult: ValidationResponse | null;
    validationError: string | null;
}

// ============================================================================
// Training Page State
// ============================================================================
export interface TrainingConfig {
    architecturePreset: string;
    useImgAugment: boolean;
    shuffleWithBuffer: boolean;
    shuffleBufferSize: number;
    epochs: number;
    batchSize: number;
    saveCheckpoints: boolean;
    checkpointFreq: number;
    useScheduler: boolean;
    targetLR: number;
    warmupSteps: number;
    realTimePlot: boolean;
    dataloaderWorkers: number;
    prefetchFactor: number;
    pinMemory: boolean;
    persistentWorkers: boolean;
    useMixedPrecision: boolean;
    jitCompile: boolean;
    jitBackend: string;
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

// ============================================================================
// Inference Page State
// ============================================================================
// One model's working study. The active model's study lives inline in InferencePageState;
// the studies of the other models are kept aside so switching back restores them unchanged.
export interface InferenceStudy {
    images: File[];
    currentIndex: number;
    generatedReport: string;
    // 'selected' runs the previewed image, 'all' runs every image independently, 'study' sends all images as one study
    runScope: 'selected' | 'all' | 'study';
    // Uploaded-image indices submitted by the active or latest run, in submission order
    runIndices: number[];
    // Whether the latest run produced one study-level report shared by every submitted image
    studyReport: boolean;
    // Saved history request of the latest completed run
    requestId: string | null;
    generationProfile: GenerationProfile;
    clinicalContext: string;
    // Per-image reports for synchronized navigation
    reports: Record<number, string>;
    // Editable report sections by image index, and the saved sections they are compared against
    drafts: Record<number, DraftSections>;
    baselineDrafts: Record<number, DraftSections>;
    persistedHistory: InferenceHistoryDetail | null;
}

export interface InferencePageState extends InferenceStudy {
    isGenerating: boolean;
    // Survives page navigation so the inference page can resume tracking the running job
    activeJobId: string | null;
    isCopied: boolean;
    selectedModelRef: string;
    modelAvailability: ModelAvailability[];
    isLoadingModels: boolean;
    // Streaming state
    streamingTokens: string;
    currentStreamingIndex: number;
}

