import { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import {
    DatasetPageState,
    DatasetProcessingConfig,
    TrainingPageState,
    TrainingConfig,
    InferencePageState,
    DatabaseBrowserPageState
} from './types';
import { ImagePathResponse, DatasetUploadResponse, LoadDatasetResponse, ProcessDatasetResponse, DatasetStatusResponse, DatasetNamesResponse } from './services/trainingService';
import { ValidationResponse } from './services/validationService';
import { TableInfo } from './services/databaseBrowser';

// ============================================================================
// Default States
// ============================================================================
const DEFAULT_DATASET_CONFIG: DatasetProcessingConfig = {
    sampleSize: 1.0,
    validationSize: 0.2,
    maxReportSize: 200,
    tokenizer: 'distilbert-base-uncased',
    imgStats: false,
    textStats: false,
    pixDist: false
};

const DEFAULT_DATASET_STATE: DatasetPageState = {
    config: DEFAULT_DATASET_CONFIG,
    imageFolderPath: '',
    imageFolderName: '',
    imageValidation: null,
    datasetFile: null,
    datasetUpload: null,
    loadResult: null,
    isLoading: false,
    uploadError: null,
    folderBrowserOpen: false,
    isProcessing: false,
    processingResult: null,
    dbStatus: null,
    datasetNames: null,
    selectedDataset: '',
    isValidating: false,
    validationResult: null,
    validationError: null
};

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
    numEncoders: 6,
    numDecoders: 6,
    embeddingDims: 768,
    attnHeads: 8,
    freezeImgEncoder: true,
    trainTemp: 1.0,
    useImgAugment: false,
    shuffleWithBuffer: true,
    shuffleBufferSize: 256,
    epochs: 100,
    batchSize: 32,
    trainSeed: 42,
    saveCheckpoints: true,
    checkpointFreq: 1,
    mixedPrecision: false,
    runTensorboard: false,
    useScheduler: false,
    targetLR: 0.001,
    warmupSteps: 1000,
    realTimePlot: true,
    useGpu: true,
    gpuId: 0
};

const DEFAULT_TRAINING_STATE: TrainingPageState = {
    config: DEFAULT_TRAINING_CONFIG,
    newSessionExpanded: true,
    resumeSessionExpanded: false,
    selectedCheckpoint: '',
    additionalEpochs: 50
};

const DEFAULT_INFERENCE_STATE: InferencePageState = {
    images: [],
    currentIndex: 0,
    generatedReport: '',
    isGenerating: false,
    isCopied: false
};

const DEFAULT_BATCH_SIZE = 200;

const DEFAULT_DATABASE_BROWSER_STATE: DatabaseBrowserPageState = {
    tables: [],
    selectedTable: '',
    rows: [],
    columns: [],
    rowCount: 0,
    columnCount: 0,
    displayName: '',
    limit: DEFAULT_BATCH_SIZE,
    offset: 0,
    loading: false,
    error: null,
    tablesLoaded: false,
    dataLoaded: false
};

// ============================================================================
// Context Types
// ============================================================================
interface AppStateContextType {
    // Dataset Page
    datasetPageState: DatasetPageState;
    setDatasetPageState: (state: DatasetPageState | ((prev: DatasetPageState) => DatasetPageState)) => void;

    // Training Page
    trainingPageState: TrainingPageState;
    setTrainingPageState: (state: TrainingPageState | ((prev: TrainingPageState) => TrainingPageState)) => void;

    // Inference Page
    inferencePageState: InferencePageState;
    setInferencePageState: (state: InferencePageState | ((prev: InferencePageState) => InferencePageState)) => void;

    // Database Browser Page
    databaseBrowserPageState: DatabaseBrowserPageState;
    setDatabaseBrowserPageState: (state: DatabaseBrowserPageState | ((prev: DatabaseBrowserPageState) => DatabaseBrowserPageState)) => void;
}

const AppStateContext = createContext<AppStateContextType | null>(null);

// ============================================================================
// Provider Component
// ============================================================================
export function AppStateProvider({ children }: { children: ReactNode }) {
    const [datasetPageState, setDatasetPageState] = useState<DatasetPageState>(DEFAULT_DATASET_STATE);
    const [trainingPageState, setTrainingPageState] = useState<TrainingPageState>(DEFAULT_TRAINING_STATE);
    const [inferencePageState, setInferencePageState] = useState<InferencePageState>(DEFAULT_INFERENCE_STATE);
    const [databaseBrowserPageState, setDatabaseBrowserPageState] = useState<DatabaseBrowserPageState>(DEFAULT_DATABASE_BROWSER_STATE);

    return (
        <AppStateContext.Provider
            value={{
                datasetPageState,
                setDatasetPageState,
                trainingPageState,
                setTrainingPageState,
                inferencePageState,
                setInferencePageState,
                databaseBrowserPageState,
                setDatabaseBrowserPageState
            }}
        >
            {children}
        </AppStateContext.Provider>
    );
}

// ============================================================================
// Custom Hooks
// ============================================================================
function useAppState() {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error('useAppState must be used within an AppStateProvider');
    }
    return context;
}

export function useDatasetPageState() {
    const { datasetPageState, setDatasetPageState } = useAppState();

    const updateConfig = useCallback((key: keyof DatasetProcessingConfig, value: DatasetProcessingConfig[keyof DatasetProcessingConfig]) => {
        setDatasetPageState(prev => ({
            ...prev,
            config: { ...prev.config, [key]: value }
        }));
    }, [setDatasetPageState]);

    const setImageFolderPath = useCallback((path: string) => {
        setDatasetPageState(prev => ({ ...prev, imageFolderPath: path }));
    }, [setDatasetPageState]);

    const setImageFolderName = useCallback((name: string) => {
        setDatasetPageState(prev => ({ ...prev, imageFolderName: name }));
    }, [setDatasetPageState]);

    const setImageValidation = useCallback((validation: ImagePathResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, imageValidation: validation }));
    }, [setDatasetPageState]);

    const setDatasetFile = useCallback((file: File | null) => {
        setDatasetPageState(prev => ({ ...prev, datasetFile: file }));
    }, [setDatasetPageState]);

    const setDatasetUpload = useCallback((upload: DatasetUploadResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, datasetUpload: upload }));
    }, [setDatasetPageState]);

    const setLoadResult = useCallback((result: LoadDatasetResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, loadResult: result }));
    }, [setDatasetPageState]);

    const setIsLoading = useCallback((loading: boolean) => {
        setDatasetPageState(prev => ({ ...prev, isLoading: loading }));
    }, [setDatasetPageState]);

    const setUploadError = useCallback((error: string | null) => {
        setDatasetPageState(prev => ({ ...prev, uploadError: error }));
    }, [setDatasetPageState]);

    const setFolderBrowserOpen = useCallback((open: boolean) => {
        setDatasetPageState(prev => ({ ...prev, folderBrowserOpen: open }));
    }, [setDatasetPageState]);

    const setIsProcessing = useCallback((processing: boolean) => {
        setDatasetPageState(prev => ({ ...prev, isProcessing: processing }));
    }, [setDatasetPageState]);

    const setProcessingResult = useCallback((result: ProcessDatasetResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, processingResult: result }));
    }, [setDatasetPageState]);

    const setDbStatus = useCallback((status: DatasetStatusResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, dbStatus: status }));
    }, [setDatasetPageState]);

    const setDatasetNames = useCallback((names: DatasetNamesResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, datasetNames: names }));
    }, [setDatasetPageState]);

    const setSelectedDataset = useCallback((dataset: string) => {
        setDatasetPageState(prev => ({ ...prev, selectedDataset: dataset }));
    }, [setDatasetPageState]);

    const setIsValidating = useCallback((validating: boolean) => {
        setDatasetPageState(prev => ({ ...prev, isValidating: validating }));
    }, [setDatasetPageState]);

    const setValidationResult = useCallback((result: ValidationResponse | null) => {
        setDatasetPageState(prev => ({ ...prev, validationResult: result }));
    }, [setDatasetPageState]);

    const setValidationError = useCallback((error: string | null) => {
        setDatasetPageState(prev => ({ ...prev, validationError: error }));
    }, [setDatasetPageState]);

    return {
        state: datasetPageState,
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
        setSelectedDataset,
        setIsValidating,
        setValidationResult,
        setValidationError
    };
}

export function useTrainingPageState() {
    const { trainingPageState, setTrainingPageState } = useAppState();

    const updateConfig = useCallback((key: keyof TrainingConfig, value: TrainingConfig[keyof TrainingConfig]) => {
        setTrainingPageState(prev => ({
            ...prev,
            config: { ...prev.config, [key]: value }
        }));
    }, [setTrainingPageState]);

    const setNewSessionExpanded = useCallback((expanded: boolean) => {
        setTrainingPageState(prev => ({ ...prev, newSessionExpanded: expanded }));
    }, [setTrainingPageState]);

    const setResumeSessionExpanded = useCallback((expanded: boolean) => {
        setTrainingPageState(prev => ({ ...prev, resumeSessionExpanded: expanded }));
    }, [setTrainingPageState]);

    const setSelectedCheckpoint = useCallback((checkpoint: string) => {
        setTrainingPageState(prev => ({ ...prev, selectedCheckpoint: checkpoint }));
    }, [setTrainingPageState]);

    const setAdditionalEpochs = useCallback((epochs: number) => {
        setTrainingPageState(prev => ({ ...prev, additionalEpochs: epochs }));
    }, [setTrainingPageState]);

    return {
        state: trainingPageState,
        updateConfig,
        setNewSessionExpanded,
        setResumeSessionExpanded,
        setSelectedCheckpoint,
        setAdditionalEpochs
    };
}

export function useInferencePageState() {
    const { inferencePageState, setInferencePageState } = useAppState();

    const setImages = useCallback((images: File[]) => {
        setInferencePageState(prev => ({ ...prev, images }));
    }, [setInferencePageState]);

    const setCurrentIndex = useCallback((index: number) => {
        setInferencePageState(prev => ({ ...prev, currentIndex: index }));
    }, [setInferencePageState]);

    const setGeneratedReport = useCallback((report: string) => {
        setInferencePageState(prev => ({ ...prev, generatedReport: report }));
    }, [setInferencePageState]);

    const setIsGenerating = useCallback((generating: boolean) => {
        setInferencePageState(prev => ({ ...prev, isGenerating: generating }));
    }, [setInferencePageState]);

    const setIsCopied = useCallback((copied: boolean) => {
        setInferencePageState(prev => ({ ...prev, isCopied: copied }));
    }, [setInferencePageState]);

    const clearImages = useCallback(() => {
        setInferencePageState(prev => ({
            ...prev,
            images: [],
            currentIndex: 0,
            generatedReport: ''
        }));
    }, [setInferencePageState]);

    return {
        state: inferencePageState,
        setImages,
        setCurrentIndex,
        setGeneratedReport,
        setIsGenerating,
        setIsCopied,
        clearImages
    };
}

export function useDatabaseBrowserState() {
    const { databaseBrowserPageState, setDatabaseBrowserPageState } = useAppState();

    const setState = useCallback((updater: Partial<DatabaseBrowserPageState> | ((prev: DatabaseBrowserPageState) => DatabaseBrowserPageState)) => {
        if (typeof updater === 'function') {
            setDatabaseBrowserPageState(updater);
        } else {
            setDatabaseBrowserPageState(prev => ({ ...prev, ...updater }));
        }
    }, [setDatabaseBrowserPageState]);

    const setTables = useCallback((tables: TableInfo[]) => {
        setDatabaseBrowserPageState(prev => ({ ...prev, tables }));
    }, [setDatabaseBrowserPageState]);

    const setSelectedTable = useCallback((table: string) => {
        setDatabaseBrowserPageState(prev => ({ ...prev, selectedTable: table }));
    }, [setDatabaseBrowserPageState]);

    const setLoading = useCallback((loading: boolean) => {
        setDatabaseBrowserPageState(prev => ({ ...prev, loading }));
    }, [setDatabaseBrowserPageState]);

    const setError = useCallback((error: string | null) => {
        setDatabaseBrowserPageState(prev => ({ ...prev, error }));
    }, [setDatabaseBrowserPageState]);

    return {
        state: databaseBrowserPageState,
        setState,
        setTables,
        setSelectedTable,
        setLoading,
        setError
    };
}
