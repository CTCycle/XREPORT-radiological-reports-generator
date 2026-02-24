import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Info,
    Play,
    RefreshCw,
    RotateCcw,
    Trash2,
    Activity,
    BarChart2,
} from 'lucide-react';
import './TrainingPage.css';
import { useTrainingPageState } from '../AppStateContext';
import TrainingDashboard from '../components/TrainingDashboard';
import MetadataModal, {
    MetadataModalState,
    PROCESSING_METADATA_ORDER,
    buildEntries,
    parseMetadataError,
} from '../components/MetadataModal';
import NewTrainingWizard from '../components/NewTrainingWizard';
import ResumeTrainingWizard from '../components/ResumeTrainingWizard';
import EvaluationWizard from '../components/EvaluationWizard';
import CheckpointEvaluationReportModal from '../components/CheckpointEvaluationReportModal';
import { ChartDataPoint, TrainingConfig } from '../types';
import { usePersistedRecord } from '../hooks/usePersistedRecord';
import {
    CheckpointInfo,
    DatasetInfo,
    JobStatusResponse,
    StartTrainingConfig,
    TrainingStatusResponse,
    deleteCheckpoint,
    deleteDataset,
    getCheckpointMetadata,
    getCheckpoints,
    getProcessingMetadata,
    getProcessedDatasetNames,
    getTrainingJobStatus,
    getTrainingStatus,
    pollJobStatus,
    resumeTraining,
    startTraining,
    stopTraining,
} from '../services/trainingService';
import {
    CheckpointEvaluationReport,
    evaluateCheckpoint,
    getCheckpointEvaluationReport,
    pollCheckpointEvaluationJobStatus,
} from '../services/inferenceService';

const EVALUATION_JOB_STORAGE_KEY = 'xreport.checkpoint_evaluation.jobs';

type StoredEvaluationJob = {
    jobId: string;
    metrics: string[];
    metricConfigs: Record<string, { dataFraction: number }>;
    pollIntervalMs?: number;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress?: number;
};

type ParsedTrainingJobResult = {
    currentEpoch?: number;
    totalEpochs?: number;
    loss?: number;
    valLoss?: number;
    accuracy?: number;
    valAccuracy?: number;
    progressPercent?: number;
    elapsedSeconds?: number;
    chartData?: ChartDataPoint[];
    availableMetrics?: string[];
    epochBoundaries?: number[];
};

function toApiMetricConfigs(metricConfigs: Record<string, { dataFraction: number }>) {
    return Object.fromEntries(
        Object.entries(metricConfigs).map(([key, value]) => [
            key,
            { data_fraction: value.dataFraction },
        ])
    );
}

function toCheckpointEvaluationResults(statusResult: Record<string, unknown> | null) {
    const jobResults = statusResult ?? {};
    const metricsResult = (jobResults.results as Record<string, unknown> | null) ?? {};
    return {
        loss: metricsResult.loss as number | undefined,
        accuracy: metricsResult.accuracy as number | undefined,
        bleu_score: metricsResult.bleu_score as number | undefined,
    };
}

function toCheckpointEvaluationReport(
    checkpoint: string,
    metrics: string[],
    metricConfigs: Record<string, { dataFraction: number }>,
    statusResult: Record<string, unknown> | null
): CheckpointEvaluationReport {
    return {
        checkpoint,
        metrics,
        metric_configs: toApiMetricConfigs(metricConfigs),
        results: toCheckpointEvaluationResults(statusResult),
    };
}

function readNumber(value: unknown): number | undefined {
    return typeof value === 'number' ? value : undefined;
}

function readStringArray(value: unknown): string[] | undefined {
    if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'string')) {
        return undefined;
    }
    return value;
}

function readNumberArray(value: unknown): number[] | undefined {
    if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'number')) {
        return undefined;
    }
    return value;
}

function readChartDataArray(value: unknown): ChartDataPoint[] | undefined {
    if (!Array.isArray(value)) {
        return undefined;
    }
    return value as ChartDataPoint[];
}

function parseTrainingJobResult(
    result: Record<string, unknown>,
    fallbackProgress?: number
): ParsedTrainingJobResult {
    const parsed: ParsedTrainingJobResult = {
        currentEpoch: readNumber(result.current_epoch),
        totalEpochs: readNumber(result.total_epochs),
        loss: readNumber(result.loss),
        valLoss: readNumber(result.val_loss),
        accuracy: readNumber(result.accuracy),
        valAccuracy: readNumber(result.val_accuracy),
        progressPercent: readNumber(result.progress_percent) ?? fallbackProgress,
        elapsedSeconds: readNumber(result.elapsed_seconds),
    };

    const chartData = readChartDataArray(result.chart_data);
    if (chartData) {
        parsed.chartData = chartData;
    }

    const availableMetrics = readStringArray(result.available_metrics);
    if (availableMetrics) {
        parsed.availableMetrics = availableMetrics;
    }

    const epochBoundaries = readNumberArray(result.epoch_boundaries);
    if (epochBoundaries) {
        parsed.epochBoundaries = epochBoundaries;
    }

    return parsed;
}

function getStatusMessage(status: JobStatusResponse): string | null {
    switch (status.status) {
        case 'pending':
            return 'Training job queued.';
        case 'running':
            return 'Training job started.';
        case 'completed':
            return 'Training completed successfully.';
        case 'cancelled':
            return 'Training cancelled.';
        case 'failed':
            return `Training failed: ${status.error ?? 'Unknown error'}`;
        default:
            return null;
    }
}

function isTerminalTrainingStatus(status: JobStatusResponse['status']): boolean {
    return status === 'cancelled' || status === 'completed' || status === 'failed';
}

function formatMetric(value: number | undefined, decimals: number): string {
    return typeof value === 'number' ? value.toFixed(decimals) : '--';
}

function formatAccuracy(value: number | undefined): string {
    return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '--';
}

export default function TrainingPage() {
    const {
        state,
        updateConfig,
        setSelectedCheckpoint,
        setAdditionalEpochs,
        setDashboardState,
        setChartData,
        setAvailableMetrics,
        setEpochBoundaries,
    } = useTrainingPageState();

    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<DatasetInfo | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [newTrainingError, setNewTrainingError] = useState<string | null>(null);
    const [resumeTrainingError, setResumeTrainingError] = useState<string | null>(null);
    const [metadataModal, setMetadataModal] = useState<MetadataModalState | null>(null);
    const [isNewWizardOpen, setIsNewWizardOpen] = useState(false);
    const [isResumeWizardOpen, setIsResumeWizardOpen] = useState(false);
    const [evaluationCheckpoint, setEvaluationCheckpoint] = useState<CheckpointInfo | null>(null);
    const [evaluationWizardOpen, setEvaluationWizardOpen] = useState(false);
    const [evaluationReportOpen, setEvaluationReportOpen] = useState(false);
    const [evaluationReportCheckpoint, setEvaluationReportCheckpoint] = useState<CheckpointInfo | null>(null);
    const [evaluationReportLoading, setEvaluationReportLoading] = useState(false);
    const [evaluationReportError, setEvaluationReportError] = useState<string | null>(null);
    const [evaluationReportResult, setEvaluationReportResult] = useState<CheckpointEvaluationReport | null>(null);
    const [evaluationReportProgress, setEvaluationReportProgress] = useState<number | null>(null);
    const [evaluationReportStatus, setEvaluationReportStatus] = useState<
        'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null
    >(null);
    const [evaluationJobs, setEvaluationJobs] = usePersistedRecord<StoredEvaluationJob>(EVALUATION_JOB_STORAGE_KEY);
    const evaluationPollers = useRef<Record<string, { stop: () => void }>>({});
    const restoredEvaluationJobs = useRef(false);
    const reportCheckpointRef = useRef<string | null>(null);
    const pollerRef = useRef<{ stop: () => void } | null>(null);
    const statusPollerRef = useRef<{ stop: () => void } | null>(null);
    const lastLoggedEpochRef = useRef<number | null>(null);
    const lastStatusRef = useRef<JobStatusResponse['status'] | null>(null);
    const stopRequestedRef = useRef(false);

    const stopPolling = useCallback(() => {
        if (pollerRef.current) {
            pollerRef.current.stop();
            pollerRef.current = null;
        }
    }, []);

    const stopStatusPolling = useCallback(() => {
        if (statusPollerRef.current) {
            statusPollerRef.current.stop();
            statusPollerRef.current = null;
        }
    }, []);

    const applyTrainingStatusSnapshot = useCallback((status: TrainingStatusResponse) => {
        setDashboardState((prev) => ({
            ...prev,
            isTraining: status.is_training,
            currentEpoch: status.current_epoch,
            totalEpochs: status.total_epochs,
            loss: status.loss,
            valLoss: status.val_loss,
            accuracy: status.accuracy,
            valAccuracy: status.val_accuracy,
            progressPercent: status.progress_percent,
            elapsedSeconds: status.elapsed_seconds,
        }));
    }, [setDashboardState]);

    const startStatusPolling = useCallback((intervalSeconds: number = 2) => {
        stopStatusPolling();
        let stopped = false;
        let timeoutId: ReturnType<typeof setTimeout> | null = null;
        let nextIntervalMs = Math.max(250, intervalSeconds * 1000);

        const stop = () => {
            stopped = true;
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
            if (statusPollerRef.current?.stop === stop) {
                statusPollerRef.current = null;
            }
        };

        const poll = async () => {
            if (stopped) return;

            const { result, error } = await getTrainingStatus();
            if (stopped) return;

            if (error) {
                console.error('Training status poll error:', error);
                timeoutId = setTimeout(poll, nextIntervalMs);
                return;
            }

            if (!result) {
                timeoutId = setTimeout(poll, nextIntervalMs);
                return;
            }

            applyTrainingStatusSnapshot(result);
            if (typeof result.poll_interval === 'number' && result.poll_interval > 0) {
                nextIntervalMs = Math.max(250, result.poll_interval * 1000);
            }

            if (!result.is_training) {
                if (
                    stopRequestedRef.current
                    && lastStatusRef.current !== 'cancelled'
                    && lastStatusRef.current !== 'completed'
                    && lastStatusRef.current !== 'failed'
                ) {
                    setDashboardState((prev) => {
                        const next = [...prev.logEntries, 'Training cancelled.'];
                        return { ...prev, logEntries: next.slice(-200) };
                    });
                    lastStatusRef.current = 'cancelled';
                }
                stopRequestedRef.current = false;
                stopPolling();
                stop();
                return;
            }

            timeoutId = setTimeout(poll, nextIntervalMs);
        };

        statusPollerRef.current = { stop };
        void poll();
    }, [applyTrainingStatusSnapshot, setDashboardState, stopPolling, stopStatusPolling]);

    useEffect(() => {
        reportCheckpointRef.current = evaluationReportCheckpoint?.name ?? null;
    }, [evaluationReportCheckpoint]);

    const updateEvaluationJobs = useCallback((
        updater: (prev: Record<string, StoredEvaluationJob>) => Record<string, StoredEvaluationJob>
    ) => {
        setEvaluationJobs(prev => updater(prev));
    }, [setEvaluationJobs]);

    const removeEvaluationJob = useCallback((checkpoint: string) => {
        updateEvaluationJobs(prev => {
            if (!prev[checkpoint]) {
                return prev;
            }
            const next = { ...prev };
            delete next[checkpoint];
            return next;
        });
    }, [updateEvaluationJobs]);

    const stopEvaluationPolling = useCallback((jobId: string) => {
        const poller = evaluationPollers.current[jobId];
        if (poller) {
            poller.stop();
            delete evaluationPollers.current[jobId];
        }
    }, []);

    const startEvaluationPolling = useCallback((
        checkpoint: string,
        jobId: string,
        jobMeta: StoredEvaluationJob
    ) => {
        if (evaluationPollers.current[jobId]) {
            return;
        }

        const pollIntervalMs = jobMeta.pollIntervalMs ?? 2000;
        const poller = pollCheckpointEvaluationJobStatus(
            jobId,
            (status) => {
                updateEvaluationJobs(prev => {
                    const current = prev[checkpoint] ?? jobMeta;
                    return {
                        ...prev,
                        [checkpoint]: {
                            ...current,
                            status: status.status,
                            progress: status.progress,
                        },
                    };
                });

                if (reportCheckpointRef.current === checkpoint) {
                    setEvaluationReportProgress(status.progress);
                    setEvaluationReportStatus(status.status);
                    setEvaluationReportLoading(status.status === 'running' || status.status === 'pending');
                }
            },
            (status) => {
                stopEvaluationPolling(jobId);
                removeEvaluationJob(checkpoint);

                if (reportCheckpointRef.current === checkpoint) {
                    setEvaluationReportStatus(status.status);
                    setEvaluationReportProgress(status.progress ?? 100);
                    setEvaluationReportLoading(false);
                }

                if (status.status === 'completed') {
                    void (async () => {
                        const { result, error } = await getCheckpointEvaluationReport(checkpoint);
                        if (reportCheckpointRef.current !== checkpoint) {
                            return;
                        }
                        if (error || !result) {
                            setEvaluationReportResult(
                                toCheckpointEvaluationReport(
                                    checkpoint,
                                    jobMeta.metrics,
                                    jobMeta.metricConfigs,
                                    status.result
                                )
                            );
                            setEvaluationReportError(error || 'Failed to load evaluation report');
                        } else {
                            setEvaluationReportResult(result);
                            setEvaluationReportError(null);
                        }
                    })();
                } else if (status.status === 'failed') {
                    if (reportCheckpointRef.current === checkpoint) {
                        setEvaluationReportError(status.error || 'Evaluation failed');
                    }
                } else if (status.status === 'cancelled') {
                    if (reportCheckpointRef.current === checkpoint) {
                        setEvaluationReportError('Evaluation was cancelled');
                    }
                }
            },
            (error) => {
                stopEvaluationPolling(jobId);
                removeEvaluationJob(checkpoint);
                if (reportCheckpointRef.current === checkpoint) {
                    setEvaluationReportLoading(false);
                    setEvaluationReportStatus('failed');
                    setEvaluationReportError(error);
                }
            },
            pollIntervalMs
        );

        evaluationPollers.current[jobId] = poller;
    }, [removeEvaluationJob, stopEvaluationPolling, updateEvaluationJobs]);

    const appendLogLine = useCallback((line: string) => {
        setDashboardState((prev) => {
            const next = [...prev.logEntries, line];
            return { ...prev, logEntries: next.slice(-200) };
        });
    }, [setDashboardState]);

    const resetLogTracking = useCallback(() => {
        lastLoggedEpochRef.current = null;
        lastStatusRef.current = null;
        stopRequestedRef.current = false;
        setDashboardState((prev) => ({ ...prev, logEntries: [] }));
    }, [setDashboardState]);

    const handleStatusTransition = useCallback((status: JobStatusResponse) => {
        if (status.status === lastStatusRef.current) {
            return;
        }

        const statusMessage = getStatusMessage(status);
        if (statusMessage) {
            appendLogLine(statusMessage);
        }

        if (isTerminalTrainingStatus(status.status)) {
            stopRequestedRef.current = false;
        }

        lastStatusRef.current = status.status;
    }, [appendLogLine]);

    const logEpochMetricsIfNeeded = useCallback((
        currentEpoch: number | undefined,
        totalEpochs: number | undefined,
        loss: number | undefined,
        valLoss: number | undefined,
        accuracy: number | undefined,
        valAccuracy: number | undefined,
    ) => {
        if (typeof currentEpoch !== 'number' || currentEpoch <= 0) {
            return;
        }
        if (currentEpoch === lastLoggedEpochRef.current) {
            return;
        }

        const epochLabel = `Epoch ${currentEpoch}/${totalEpochs ?? '--'}`;
        const line = [
            epochLabel,
            `loss ${formatMetric(loss, 4)}`,
            `val_loss ${formatMetric(valLoss, 4)}`,
            `acc ${formatAccuracy(accuracy)}`,
            `val_acc ${formatAccuracy(valAccuracy)}`,
        ].join(' | ');

        appendLogLine(line);
        lastLoggedEpochRef.current = currentEpoch;
    }, [appendLogLine]);

    const applyJobStatus = useCallback((status: JobStatusResponse | null) => {
        if (!status) return;

        const result = status.result ?? {};
        const parsedResult = parseTrainingJobResult(result, status.progress);
        const {
            currentEpoch,
            totalEpochs,
            loss,
            valLoss,
            accuracy,
            valAccuracy,
            progressPercent,
            elapsedSeconds,
            chartData,
            availableMetrics,
            epochBoundaries,
        } = parsedResult;

        setDashboardState((prev) => ({
            ...prev,
            isTraining: status.status === 'running' || status.status === 'pending',
            currentEpoch: currentEpoch ?? prev.currentEpoch,
            totalEpochs: totalEpochs ?? prev.totalEpochs,
            loss: loss ?? prev.loss,
            valLoss: valLoss ?? prev.valLoss,
            accuracy: accuracy ?? prev.accuracy,
            valAccuracy: valAccuracy ?? prev.valAccuracy,
            progressPercent: progressPercent ?? prev.progressPercent,
            elapsedSeconds: elapsedSeconds ?? prev.elapsedSeconds,
        }));

        handleStatusTransition(status);
        logEpochMetricsIfNeeded(
            currentEpoch,
            totalEpochs,
            loss,
            valLoss,
            accuracy,
            valAccuracy,
        );

        if (chartData) {
            setChartData(chartData);
        }
        if (availableMetrics) {
            setAvailableMetrics(availableMetrics);
        }
        if (epochBoundaries) {
            setEpochBoundaries(epochBoundaries);
        }
    }, [
        handleStatusTransition,
        logEpochMetricsIfNeeded,
        setAvailableMetrics,
        setChartData,
        setDashboardState,
        setEpochBoundaries,
    ]);

    const startPolling = useCallback((jobId: string, intervalSeconds: number = 2) => {
        stopPolling();
        pollerRef.current = pollJobStatus(
            getTrainingJobStatus,
            jobId,
            (status) => applyJobStatus(status),
            (status) => applyJobStatus(status),
            (pollError) => {
                console.error('Training poll error:', pollError);
                stopPolling();
            },
            intervalSeconds * 1000
        );
    }, [applyJobStatus, stopPolling]);

    const fetchDatasets = useCallback(async () => {
        const { result, error } = await getProcessedDatasetNames();
        if (error) {
            console.error('Failed to fetch datasets:', error);
            return;
        }
        if (result) {
            setDatasets(result.datasets);
            setSelectedDataset((prev) => {
                if (result.datasets.length === 0) return null;
                if (prev && result.datasets.some((ds) => ds.name === prev.name)) {
                    return result.datasets.find((ds) => ds.name === prev.name) || result.datasets[0];
                }
                return result.datasets[0];
            });
        }
    }, []);

    const fetchCheckpoints = useCallback(async () => {
        const { result, error: fetchError } = await getCheckpoints();
        if (fetchError) {
            console.error('Failed to fetch checkpoints:', fetchError);
            return;
        }
        if (result) {
            setCheckpoints(result.checkpoints);
            if (result.checkpoints.length > 0) {
                const exists = result.checkpoints.some((cp) => cp.name === state.selectedCheckpoint);
                if (!exists) {
                    setSelectedCheckpoint(result.checkpoints[0].name);
                }
            } else {
                setSelectedCheckpoint('');
            }
        }
    }, [setSelectedCheckpoint, state.selectedCheckpoint]);

    useEffect(() => {
        const checkTrainingStatus = async () => {
            const { result } = await getTrainingStatus();
            if (result) {
                applyTrainingStatusSnapshot(result);
            }
            if (result?.is_training && result.job_id) {
                startPolling(result.job_id, result.poll_interval);
                startStatusPolling(result.poll_interval);
            }
        };
        checkTrainingStatus();
        return () => {
            stopPolling();
            stopStatusPolling();
        };
    }, [applyTrainingStatusSnapshot, startPolling, startStatusPolling, stopPolling, stopStatusPolling]);

    useEffect(() => {
        if (restoredEvaluationJobs.current) {
            return;
        }
        restoredEvaluationJobs.current = true;

        const entries = Object.entries(evaluationJobs);
        if (entries.length === 0) {
            return;
        }
        entries.forEach(([checkpoint, job]) => {
            startEvaluationPolling(checkpoint, job.jobId, job);
        });
    }, [evaluationJobs, startEvaluationPolling]);

    useEffect(() => {
        return () => {
            Object.values(evaluationPollers.current).forEach(poller => poller.stop());
            evaluationPollers.current = {};
        };
    }, []);

    useEffect(() => {
        fetchDatasets();
        fetchCheckpoints();
    }, [fetchCheckpoints, fetchDatasets]);

    const handleConfigChange = (key: keyof TrainingConfig, value: TrainingConfig[keyof TrainingConfig]) => {
        updateConfig(key, value);
    };

    const handleStartTraining = async (checkpointName: string) => {
        setIsLoading(true);
        setNewTrainingError(null);
        setChartData([]);
        setAvailableMetrics([]);
        setEpochBoundaries([]);
        resetLogTracking();

        const config: StartTrainingConfig = {
            dataset_name: selectedDataset?.name ?? '',
            epochs: state.config.epochs,
            batch_size: state.config.batchSize,
            num_encoders: state.config.numEncoders,
            num_decoders: state.config.numDecoders,
            embedding_dims: state.config.embeddingDims,
            attention_heads: state.config.attnHeads,
            train_temp: state.config.trainTemp,
            freeze_img_encoder: state.config.freezeImgEncoder,
            use_img_augmentation: state.config.useImgAugment,
            shuffle_with_buffer: state.config.shuffleWithBuffer,
            shuffle_size: state.config.shuffleBufferSize,
            save_checkpoints: state.config.saveCheckpoints,
            checkpoint_id: checkpointName,
            use_device_GPU: state.config.useGpu,
            device_ID: state.config.gpuId,
            jit_compile: state.config.jitCompile,
            jit_backend: state.config.jitBackend,
            use_mixed_precision: state.config.useMixedPrecision,
            dataloader_workers: state.config.dataloaderWorkers,
            prefetch_factor: state.config.prefetchFactor,
            pin_memory: state.config.pinMemory,
            persistent_workers: state.config.persistentWorkers,
            plot_training_metrics: state.config.realTimePlot,
            use_scheduler: state.config.useScheduler,
            target_LR: state.config.targetLR,
            warmup_steps: state.config.warmupSteps,
        };

        const { result: startResult, error: trainError } = await startTraining(config);
        setIsLoading(false);

        if (trainError) {
            setNewTrainingError(trainError);
            console.error('Training failed:', trainError);
            return;
        }
        if (startResult) {
            setIsNewWizardOpen(false);
            appendLogLine(`Training job started (ID: ${startResult.job_id}).`);
            startPolling(startResult.job_id, startResult.poll_interval);
            startStatusPolling(startResult.poll_interval);
        }
    };

    const handleResumeTraining = async () => {
        if (!state.selectedCheckpoint) return;

        setIsLoading(true);
        setResumeTrainingError(null);
        resetLogTracking();

        const { result: startResult, error: resumeError } = await resumeTraining(
            state.selectedCheckpoint,
            state.additionalEpochs
        );
        setIsLoading(false);

        if (resumeError) {
            setResumeTrainingError(resumeError);
            console.error('Resume training failed:', resumeError);
            return;
        }
        if (startResult) {
            setIsResumeWizardOpen(false);
            appendLogLine(`Resume job started (ID: ${startResult.job_id}).`);
            startPolling(startResult.job_id, startResult.poll_interval);
            startStatusPolling(startResult.poll_interval);
        }
    };

    const handleEvaluationConfirm = async (payload: {
        metrics: string[];
        metricConfigs: Record<string, { dataFraction: number }>;
    }) => {
        if (!evaluationCheckpoint) {
            return;
        }

        const checkpointName = evaluationCheckpoint.name;
        const metricConfigsForApi = toApiMetricConfigs(payload.metricConfigs);
        setEvaluationWizardOpen(false);
        setEvaluationCheckpoint(null);
        setEvaluationReportCheckpoint(evaluationCheckpoint);
        setEvaluationReportResult({
            checkpoint: checkpointName,
            metrics: payload.metrics,
            metric_configs: metricConfigsForApi,
        });
        setEvaluationReportError(null);
        setEvaluationReportProgress(0);
        setEvaluationReportStatus('pending');
        setEvaluationReportLoading(true);
        setEvaluationReportOpen(true);

        const { result: jobResult, error: startError } = await evaluateCheckpoint(
            checkpointName,
            payload.metrics,
            10,
            metricConfigsForApi,
            42
        );

        if (startError || !jobResult) {
            setEvaluationReportLoading(false);
            setEvaluationReportError(startError || 'Failed to start evaluation job');
            return;
        }

        const jobMeta: StoredEvaluationJob = {
            jobId: jobResult.job_id,
            metrics: payload.metrics,
            metricConfigs: payload.metricConfigs,
            pollIntervalMs: (jobResult.poll_interval ?? 2) * 1000,
            status: jobResult.status,
            progress: 0,
        };

        updateEvaluationJobs(prev => ({
            ...prev,
            [checkpointName]: jobMeta,
        }));
        setEvaluationReportProgress(0);
        setEvaluationReportStatus(jobResult.status);
        setEvaluationReportLoading(true);
        startEvaluationPolling(checkpointName, jobResult.job_id, jobMeta);
    };

    const handleVisualizeEvaluationReport = async (checkpoint: CheckpointInfo) => {
        setEvaluationReportCheckpoint(checkpoint);
        setEvaluationReportResult(null);
        setEvaluationReportError(null);
        setEvaluationReportProgress(null);
        setEvaluationReportStatus(null);
        setEvaluationReportLoading(true);
        setEvaluationReportOpen(true);

        const activeJob = evaluationJobs[checkpoint.name];
        if (activeJob && activeJob.status !== 'completed' && activeJob.status !== 'failed' && activeJob.status !== 'cancelled') {
            setEvaluationReportResult({
                checkpoint: checkpoint.name,
                metrics: activeJob.metrics,
                metric_configs: toApiMetricConfigs(activeJob.metricConfigs),
            });
            setEvaluationReportProgress(activeJob.progress ?? 0);
            setEvaluationReportStatus(activeJob.status ?? 'running');
            setEvaluationReportLoading(activeJob.status === 'running' || activeJob.status === 'pending');
            startEvaluationPolling(checkpoint.name, activeJob.jobId, activeJob);
            return;
        }

        const { result, error } = await getCheckpointEvaluationReport(checkpoint.name);
        if (error || !result) {
            setEvaluationReportLoading(false);
            setEvaluationReportError(error || 'Failed to load evaluation report');
            return;
        }

        setEvaluationReportResult(result);
        setEvaluationReportProgress(100);
        setEvaluationReportStatus('completed');
        setEvaluationReportLoading(false);
    };

    const handleStopTraining = async () => {
        if (!stopRequestedRef.current) {
            appendLogLine('Stop requested. Finishing current batch and shutting down training.');
        }
        stopRequestedRef.current = true;
        const { error: stopError } = await stopTraining();
        if (stopError) {
            console.error('Stop training failed:', stopError);
            stopRequestedRef.current = false;
        }
    };

    const handleShowDatasetMetadata = async (dataset: DatasetInfo) => {
        setMetadataModal({
            title: 'Dataset Processing Metadata',
            subtitle: dataset.name,
        });

        const { result, error } = await getProcessingMetadata(dataset.name);
        if (error || !result) {
            const parsedError = error ? parseMetadataError(error) : null;
            const friendlyError = parsedError?.includes('No processing metadata found')
                ? `No processing metadata found for "${dataset.name}". Run "Build Dataset" on the Dataset page to generate it.`
                : parsedError || 'No metadata found';
            setMetadataModal({
                title: 'Dataset Processing Metadata',
                subtitle: dataset.name,
                error: friendlyError,
            });
            return;
        }

        const metadata = {
            dataset_name: result.dataset_name,
            ...result.metadata,
        };
        setMetadataModal({
            title: 'Dataset Processing Metadata',
            subtitle: result.dataset_name,
            sections: [
                {
                    title: 'Processing Parameters',
                    entries: buildEntries(metadata, PROCESSING_METADATA_ORDER),
                },
            ],
        });
    };

    const handleShowCheckpointMetadata = async (checkpoint: CheckpointInfo) => {
        setMetadataModal({
            title: 'Checkpoint Metadata',
            subtitle: checkpoint.name,
        });

        const { result, error } = await getCheckpointMetadata(checkpoint.name);
        if (error || !result) {
            const parsedError = error ? parseMetadataError(error) : null;
            setMetadataModal({
                title: 'Checkpoint Metadata',
                subtitle: checkpoint.name,
                error: parsedError || 'No metadata found',
            });
            return;
        }

        setMetadataModal({
            title: 'Checkpoint Metadata',
            subtitle: result.checkpoint,
            sections: [
                { title: 'Training Configuration', entries: buildEntries(result.configuration) },
                { title: 'Dataset Metadata', entries: buildEntries(result.metadata) },
                { title: 'Session Summary', entries: buildEntries(result.session) },
            ],
        });
    };

    const handleDeleteDataset = async (dataset: DatasetInfo) => {
        const confirmed = globalThis.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`);
        if (!confirmed) return;

        const { error } = await deleteDataset(dataset.name);
        if (error) {
            console.error('Failed to delete dataset:', error);
            return;
        }

        await fetchDatasets();
    };

    const handleDeleteCheckpoint = async (checkpoint: CheckpointInfo) => {
        const confirmed = globalThis.confirm(`Delete checkpoint "${checkpoint.name}"? This cannot be undone.`);
        if (!confirmed) return;

        const { error } = await deleteCheckpoint(checkpoint.name);
        if (error) {
            console.error('Failed to delete checkpoint:', error);
            return;
        }

        await fetchCheckpoints();
    };

    const selectedCheckpointInfo = useMemo(
        () => checkpoints.find((cp) => cp.name === state.selectedCheckpoint) || null,
        [checkpoints, state.selectedCheckpoint]
    );

    return (
        <div className="training-container">
            <div className="header">
                <h1>XREPORT Transformer</h1>
                <p>Configure and monitor your training sessions</p>
            </div>

            <div className="training-panels">
                <div className="training-panel">
                    <div className="panel-left">
                        <div className="panel-header">
                            <div>
                                <h3>New Training Session</h3>
                                <p>Select a processed dataset to configure your next run.</p>
                            </div>
                            <button
                                className="panel-refresh"
                                onClick={fetchDatasets}
                                type="button"
                                aria-label="Refresh datasets"
                            >
                                <RefreshCw size={16} />
                            </button>
                        </div>

                        <div className="panel-list">
                            {datasets.length === 0 && (
                                <div className="panel-empty">No datasets available yet.</div>
                            )}
                            {datasets.map((dataset) => {
                                const isSelected = selectedDataset?.name === dataset.name;
                                return (
                                    <div
                                        key={dataset.name}
                                        className={`panel-row ${isSelected ? 'selected' : ''}`}
                                    >
                                        <button
                                            type="button"
                                            className="panel-row-main panel-row-main-button"
                                            aria-pressed={isSelected}
                                            onClick={() => setSelectedDataset(dataset)}
                                        >
                                            <span className="panel-row-title">{dataset.name}</span>
                                            <span className="panel-row-count">{dataset.row_count.toLocaleString()} rows</span>
                                        </button>
                                        <div className="panel-row-actions">
                                            <button
                                                type="button"
                                                className="icon-button"
                                                title="Show metadata"
                                                onClick={() => handleShowDatasetMetadata(dataset)}
                                            >
                                                <Info size={15} />
                                            </button>
                                            <button
                                                type="button"
                                                className="icon-button danger"
                                                title="Delete dataset"
                                                onClick={() => handleDeleteDataset(dataset)}
                                            >
                                                <Trash2 size={15} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>



                    <div className="panel-right">
                        <div className="panel-card">
                            <div className="panel-card-header">
                                <div className="panel-card-title-row">
                                    <Play size={18} />
                                    <h4>Initialize Training</h4>
                                </div>
                                <p>Launch the configuration wizard to set up your training run.</p>
                            </div>
                            <div className="panel-card-summary">
                                <span>Selected Dataset</span>
                                <strong>{selectedDataset?.name || 'None selected'}</strong>
                                <span>Samples</span>
                                <strong>{selectedDataset ? selectedDataset.row_count.toLocaleString() : 'N/A'}</strong>
                            </div>
                            <button
                                className="btn btn-primary"
                                type="button"
                                onClick={() => {
                                    setNewTrainingError(null);
                                    setIsNewWizardOpen(true);
                                }}
                                disabled={!selectedDataset}
                            >
                                <Play size={16} />
                                Configure Training
                            </button>
                        </div>
                    </div>
                </div>

                <div className="training-panel">
                    <div className="panel-left">
                        <div className="panel-header">
                            <div>
                                <h3>Resume Training</h3>
                                <p>Pick a checkpoint to continue training from a saved state.</p>
                            </div>
                            <button
                                className="panel-refresh"
                                onClick={fetchCheckpoints}
                                type="button"
                                aria-label="Refresh checkpoints"
                            >
                                <RefreshCw size={16} />
                            </button>
                        </div>
                        <div className="panel-list">
                            {checkpoints.length === 0 && (
                                <div className="panel-empty">No checkpoints available yet.</div>
                            )}
                            {checkpoints.map((checkpoint) => {
                                const isSelected = state.selectedCheckpoint === checkpoint.name;
                                return (
                                    <div
                                        key={checkpoint.name}
                                        className={`panel-row ${isSelected ? 'selected' : ''}`}
                                    >
                                        <button
                                            type="button"
                                            className="panel-row-main panel-row-main-button"
                                            aria-pressed={isSelected}
                                            onClick={() => setSelectedCheckpoint(checkpoint.name)}
                                        >
                                            <span className="panel-row-title">{checkpoint.name}</span>
                                            <span className="panel-row-meta">
                                                {checkpoint.epochs} epochs · loss {checkpoint.loss.toFixed(4)}
                                            </span>
                                        </button>
                                        <div className="panel-row-actions">
                                            <button
                                                type="button"
                                                className="icon-button"
                                                title="Show metadata"
                                                onClick={() => handleShowCheckpointMetadata(checkpoint)}
                                            >
                                                <Info size={15} />
                                            </button>
                                            <button
                                                type="button"
                                                className="icon-button"
                                                title="View evaluation report"
                                                onClick={() => handleVisualizeEvaluationReport(checkpoint)}
                                            >
                                                <BarChart2 size={15} />
                                            </button>
                                            <button
                                                type="button"
                                                className="icon-button danger"
                                                title="Delete checkpoint"
                                                onClick={() => handleDeleteCheckpoint(checkpoint)}
                                            >
                                                <Trash2 size={15} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>



                    <div className="panel-right">
                        <div className="panel-card">
                            <div className="panel-card-header">
                                <div className="panel-card-title-row">
                                    <RotateCcw size={18} />
                                    <h4>Checkpoint Actions</h4>
                                </div>
                                <p>Resume training or evaluate the performance of the selected checkpoint.</p>
                            </div>
                            <div className="panel-card-summary">
                                <span>Selected Checkpoint</span>
                                <strong>{state.selectedCheckpoint || 'None selected'}</strong>
                                <span>Epochs</span>
                                <strong>{selectedCheckpointInfo ? selectedCheckpointInfo.epochs : 'N/A'}</strong>
                            </div>
                            <div className="panel-card-actions">
                                <button
                                    className="btn btn-primary panel-card-action-btn"
                                    type="button"
                                    onClick={() => {
                                        setResumeTrainingError(null);
                                        setIsResumeWizardOpen(true);
                                    }}
                                    disabled={!state.selectedCheckpoint}
                                >
                                    <RotateCcw size={16} />
                                    Resume Training
                                </button>
                                <button
                                    className="btn btn-secondary panel-card-action-btn panel-card-action-btn-center"
                                    type="button"
                                    onClick={() => {
                                        if (state.selectedCheckpoint && selectedCheckpointInfo) {
                                            setEvaluationCheckpoint(selectedCheckpointInfo);
                                            setEvaluationWizardOpen(true);
                                        }
                                    }}
                                    disabled={!state.selectedCheckpoint}
                                >
                                    <Activity size={16} />
                                    Evaluate Model
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <TrainingDashboard
                onStopTraining={handleStopTraining}
                dashboardState={state.dashboardState}
            />

            <NewTrainingWizard
                isOpen={isNewWizardOpen}
                config={state.config}
                onConfigChange={handleConfigChange}
                onClose={() => setIsNewWizardOpen(false)}
                onConfirm={handleStartTraining}
                isLoading={isLoading}
                selectedDatasetLabel={selectedDataset?.name ?? ''}
                error={newTrainingError}
            />

            <ResumeTrainingWizard
                isOpen={isResumeWizardOpen}
                checkpoints={checkpoints}
                selectedCheckpoint={state.selectedCheckpoint}
                onCheckpointChange={setSelectedCheckpoint}
                additionalEpochs={state.additionalEpochs}
                onAdditionalEpochsChange={setAdditionalEpochs}
                onClose={() => setIsResumeWizardOpen(false)}
                onConfirm={handleResumeTraining}
                isLoading={isLoading}
                error={resumeTrainingError}
            />

            <MetadataModal state={metadataModal} onClose={() => setMetadataModal(null)} />

            <EvaluationWizard
                isOpen={evaluationWizardOpen}
                onClose={() => {
                    setEvaluationWizardOpen(false);
                    setEvaluationCheckpoint(null);
                }}
                checkpointName={evaluationCheckpoint?.name ?? ''}
                onConfirm={handleEvaluationConfirm}
            />

            <CheckpointEvaluationReportModal
                isOpen={evaluationReportOpen}
                checkpointName={evaluationReportCheckpoint?.name ?? null}
                isLoading={evaluationReportLoading}
                report={evaluationReportResult}
                error={evaluationReportError}
                progress={evaluationReportProgress}
                status={evaluationReportStatus}
                onClose={() => {
                    setEvaluationReportOpen(false);
                    setEvaluationReportCheckpoint(null);
                }}
            />
        </div>
    );
}
