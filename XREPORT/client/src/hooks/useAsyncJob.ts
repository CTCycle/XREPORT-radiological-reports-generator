import { useCallback, useEffect, useRef, useState } from 'react';
import { createJobStatusPoller, PollerHandle } from '../services/jobPolling';
import { JobCancelResponse, JobLifecycleStatus, JobStartResponse, JobStatusResponse } from '../types/jobs';

type AsyncJobResult<T> = {
    result: T | null;
    error: string | null;
};

interface UseAsyncJobOptions<TStartArgs extends unknown[], TParsedResult> {
    startJob: (...args: TStartArgs) => Promise<AsyncJobResult<JobStartResponse>>;
    getStatus: (jobId: string) => Promise<AsyncJobResult<JobStatusResponse>>;
    cancelJob?: (jobId: string) => Promise<AsyncJobResult<JobCancelResponse>>;
    parseResult?: (result: Record<string, unknown> | null, status: JobStatusResponse) => TParsedResult;
    onUpdate?: (status: JobStatusResponse, parsedResult?: TParsedResult) => void;
    onComplete?: (status: JobStatusResponse, parsedResult?: TParsedResult) => void;
}

interface UseAsyncJobState<TParsedResult> {
    jobId: string | null;
    status: JobLifecycleStatus | null;
    progress: number;
    error: string | null;
    result: TParsedResult | undefined;
    isRunning: boolean;
    start: (...args: unknown[]) => Promise<JobStartResponse | null>;
    attach: (existingJobId: string, pollIntervalSeconds?: number, initialStatus?: JobLifecycleStatus) => void;
    cancel: () => Promise<boolean>;
    reset: () => void;
}

export function useAsyncJob<TStartArgs extends unknown[] = [], TParsedResult = never>(
    options: UseAsyncJobOptions<TStartArgs, TParsedResult>
): UseAsyncJobState<TParsedResult> {
    const { startJob, getStatus, cancelJob, parseResult, onUpdate, onComplete } = options;

    const [jobId, setJobId] = useState<string | null>(null);
    const [status, setStatus] = useState<JobLifecycleStatus | null>(null);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<TParsedResult | undefined>(undefined);

    const pollerRef = useRef<PollerHandle | null>(null);
    const mountedRef = useRef(true);

    const stopPolling = useCallback(() => {
        if (pollerRef.current) {
            pollerRef.current.stop();
            pollerRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => {
            mountedRef.current = false;
            stopPolling();
        };
    }, [stopPolling]);

    const applyStatus = useCallback((currentStatus: JobStatusResponse) => {
        const parsedResult = parseResult ? parseResult(currentStatus.result, currentStatus) : undefined;
        if (!mountedRef.current) {
            return;
        }
        setStatus(currentStatus.status);
        setProgress(currentStatus.progress ?? 0);
        setError(currentStatus.error ?? null);
        if (parsedResult !== undefined) {
            setResult(parsedResult);
        }
        onUpdate?.(currentStatus, parsedResult);
        return parsedResult;
    }, [onUpdate, parseResult]);

    const start = useCallback(async (...args: unknown[]) => {
        stopPolling();
        setError(null);
        setResult(undefined);
        setProgress(0);

        const { result: startedJob, error: startError } = await startJob(...(args as TStartArgs));

        if (!mountedRef.current) {
            return null;
        }

        if (startError || !startedJob) {
            setError(startError ?? 'Failed to start job');
            setStatus('failed');
            return null;
        }

        setJobId(startedJob.job_id);
        setStatus(startedJob.status);

        const pollIntervalMs = Math.max(250, (startedJob.poll_interval ?? 2) * 1000);

        pollerRef.current = createJobStatusPoller(
            getStatus,
            startedJob.job_id,
            (currentStatus) => {
                applyStatus(currentStatus);
            },
            (currentStatus) => {
                stopPolling();
                const parsedResult = applyStatus(currentStatus);
                onComplete?.(currentStatus, parsedResult);
            },
            (pollError) => {
                stopPolling();
                if (!mountedRef.current) {
                    return;
                }
                setError(pollError);
                setStatus('failed');
            },
            pollIntervalMs,
        );

        return startedJob;
    }, [applyStatus, getStatus, onComplete, startJob, stopPolling]);

    const attach = useCallback((
        existingJobId: string,
        pollIntervalSeconds: number = 2,
        initialStatus: JobLifecycleStatus = 'running',
    ) => {
        stopPolling();
        setJobId(existingJobId);
        setStatus(initialStatus);
        setError(null);
        setProgress(0);

        const pollIntervalMs = Math.max(250, pollIntervalSeconds * 1000);
        pollerRef.current = createJobStatusPoller(
            getStatus,
            existingJobId,
            (currentStatus) => {
                applyStatus(currentStatus);
            },
            (currentStatus) => {
                stopPolling();
                const parsedResult = applyStatus(currentStatus);
                onComplete?.(currentStatus, parsedResult);
            },
            (pollError) => {
                stopPolling();
                if (!mountedRef.current) {
                    return;
                }
                setError(pollError);
                setStatus('failed');
            },
            pollIntervalMs,
        );
    }, [applyStatus, getStatus, onComplete, stopPolling]);

    const cancel = useCallback(async () => {
        if (!cancelJob || !jobId) {
            return false;
        }
        const { error: cancelError } = await cancelJob(jobId);
        if (!mountedRef.current) {
            return false;
        }
        if (cancelError) {
            setError(cancelError);
            return false;
        }
        return true;
    }, [cancelJob, jobId]);

    const reset = useCallback(() => {
        stopPolling();
        setJobId(null);
        setStatus(null);
        setProgress(0);
        setError(null);
        setResult(undefined);
    }, [stopPolling]);

    return {
        jobId,
        status,
        progress,
        error,
        result,
        isRunning: status === 'pending' || status === 'running',
        start,
        attach,
        cancel,
        reset,
    };
}
