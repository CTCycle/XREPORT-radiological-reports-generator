import { useMemo } from 'react';

export type JobExecutionStatus =
    | 'pending'
    | 'running'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | null
    | undefined;

interface JobProgressState {
    isRunning: boolean;
    showProgress: boolean;
}

export function useJobProgressState(
    isLoading: boolean,
    status: JobExecutionStatus
): JobProgressState {
    return useMemo(() => {
        const isRunning = status === 'running' || status === 'pending';
        return {
            isRunning,
            showProgress: isLoading || isRunning,
        };
    }, [isLoading, status]);
}
