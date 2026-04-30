export interface PollerHandle {
    stop: () => void;
}

interface PollingResult<TStatus> {
    result: TStatus | null;
    error: string | null;
}

type StatusFetcher<TStatus> = (jobId: string) => Promise<PollingResult<TStatus>>;

export function createJobStatusPoller<TStatus extends { status: string }>(
    fetchStatus: StatusFetcher<TStatus>,
    jobId: string,
    onUpdate: (status: TStatus) => void,
    onComplete: (status: TStatus) => void,
    onError: (error: string) => void,
    intervalMs: number = 2000
): PollerHandle {
    let stopped = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
        if (stopped) {
            return;
        }

        const { result, error } = await fetchStatus(jobId);

        if (stopped) {
            return;
        }

        if (error) {
            onError(error);
            return;
        }

        if (!result) {
            onError('No result returned');
            return;
        }

        onUpdate(result);

        if (result.status === 'completed' || result.status === 'failed' || result.status === 'cancelled') {
            onComplete(result);
            return;
        }

        timeoutId = setTimeout(poll, intervalMs);
    };

    void poll();

    return {
        stop: () => {
            stopped = true;
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        },
    };
}
