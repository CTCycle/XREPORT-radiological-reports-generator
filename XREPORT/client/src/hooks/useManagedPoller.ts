import { useCallback, useEffect, useRef } from 'react';

export interface PollerHandle {
    stop: () => void;
}

type PollerFactory = () => PollerHandle;

interface ManagedPoller {
    startPolling: (createPoller: PollerFactory) => void;
    stopPolling: () => void;
}

export function useManagedPoller(): ManagedPoller {
    const pollerRef = useRef<PollerHandle | null>(null);

    const stopPolling = useCallback(() => {
        if (!pollerRef.current) {
            return;
        }
        pollerRef.current.stop();
        pollerRef.current = null;
    }, []);

    const startPolling = useCallback((createPoller: PollerFactory) => {
        stopPolling();
        pollerRef.current = createPoller();
    }, [stopPolling]);

    useEffect(() => () => {
        stopPolling();
    }, [stopPolling]);

    return {
        startPolling,
        stopPolling,
    };
}
