import { useCallback, useEffect, useRef } from 'react';

type PollerHandle = {
    stop: () => void;
};

type PollerFactory = () => PollerHandle;

interface JobPollerRegistry {
    startPoller: (jobId: string, factory: PollerFactory) => boolean;
    stopPoller: (jobId: string) => void;
    stopAllPollers: () => void;
}

export function useJobPollerRegistry(): JobPollerRegistry {
    const pollersRef = useRef<Record<string, PollerHandle>>({});

    const stopPoller = useCallback((jobId: string) => {
        const poller = pollersRef.current[jobId];
        if (!poller) {
            return;
        }
        poller.stop();
        delete pollersRef.current[jobId];
    }, []);

    const stopAllPollers = useCallback(() => {
        Object.values(pollersRef.current).forEach((poller) => poller.stop());
        pollersRef.current = {};
    }, []);

    const startPoller = useCallback((jobId: string, factory: PollerFactory) => {
        if (pollersRef.current[jobId]) {
            return false;
        }
        pollersRef.current[jobId] = factory();
        return true;
    }, []);

    useEffect(() => () => {
        stopAllPollers();
    }, [stopAllPollers]);

    return {
        startPoller,
        stopPoller,
        stopAllPollers,
    };
}
