import { useCallback, useState } from 'react';

function readPersistedRecord<T>(storageKey: string): Record<string, T> {
    try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) {
            return {};
        }

        const parsed: unknown = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return {};
        }

        return parsed as Record<string, T>;
    } catch {
        return {};
    }
}

function writePersistedRecord<T>(storageKey: string, value: Record<string, T>) {
    try {
        localStorage.setItem(storageKey, JSON.stringify(value));
    } catch {
        // Ignore storage errors (private mode or quota limits)
    }
}

export function usePersistedRecord<T>(storageKey: string) {
    const [record, setRecord] = useState<Record<string, T>>(() => readPersistedRecord<T>(storageKey));

    const setPersistedRecord = useCallback((
        updater: Record<string, T> | ((prev: Record<string, T>) => Record<string, T>)
    ) => {
        setRecord((prev) => {
            const next = typeof updater === 'function'
                ? (updater as (prev: Record<string, T>) => Record<string, T>)(prev)
                : updater;
            writePersistedRecord(storageKey, next);
            return next;
        });
    }, [storageKey]);

    return [record, setPersistedRecord] as const;
}
