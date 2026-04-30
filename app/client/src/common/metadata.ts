import { MetadataEntry } from '../types/metadata';

export const PROCESSING_METADATA_ORDER = [
    'source_dataset',
    'dataset_name',
    'date',
    'seed',
    'sample_size',
    'validation_size',
    'vocabulary_size',
    'max_report_size',
    'tokenizer',
];

export const formatMetadataValue = (value: unknown): string => {
    if (value === null || value === undefined || value === '') {
        return 'N/A';
    }
    if (typeof value === 'object') {
        try {
            return JSON.stringify(value, null, 2);
        } catch {
            return String(value);
        }
    }
    return String(value);
};

export const formatMetadataLabel = (label: string): string => {
    return label
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (char) => char.toUpperCase());
};

export const stripHashFields = (data: Record<string, unknown>): Record<string, unknown> => {
    return Object.fromEntries(
        Object.entries(data).filter(([key]) => !key.toLowerCase().includes('hash'))
    );
};

export const buildEntries = (data: Record<string, unknown>, preferredOrder: string[] = []): MetadataEntry[] => {
    const sanitized = stripHashFields(data);
    const preferredSet = new Set(preferredOrder);
    const orderedKeys = preferredOrder.filter((key) => key in sanitized);
    const remainingKeys = Object.keys(sanitized)
        .filter((key) => !preferredSet.has(key))
        .sort((a, b) => a.localeCompare(b));
    const keys = preferredOrder.length > 0 ? [...orderedKeys, ...remainingKeys] : Object.keys(sanitized);
    return keys.map((label) => ({
        label: formatMetadataLabel(label),
        value: formatMetadataValue(sanitized[label]),
    }));
};

export const parseMetadataError = (error: string): string => {
    const markerIndex = error.indexOf(':');
    if (markerIndex >= 0) {
        const candidate = error.slice(markerIndex + 1).trim();
        if (!candidate.startsWith('{') || !candidate.endsWith('}')) {
            return error;
        }
        try {
            const parsed = JSON.parse(candidate) as { detail?: string };
            if (parsed.detail) {
                return parsed.detail;
            }
        } catch {
            return error;
        }
    }
    return error;
};
