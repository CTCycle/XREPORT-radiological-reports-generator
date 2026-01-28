import React from 'react';
import { X } from 'lucide-react';
import '../pages/TrainingPage.css';

export type MetadataEntry = {
    label: string;
    value: string;
};

export type MetadataSection = {
    title: string;
    entries: MetadataEntry[];
};

export type MetadataModalState = {
    title: string;
    subtitle?: string;
    sections?: MetadataSection[];
    error?: string;
};

export const PROCESSING_METADATA_ORDER = [
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
    const match = error.match(/:\s*(\{.*\})\s*$/);
    if (match) {
        try {
            const parsed = JSON.parse(match[1]) as { detail?: string };
            if (parsed.detail) {
                return parsed.detail;
            }
        } catch {
            return error;
        }
    }
    return error;
};

interface MetadataModalProps {
    state: MetadataModalState | null;
    onClose: () => void;
}

export default function MetadataModal({ state, onClose }: MetadataModalProps) {
    if (!state) return null;

    return (
        <div className="metadata-backdrop" onClick={onClose}>
            <div className="metadata-modal" onClick={(event) => event.stopPropagation()}>
                <div className="metadata-header">
                    <div>
                        <h3>{state.title}</h3>
                        {state.subtitle && <p className="metadata-subtitle">{state.subtitle}</p>}
                    </div>
                    <button className="metadata-close" onClick={onClose} aria-label="Close metadata dialog">
                        <X size={18} />
                    </button>
                </div>
                <div className="metadata-body">
                    {state.error && <div className="metadata-error">{state.error}</div>}
                    {!state.error && state.sections?.map((section) => (
                        <div className="metadata-section" key={section.title}>
                            <h4>{section.title}</h4>
                            <div className="metadata-grid">
                                {section.entries.map((entry) => (
                                    <div className="metadata-row" key={`${section.title}-${entry.label}`}>
                                        <span className="metadata-label">{entry.label}</span>
                                        <span className="metadata-value">{entry.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="metadata-footer">
                    <button className="btn btn-secondary" type="button" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}
