export function asRecord(value: unknown): Record<string, unknown> | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return null;
    }
    return value as Record<string, unknown>;
}

export function readBoolean(value: unknown): boolean | undefined {
    return typeof value === 'boolean' ? value : undefined;
}

export function readNumber(value: unknown): number | undefined {
    return typeof value === 'number' ? value : undefined;
}

export function readString(value: unknown): string | undefined {
    return typeof value === 'string' ? value : undefined;
}

export function readNumberArray(value: unknown): number[] | undefined {
    if (!Array.isArray(value)) {
        return undefined;
    }
    if (value.some((entry) => typeof entry !== 'number')) {
        return undefined;
    }
    return value;
}
