// Inference service API functions

export interface CheckpointInfo {
    name: string;
    created: string | null;
}

export interface CheckpointsResponse {
    checkpoints: CheckpointInfo[];
    success: boolean;
    message: string;
}

export interface GenerationResponse {
    success: boolean;
    message: string;
    reports: Record<string, string> | null;
}

export type GenerationMode = 'greedy_search' | 'beam_search';

export interface InferenceStreamMessage {
    type: 'start' | 'token' | 'complete' | 'error' | 'pong';
    image_index?: number;
    token?: string;
    step?: number;
    total?: number;
    total_images?: number;
    checkpoint?: string;
    mode?: string;
    reports?: Record<string, string>;
    message?: string;
}

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Get list of available checkpoints for inference
 */
export async function getInferenceCheckpoints(): Promise<{
    result: CheckpointsResponse | null;
    error: string | null;
}> {
    try {
        const response = await fetch('/api/inference/checkpoints');
        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }
        const payload = await readJson<CheckpointsResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Generate reports for uploaded images
 */
export async function generateReports(
    images: File[],
    checkpoint: string,
    generationMode: GenerationMode
): Promise<{ result: GenerationResponse | null; error: string | null }> {
    try {
        const formData = new FormData();
        formData.append('checkpoint', checkpoint);
        formData.append('generation_mode', generationMode);

        for (const image of images) {
            formData.append('images', image);
        }

        const response = await fetch('/api/inference/generate', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const body = await response.text();
            return {
                result: null,
                error: `${response.status} ${response.statusText}: ${body}`,
            };
        }

        const payload = await readJson<GenerationResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Connect to inference WebSocket for streaming updates
 */
export function connectInferenceWebSocket(
    onMessage: (message: InferenceStreamMessage) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
): WebSocket {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/inference/ws`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('Inference WebSocket connected');
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data) as InferenceStreamMessage;
            onMessage(message);
        } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
        }
    };

    ws.onerror = (error) => {
        console.error('Inference WebSocket error:', error);
        if (onError) {
            onError(error);
        }
    };

    ws.onclose = () => {
        console.log('Inference WebSocket closed');
        if (onClose) {
            onClose();
        }
    };

    return ws;
}

/**
 * Close inference WebSocket connection
 */
export function disconnectInferenceWebSocket(ws: WebSocket | null): void {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
}
