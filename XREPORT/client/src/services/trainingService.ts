// Training service API functions

export interface ImagePathResponse {
    valid: boolean;
    folder_path: string;
    image_count: number;
    message: string;
}

export interface DatasetUploadResponse {
    success: boolean;
    filename: string;
    row_count: number;
    column_count: number;
    columns: string[];
    message: string;
}

export interface LoadDatasetRequest {
    image_folder_path: string;
    sample_size?: number;
    seed?: number;
}

export interface LoadDatasetResponse {
    success: boolean;
    total_images: number;
    matched_records: number;
    unmatched_records: number;
    message: string;
}

export interface DirectoryItem {
    name: string;
    path: string;
    is_dir: boolean;
    image_count: number;
}

export interface BrowseResponse {
    current_path: string;
    parent_path: string | null;
    items: DirectoryItem[];
    drives: string[];
}

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

/**
 * Validate an image folder path on the server
 */
export async function validateImagePath(
    folderPath: string
): Promise<{ result: ImagePathResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/images/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath }),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<ImagePathResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Upload a CSV or XLSX dataset file
 */
export async function uploadDataset(
    file: File
): Promise<{ result: DatasetUploadResponse | null; error: string | null }> {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/training/dataset/upload', {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<DatasetUploadResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Load dataset by merging images folder with uploaded dataset
 */
export async function loadDataset(
    request: LoadDatasetRequest
): Promise<{ result: LoadDatasetResponse | null; error: string | null }> {
    try {
        const response = await fetch('/api/training/dataset/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_folder_path: request.image_folder_path,
                sample_size: request.sample_size ?? 1.0,
                seed: request.seed ?? 42,
            }),
        });
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<LoadDatasetResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

/**
 * Browse server-side directories
 */
export async function browseDirectory(
    path: string = ''
): Promise<{ result: BrowseResponse | null; error: string | null }> {
    try {
        const url = path
            ? `/api/training/browse?path=${encodeURIComponent(path)}`
            : '/api/training/browse';
        const response = await fetch(url);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<BrowseResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}
