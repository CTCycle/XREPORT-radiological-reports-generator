export interface TableInfo {
    table_name: string;
    display_name: string;
}

export interface TableListResponse {
    tables: TableInfo[];
}

export interface TableDataResponse {
    table_name: string;
    display_name: string;
    row_count: number;
    column_count: number;
    columns: string[];
    data: Record<string, unknown>[];
}

async function readJson<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
        throw new Error(`Unexpected response content-type: ${contentType}`);
    }
    return (await response.json()) as T;
}

export async function fetchTableList(): Promise<{ tables: TableInfo[]; error: string | null }> {
    try {
        const response = await fetch('/data/browser/tables');
        if (!response.ok) {
            const body = await response.text();
            return { tables: [], error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TableListResponse>(response);
        return { tables: payload.tables, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { tables: [], error: message };
    }
}

export async function fetchTableData(
    tableName: string,
    limit: number,
    offset: number,
): Promise<{ result: TableDataResponse | null; error: string | null }> {
    if (!tableName) return { result: null, error: 'No table selected.' };
    try {
        const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
        const response = await fetch(`/data/browser/data/${encodeURIComponent(tableName)}?${params.toString()}`);
        if (!response.ok) {
            const body = await response.text();
            return { result: null, error: `${response.status} ${response.statusText}: ${body}` };
        }
        const payload = await readJson<TableDataResponse>(response);
        return { result: payload, error: null };
    } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return { result: null, error: message };
    }
}

