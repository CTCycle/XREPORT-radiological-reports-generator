export type JobLifecycleStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface JobStartResponse {
    job_id: string;
    job_type: string;
    status: JobLifecycleStatus;
    message: string;
    poll_interval?: number;
}

export interface JobStatusResponse {
    job_id: string;
    job_type: string;
    status: JobLifecycleStatus;
    progress: number;
    result: Record<string, unknown> | null;
    error: string | null;
}

export interface JobCancelResponse {
    job_id: string;
    success: boolean;
    message: string;
}
