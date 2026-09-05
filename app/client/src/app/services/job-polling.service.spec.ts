import { firstValueFrom, take, toArray } from 'rxjs';
import { JobPollingService } from './job-polling.service';

describe('JobPollingService', () => {
  it('emits terminal status and completes without another poll', async () => {
    const service = new JobPollingService();
    let calls = 0;
    const statuses = await firstValueFrom(service.poll(async () => {
      calls += 1;
      return { result: { job_id: 'job-1', job_type: 'test', status: 'completed' as const, progress: 100, result: { ok: true }, error: null }, error: null };
    }, 'job-1', 0.25).pipe(take(1)));
    expect(statuses.status).toBe('completed');
    expect(calls).toBe(1);
  });

  it('recovers from a transient request error without terminating the job', async () => {
    const service = new JobPollingService();
    let calls = 0;
    const statuses = await firstValueFrom(service.poll(async () => {
      calls += 1;
      if (calls === 1) return { result: null, error: 'offline' };
      return { result: { job_id: 'job-2', job_type: 'test', status: 'completed' as const, progress: 100, result: { ok: true }, error: null }, error: null };
    }, 'job-2', 0.25).pipe(toArray()));

    expect(statuses.map((status) => status.status)).toEqual(['pending', 'completed']);
    expect(statuses[0].error).toBe('offline');
    expect(calls).toBe(2);
  });

  it('keeps polling through repeated transport failures', async () => {
    const service = new JobPollingService();
    let calls = 0;
    const statuses = await firstValueFrom(service.poll(async () => {
      calls += 1;
      return { result: null, error: 'offline' };
    }, 'job-3', 0.25).pipe(take(3), toArray()));

    expect(statuses.map((status) => status.status)).toEqual(['pending', 'pending', 'pending']);
    expect(statuses.at(-1)).toMatchObject({ job_id: 'job-3', status: 'pending', error: 'offline' });
    expect(calls).toBe(3);
  });

  it('terminates when the backend explicitly reports that the job is missing', async () => {
    const service = new JobPollingService();
    let calls = 0;
    const statuses = await firstValueFrom(service.poll(async () => {
      calls += 1;
      return { result: null, error: '404 Not Found: Job not found' };
    }, 'job-4', 0.25).pipe(toArray()));

    expect(statuses).toHaveLength(1);
    expect(statuses[0]).toMatchObject({ job_id: 'job-4', status: 'failed', error: '404 Not Found: Job not found' });
    expect(calls).toBe(1);
  });
});
