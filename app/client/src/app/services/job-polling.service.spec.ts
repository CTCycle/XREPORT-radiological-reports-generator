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

  it('fails locally after three consecutive status request failures', async () => {
    const service = new JobPollingService();
    let calls = 0;
    const statuses = await firstValueFrom(service.poll(async () => {
      calls += 1;
      return { result: null, error: 'offline' };
    }, 'job-3', 0.25).pipe(toArray()));

    expect(statuses.map((status) => status.status)).toEqual(['pending', 'pending', 'failed']);
    expect(statuses.at(-1)).toMatchObject({ job_id: 'job-3', status: 'failed', error: 'offline' });
    expect(calls).toBe(3);
  });
});
