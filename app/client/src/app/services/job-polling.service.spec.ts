import { firstValueFrom, take } from 'rxjs';
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

  it('converts request errors into a terminal failed status', async () => {
    const service = new JobPollingService();
    const status = await firstValueFrom(service.poll(async () => ({ result: null, error: 'offline' }), 'job-2', 0.25));
    expect(status).toMatchObject({ job_id: 'job-2', status: 'failed', error: 'offline' });
  });
});
