import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { DatasetApiService } from '../services/dataset-api.service';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import { TrainingPage } from './training.page';

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

describe('TrainingPage worker lifecycle state', () => {
  it('maps an API worker phase into the dashboard while numeric progress is zero', async () => {
    const poll = vi.fn(() => of({
      job_id: 'training-phase',
      job_type: 'training',
      status: 'running' as const,
      poll_interval: 1,
      progress: 0,
      result: {
        current_epoch: 0,
        total_epochs: 1,
        progress_percent: 0,
        worker_phase: 'model_loading_started',
        worker_phase_status: 'started',
        worker_phase_elapsed_seconds: 4,
        worker_elapsed_seconds: 8,
      },
      error: null,
    }));

    await TestBed.configureTestingModule({
      imports: [TrainingPage],
      providers: [
        { provide: DatasetApiService, useValue: { getProcessedNames: () => apiResult({ datasets: [] }) } },
        { provide: TrainingApiService, useValue: { getCheckpoints: () => apiResult({ checkpoints: [] }) } },
        { provide: ValidationApiService, useValue: {} },
        { provide: JobsApiService, useValue: { list: () => apiResult({ jobs: [] }), get: vi.fn() } },
        { provide: JobPollingService, useValue: { poll } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
    (fixture.componentInstance as unknown as { pollTraining: (jobId: string, interval: number) => void }).pollTraining('training-phase', 1);
    await fixture.whenStable();

    const dashboard = fixture.componentInstance.state().dashboardState;
    expect(poll).toHaveBeenCalledWith(expect.any(Function), 'training-phase', 1);
    expect(dashboard.isTraining).toBe(true);
    expect(dashboard.progressPercent).toBe(0);
    expect(dashboard.workerPhase).toBe('model_loading_started');
    expect(dashboard.workerPhaseElapsedSeconds).toBe(4);
    expect(dashboard.workerElapsedSeconds).toBe(8);
  });
});
