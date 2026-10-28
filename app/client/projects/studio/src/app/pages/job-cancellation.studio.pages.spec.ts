import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { AppStateService } from '@xreport/shared/app/services/app-state.service';
import { DatasetApiService } from '../services/dataset-api.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import type { CheckpointInfo } from '@xreport/shared/app/types/trainingApi';
import { TrainingPage } from './training.page';

interface ActiveJobPage { activeJobId: string | null }

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function activeJob(page: object) {
  return page as ActiveJobPage;
}

describe('cooperative page cancellation', () => {
  it('keeps training active until polling observes the terminal state', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'training-1', success: true, message: 'Cancellation requested' }));
    await TestBed.configureTestingModule({
      imports: [TrainingPage],
      providers: [
        { provide: DatasetApiService, useValue: { getProcessedNames: () => apiResult({ datasets: [], count: 0 }) } },
        { provide: TrainingApiService, useValue: { getCheckpoints: () => apiResult({ checkpoints: [] }) } },
        { provide: ValidationApiService, useValue: {} },
        { provide: JobsApiService, useValue: { list: () => apiResult({ jobs: [] }), cancel } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateDashboard({ isTraining: true });
    activeJob(page).activeJobId = 'training-1';

    await page.stopTraining();

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledWith('training-1');
    expect(appState.training().dashboardState.isTraining).toBe(true);
    expect(activeJob(page).activeJobId).toBe('training-1');
  });

  it('reports a cancelled training run without presenting it as an error', async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingPage],
      providers: [
        { provide: DatasetApiService, useValue: { getProcessedNames: () => apiResult({ datasets: [], count: 0 }) } },
        { provide: TrainingApiService, useValue: { getCheckpoints: () => apiResult({ checkpoints: [] }) } },
        { provide: ValidationApiService, useValue: {} },
        { provide: JobsApiService, useValue: { list: () => apiResult({ jobs: [] }) } },
        {
          provide: JobPollingService,
          useValue: {
            poll: () => of({ job_id: 'training-cancelled', status: 'cancelled', result: null, error: null, progress: 0 }),
          },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    (page as unknown as { pollTraining(jobId: string, pollInterval: number): void }).pollTraining('training-cancelled', 0.25);

    expect(appState.training().dashboardState.isTraining).toBe(false);
    expect(appState.training().dashboardState.logEntries.at(-1)).toBe('Training cancelled.');
    expect(page.trainingError()).toBeNull();
    expect(activeJob(page).activeJobId).toBeNull();
  });

  it('keeps training active and surfaces a rejected cancellation', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'training-2', success: false, message: 'Job cannot be cancelled' }));
    await TestBed.configureTestingModule({
      imports: [TrainingPage],
      providers: [
        { provide: DatasetApiService, useValue: { getProcessedNames: () => apiResult({ datasets: [], count: 0 }) } },
        { provide: TrainingApiService, useValue: { getCheckpoints: () => apiResult({ checkpoints: [] }) } },
        { provide: ValidationApiService, useValue: {} },
        { provide: JobsApiService, useValue: { list: () => apiResult({ jobs: [] }), cancel } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateDashboard({ isTraining: true });
    activeJob(page).activeJobId = 'training-2';

    await page.stopTraining();

    expect(appState.training().dashboardState.isTraining).toBe(true);
    expect(activeJob(page).activeJobId).toBe('training-2');
    expect(page.trainingError()).toBe('Job cannot be cancelled');
  });
});

describe('checkpoint evaluation configuration', () => {
  it('lets the backend apply the configured seed when the UI does not override it', async () => {
    const evaluateCheckpoint = vi.fn(() => Promise.resolve({ result: null, error: 'stop after request capture' }));
    await TestBed.configureTestingModule({
      imports: [TrainingPage],
      providers: [
        { provide: DatasetApiService, useValue: { getProcessedNames: () => apiResult({ datasets: [], count: 0 }) } },
        { provide: TrainingApiService, useValue: { getCheckpoints: () => apiResult({ checkpoints: [] }) } },
        { provide: ValidationApiService, useValue: { evaluateCheckpoint } },
        { provide: JobsApiService, useValue: { list: () => apiResult({ jobs: [] }) } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(TrainingPage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    page.evaluationCheckpoint.set({ name: 'checkpoint-1' } as CheckpointInfo);

    await page.runEvaluation({
      metrics: ['evaluation_report'],
      metricConfigs: { evaluation_report: { dataFraction: 1 } },
    });

    expect(evaluateCheckpoint).toHaveBeenCalledTimes(1);
    expect(evaluateCheckpoint).toHaveBeenCalledWith(
      'checkpoint-1',
      ['evaluation_report'],
      10,
      { evaluation_report: { data_fraction: 1 } },
    );
  });
});

