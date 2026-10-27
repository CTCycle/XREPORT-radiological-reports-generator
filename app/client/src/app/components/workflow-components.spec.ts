import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { DatasetApiService } from '../services/dataset-api.service';
import { JobsApiService } from '../services/jobs-api.service';
import { TrainingApiService } from '../services/training-api.service';
import type { TrainingDashboardState } from '../types';
import { DatasetTrainingJourneyComponent } from './dataset-training-journey.component';
import { TrainingDashboardComponent } from './training-dashboard.component';

@Component({
  standalone: true,
  imports: [DatasetTrainingJourneyComponent],
  template: '<app-dataset-training-journey (routeRequested)="route = $event" (walkthroughRequested)="walkthroughStarted = true" />',
})
class JourneyHostComponent {
  route = '';
  walkthroughStarted = false;
}

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function emptyDatasetApi() {
  return {
    getStatus: () => apiResult({ has_data: false, row_count: 0, message: 'No data' }),
    getNames: () => apiResult({ datasets: [], count: 0 }),
    getProcessedNames: () => apiResult({ datasets: [], count: 0 }),
  };
}

function emptyTrainingApi() {
  return {
    getCheckpoints: () => apiResult({ checkpoints: [] }),
  };
}

function emptyJobsApi() {
  return { list: () => apiResult({ jobs: [] }) };
}

describe('DatasetTrainingJourneyComponent', () => {
  it('starts at source upload and keeps later actions locked when no data exists', async () => {
    await TestBed.configureTestingModule({
      imports: [JourneyHostComponent],
      providers: [{ provide: DatasetApiService, useValue: emptyDatasetApi() }, { provide: TrainingApiService, useValue: emptyTrainingApi() }, { provide: JobsApiService, useValue: emptyJobsApi() }],
    }).compileComponents();

    const fixture = TestBed.createComponent(JourneyHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Next: upload and load your source data');
    expect(fixture.nativeElement.querySelectorAll('.guidance-journey-step')).toHaveLength(5);
    expect(fixture.nativeElement.querySelector('.guidance-journey-step.is-current')).toBeTruthy();

    const steps = fixture.nativeElement.querySelectorAll('.guidance-journey-step-toggle');
    (steps[2] as HTMLButtonElement).click();
    fixture.detectChanges();
    const lockedAction = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Complete previous step')) as HTMLButtonElement;
    expect(lockedAction.disabled).toBe(true);

    const walkthrough = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Walk me through it')) as HTMLButtonElement;
    walkthrough.click();
    expect(fixture.componentInstance.walkthroughStarted).toBe(true);
  });

  it('recognizes a processed dataset and exposes the checkpoint branch', async () => {
    const datasetApi = {
      getStatus: () => apiResult({ has_data: true, row_count: 12, message: 'Found 12 records' }),
      getNames: () => apiResult({ datasets: [{ name: 'source', folder_path: 'images', row_count: 12, has_validation_report: false }], count: 1 }),
      getProcessedNames: () => apiResult({ datasets: [{ name: 'source', folder_path: 'processed', row_count: 10, has_validation_report: false }], count: 1 }),
    };
    const trainingApi = {
      getCheckpoints: () => apiResult({ checkpoints: [{ name: 'source-epoch-4', epochs: 4, loss: 0.2, val_loss: 0.3 }] }),
    };
    const jobsApi = { list: () => apiResult({ jobs: [] }) };

    await TestBed.configureTestingModule({
      imports: [JourneyHostComponent],
      providers: [{ provide: DatasetApiService, useValue: datasetApi }, { provide: TrainingApiService, useValue: trainingApi }, { provide: JobsApiService, useValue: jobsApi }],
    }).compileComponents();

    const fixture = TestBed.createComponent(JourneyHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Your checkpoint branch is ready to resume');
    expect(fixture.nativeElement.querySelectorAll('.guidance-journey-step.is-completed')).toHaveLength(4);
    expect(fixture.nativeElement.querySelector('.guidance-journey-step.is-available')).toBeTruthy();

    const openCheckpoint = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Open Checkpoints')) as HTMLButtonElement;
    openCheckpoint.click();
    expect(fixture.componentInstance.route).toBe('/training');
  });

  it('shows a short unavailable state with locked steps when a status call fails', async () => {
    const datasetApi = { ...emptyDatasetApi(), getStatus: () => Promise.resolve({ result: null, error: '0 Unknown Error: {}' }) };

    await TestBed.configureTestingModule({
      imports: [JourneyHostComponent],
      providers: [{ provide: DatasetApiService, useValue: datasetApi }, { provide: TrainingApiService, useValue: emptyTrainingApi() }, { provide: JobsApiService, useValue: emptyJobsApi() }],
    }).compileComponents();

    const fixture = TestBed.createComponent(JourneyHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Live status unavailable');
    expect(text).not.toContain('Unknown Error');
    expect(fixture.nativeElement.querySelector('.guidance-journey-step.is-current')).toBeTruthy();
  });
});

const dashboardState = (): TrainingDashboardState => ({
  isTraining: false,
  currentEpoch: 0,
  totalEpochs: 1,
  loss: 0,
  valLoss: 0,
  accuracy: 0,
  valAccuracy: 0,
  progressPercent: 0,
  elapsedSeconds: 0,
  workerPhase: 'idle',
  workerPhaseStatus: 'idle',
  workerPhaseElapsedSeconds: 0,
  workerElapsedSeconds: 0,
  chartData: [],
  availableMetrics: [],
  epochBoundaries: [],
  logEntries: [],
});

@Component({
  standalone: true,
  imports: [TrainingDashboardComponent],
  template: '<app-training-dashboard [dashboardState]="state" [error]="error" />',
})
class DashboardHostComponent {
  state = dashboardState();
  error: string | null = null;
}

describe('TrainingDashboardComponent', () => {
  it('shows unavailable metrics as placeholders rather than zeros while idle and keeps stop disabled', async () => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Waiting to start');
    expect(text).toContain('Idle');
    expect(text).not.toContain('0.0000');
    expect(fixture.nativeElement.querySelectorAll('.metric-value.unavailable').length).toBe(5);
    expect(fixture.nativeElement.querySelector('[role="progressbar"]').getAttribute('aria-valuenow')).toBe('0');
    expect((fixture.nativeElement.querySelector('.btn-stop') as HTMLButtonElement).disabled).toBe(true);
  });

  it.each([
    ['pending', 'Initializing'],
    ['failed', 'Failed'],
    ['cancelled', 'Interrupted'],
    ['completed', 'Completed'],
  ] as const)('labels a %s job as %s', async (sessionStatus, label) => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.componentInstance.state = { ...dashboardState(), sessionStatus, isTraining: sessionStatus === 'pending' };
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.dashboard-status').textContent).toContain(label);
  });

  it('renders real zero-valued metrics once training data exists and plots both series', async () => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.componentInstance.state = {
      ...dashboardState(),
      isTraining: true,
      sessionStatus: 'running',
      currentEpoch: 1,
      totalEpochs: 3,
      loss: 0,
      accuracy: 0.5,
      chartData: [{ batch: 1, loss: 2, val_loss: 2.5 }, { batch: 2, loss: 1.5 }, { batch: 3, loss: 1.2, val_loss: 1.4 }],
      availableMetrics: ['loss', 'val_loss'],
      logEntries: ['10:00:01 Epoch started', 'WARNING: slow dataloader', 'Training failed: out of memory'],
    };
    fixture.detectChanges();

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('0.0000');
    expect(text).toContain('50.00%');
    expect(fixture.nativeElement.querySelectorAll('.chart-series-line').length).toBe(2);
    expect(fixture.nativeElement.querySelector('.log-line')).toBeNull();
    fixture.nativeElement.querySelector('.log-toggle').click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.log-line.warning')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.log-line.error')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.log-time').textContent).toBe('10:00:01');
  });

  it('renders the active worker phase and a terminal error accessibly', async () => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.componentInstance.state = {
      ...dashboardState(),
      isTraining: true,
      workerPhase: 'model_loading_started',
      workerPhaseStatus: 'started',
      workerPhaseElapsedSeconds: 4,
    };
    fixture.componentInstance.error = 'Training worker stalled during model loading';
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Loading model');
    expect(fixture.nativeElement.textContent).toContain('4s in phase');
    expect(fixture.nativeElement.querySelector('[role="alert"]')?.textContent).toContain('Training worker stalled');
    expect((fixture.nativeElement.querySelector('.btn-stop') as HTMLButtonElement).disabled).toBe(false);
  });
});
