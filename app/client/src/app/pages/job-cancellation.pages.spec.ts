import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { AppStateService } from '../services/app-state.service';
import { DatasetApiService } from '../services/dataset-api.service';
import { GuidanceService } from '../services/guidance.service';
import { InferenceApiService } from '../services/inference-api.service';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import type { CheckpointInfo } from '../types/trainingApi';
import type { ModelAvailability } from '../types/inferenceApi';
import { InferencePage } from './inference.page';
import { TrainingPage } from './training.page';

interface ActiveJobPage { activeJobId: string | null }

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function activeJob(page: object) {
  return page as ActiveJobPage;
}

function hiddenGuidance() {
  return {
    requestTour: () => undefined,
    shouldShow: () => false,
  };
}

function inferenceModel(modelRef: string, status: string): ModelAvailability {
  return {
    model_ref: modelRef,
    provider: 'huggingface',
    origin: 'public',
    display_name: 'CXRMate Multi',
    description: 'Test model',
    status,
    enabled: true,
    validation_status: 'pending',
    validation_receipt_status: 'missing',
    category: 'radiology',
    recommended: false,
    research_only: true,
    gated: false,
    access_policy: 'open',
    anatomy_coverage: 'chest_xray',
    hardware_demand: 'moderate',
    input_semantics: 'single_image',
    capabilities: { clinical_context: false, multiple_current_views: false, findings: true, impression: true, grounding: false },
    trust_remote_code: false,
    remote_code_approved: false,
    output_sections: ['findings', 'impression'],
    installation_state: 'staged',
    integrity_status: 'verified',
  } as unknown as ModelAvailability;
}

describe('cooperative page cancellation', () => {
  it('keeps inference active until polling observes the terminal state', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'generation-1', success: true, message: 'Cancellation requested' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true }));
    activeJob(page).activeJobId = 'generation-1';

    await page.cancelGeneration();

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledWith('generation-1');
    expect(appState.inference().isGenerating).toBe(true);
    expect(activeJob(page).activeJobId).toBe('generation-1');
    expect(page.progressMessage()).toBe('Cancellation requested…');
  });

  it('keeps inference active and surfaces a rejected cancellation', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'generation-2', success: false, message: 'Job cannot be cancelled' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true }));
    activeJob(page).activeJobId = 'generation-2';

    await page.cancelGeneration();

    expect(appState.inference().isGenerating).toBe(true);
    expect(activeJob(page).activeJobId).toBe('generation-2');
    expect(page.generationError()).toBe('Job cannot be cancelled');
  });

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

describe('inference model readiness refresh', () => {
  it('refreshes the selected model after generation reaches a terminal state', async () => {
    const modelRef = 'huggingface:aehrc/cxrmate-multi-tf';
    const getModels = vi.fn()
      .mockResolvedValueOnce(apiResult({ models: [inferenceModel(modelRef, 'unvalidated')] }))
      .mockResolvedValueOnce(apiResult({ models: [inferenceModel(modelRef, 'ready')] }));
    const generateReports = vi.fn(() => apiResult({ job_id: 'generation-1', poll_interval: 0.25 }));
    const poll = vi.fn(() => of({
      job_id: 'generation-1',
      job_type: 'inference',
      status: 'completed',
      poll_interval: 0.25,
      progress: 100,
      result: null,
      error: null,
    }));

    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels, generateReports } },
        { provide: JobsApiService, useValue: { get: vi.fn() } },
        { provide: JobPollingService, useValue: { poll } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const state = TestBed.inject(AppStateService);
    state.updateInference((current) => ({
      ...current,
      selectedModelRef: modelRef,
      images: [new File(['image'], 'fixture.png', { type: 'image/png' })],
    }));

    await fixture.componentInstance.generate();
    await fixture.whenStable();

    expect(getModels).toHaveBeenCalledTimes(2);
    expect(state.inference().selectedModelRef).toBe(modelRef);
    expect(state.inference().modelAvailability[0]?.status).toBe('ready');
    expect(state.inference().isGenerating).toBe(false);
  });

  it('identifies a cancelled generation in its alert', async () => {
    const modelRef = 'huggingface:aehrc/cxrmate-multi-tf';
    const getModels = vi.fn()
      .mockResolvedValueOnce(apiResult({ models: [inferenceModel(modelRef, 'unvalidated')] }))
      .mockResolvedValueOnce(apiResult({ models: [inferenceModel(modelRef, 'ready')] }));
    const generateReports = vi.fn(() => apiResult({ job_id: 'generation-cancelled', poll_interval: 0.25 }));
    const poll = vi.fn(() => of({
      job_id: 'generation-cancelled',
      job_type: 'inference',
      status: 'cancelled',
      poll_interval: 0.25,
      progress: 100,
      result: null,
      error: null,
    }));

    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels, generateReports } },
        { provide: JobsApiService, useValue: { get: vi.fn() } },
        { provide: JobPollingService, useValue: { poll } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const state = TestBed.inject(AppStateService);
    state.updateInference((current) => ({
      ...current,
      selectedModelRef: modelRef,
      images: [new File(['image'], 'fixture.png', { type: 'image/png' })],
    }));

    await fixture.componentInstance.generate();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('.generation-error[role="alert"]')?.textContent).toContain('Generation cancelled');
    expect(fixture.nativeElement.querySelector('.generation-error[role="alert"]')?.textContent).toContain('Generation cancelled.');
    expect(fixture.componentInstance.generationCancelled()).toBe(true);
    expect(state.inference().isGenerating).toBe(false);
    expect(getModels).toHaveBeenCalledTimes(2);
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
