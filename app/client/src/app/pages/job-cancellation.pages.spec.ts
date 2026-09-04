import { TestBed } from '@angular/core/testing';
import { AppStateService } from '../services/app-state.service';
import { DatasetApiService } from '../services/dataset-api.service';
import { GuidanceService } from '../services/guidance.service';
import { InferenceApiService } from '../services/inference-api.service';
import { JobsApiService } from '../services/jobs-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import { InferencePage } from './inference.page';
import { TrainingPage } from './training.page';

type ActiveJobPage = { activeJobId: string | null };

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function activeJob(page: object) {
  return page as ActiveJobPage;
}

describe('cooperative page cancellation', () => {
  it('keeps inference active until polling observes the terminal state', async () => {
    const cancel = jasmine.createSpy('cancel').and.callFake(() => apiResult({ job_id: 'generation-1', success: true, message: 'Cancellation requested' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: GuidanceService, useValue: { requestTour: () => undefined } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true }));
    activeJob(page).activeJobId = 'generation-1';

    await page.cancelGeneration();

    expect(cancel).toHaveBeenCalledOnceWith('generation-1');
    expect(appState.inference().isGenerating).toBeTrue();
    expect(activeJob(page).activeJobId).toBe('generation-1');
    expect(page.progressMessage()).toBe('Cancellation requested…');
  });

  it('keeps inference active and surfaces a rejected cancellation', async () => {
    const cancel = jasmine.createSpy('cancel').and.callFake(() => apiResult({ job_id: 'generation-2', success: false, message: 'Job cannot be cancelled' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: GuidanceService, useValue: { requestTour: () => undefined } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true }));
    activeJob(page).activeJobId = 'generation-2';

    await page.cancelGeneration();

    expect(appState.inference().isGenerating).toBeTrue();
    expect(activeJob(page).activeJobId).toBe('generation-2');
    expect(page.generationError()).toBe('Job cannot be cancelled');
  });

  it('keeps training active until polling observes the terminal state', async () => {
    const cancel = jasmine.createSpy('cancel').and.callFake(() => apiResult({ job_id: 'training-1', success: true, message: 'Cancellation requested' }));
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

    expect(cancel).toHaveBeenCalledOnceWith('training-1');
    expect(appState.training().dashboardState.isTraining).toBeTrue();
    expect(activeJob(page).activeJobId).toBe('training-1');
  });

  it('keeps training active and surfaces a rejected cancellation', async () => {
    const cancel = jasmine.createSpy('cancel').and.callFake(() => apiResult({ job_id: 'training-2', success: false, message: 'Job cannot be cancelled' }));
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

    expect(appState.training().dashboardState.isTraining).toBeTrue();
    expect(activeJob(page).activeJobId).toBe('training-2');
    expect(page.trainingError()).toBe('Job cannot be cancelled');
  });
});
