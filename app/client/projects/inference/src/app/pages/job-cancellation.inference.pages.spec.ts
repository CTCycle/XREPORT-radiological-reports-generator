import { TestBed } from '@angular/core/testing';
import { NEVER, of } from 'rxjs';
import { vi } from 'vitest';
import { AppStateService } from '@xreport/shared/app/services/app-state.service';
import { GuidanceService } from '@xreport/shared/app/services/guidance.service';
import { InferenceApiService } from '@xreport/shared/app/services/inference-api.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import type { ModelAvailability } from '@xreport/shared/app/types/inferenceApi';
import { InferencePage } from './inference.page';

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function hiddenGuidance() {
  return {
    requestTour: () => undefined,
    shouldShow: () => false,
  };
}

function inferenceModel(modelRef: string, status: string, validationStatus: ModelAvailability['validation_status'] = 'pending'): ModelAvailability {
  return {
    model_ref: modelRef,
    provider: 'huggingface',
    origin: 'public',
    display_name: 'CXRMate Multi',
    description: 'Test model',
    status,
    enabled: true,
    validation_status: validationStatus,
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

function customInferenceModel(modelRef: string): ModelAvailability {
  return {
    ...inferenceModel(modelRef, 'ready', 'passed'),
    display_name: modelRef,
    origin: 'custom',
    validation_receipt_status: 'passed',
  } as ModelAvailability;
}

describe('cooperative page cancellation', () => {
  it('keeps inference active until polling observes the terminal state', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'generation-1', success: true, message: 'Cancellation requested' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: JobPollingService, useValue: { poll: () => NEVER } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true, activeJobId: 'generation-1' }));
    await fixture.whenStable();

    await page.cancelGeneration();

    expect(cancel).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledWith('generation-1');
    expect(appState.inference().isGenerating).toBe(true);
    expect(appState.inference().activeJobId).toBe('generation-1');
    expect(page.progressMessage()).toBe('Cancellation requested…');
  });

  it('keeps inference active and surfaces a rejected cancellation', async () => {
    const cancel = vi.fn(() => apiResult({ job_id: 'generation-2', success: false, message: 'Job cannot be cancelled' }));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [] }) } },
        { provide: JobsApiService, useValue: { cancel } },
        { provide: JobPollingService, useValue: { poll: () => NEVER } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    const page = fixture.componentInstance;
    const appState = TestBed.inject(AppStateService);
    appState.updateInference((state) => ({ ...state, isGenerating: true, activeJobId: 'generation-2' }));
    await fixture.whenStable();

    await page.cancelGeneration();

    expect(appState.inference().isGenerating).toBe(true);
    expect(appState.inference().activeJobId).toBe('generation-2');
    expect(page.generationError()).toBe('Job cannot be cancelled');
  });
});

describe('inference model readiness refresh', () => {
  it('keeps a large model catalogue in a bounded scroll region', async () => {
    const models = Array.from({ length: 40 }, (_, index) => customInferenceModel(`xreport:checkpoint-${index + 1}`));
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models }) } },
        { provide: JobsApiService, useValue: { get: vi.fn() } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();
    fixture.detectChanges();

    const scrollRegion = fixture.nativeElement.querySelector('[aria-label="Scrollable model catalogue"]') as HTMLElement | null;
    expect(scrollRegion).not.toBeNull();
    expect(scrollRegion?.classList.contains('model-groups')).toBe(true);
    expect(scrollRegion?.getAttribute('tabindex')).toBe('0');
    expect(fixture.nativeElement.querySelectorAll('.model-card')).toHaveLength(40);
    expect(fixture.nativeElement.textContent).toContain('40 models');
  });

  it('does not offer generation for a degraded public model', async () => {
    const modelRef = 'huggingface:future/degraded-model';
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [inferenceModel(modelRef, 'ready', 'degraded')] }) } },
        { provide: JobsApiService, useValue: { get: vi.fn() } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();

    expect(fixture.componentInstance.canGenerate()).toBe(false);
    expect(fixture.componentInstance.modelNotice()).toContain('failed qualification');
  });

  it('does not offer generation for a gated public model that is not ready', async () => {
    const modelRef = 'huggingface:google/medgemma-1.5-4b-it';
    const model = inferenceModel(modelRef, 'not_installed');
    model.gated = true;
    model.access_policy = 'gated';
    await TestBed.configureTestingModule({
      imports: [InferencePage],
      providers: [
        { provide: InferenceApiService, useValue: { getModels: () => apiResult({ models: [model] }) } },
        { provide: JobsApiService, useValue: { get: vi.fn() } },
        { provide: GuidanceService, useValue: hiddenGuidance() },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InferencePage);
    await fixture.whenStable();

    expect(fixture.componentInstance.canGenerate()).toBe(false);
    expect(fixture.componentInstance.modelNotice()).toContain('Access is required');
  });

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
