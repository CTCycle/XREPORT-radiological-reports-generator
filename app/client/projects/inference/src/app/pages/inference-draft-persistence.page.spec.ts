import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { NEVER, Observable, of } from 'rxjs';
import { vi } from 'vitest';
import { AppStateService } from '@xreport/shared/app/services/app-state.service';
import { GuidanceService } from '@xreport/shared/app/services/guidance.service';
import { InferenceApiService } from '@xreport/shared/app/services/inference-api.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import type { InferenceHistoryDetail, ModelAvailability } from '@xreport/shared/app/types/inferenceApi';
import { InferencePage } from './inference.page';

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function report(index: number, imageName: string, findings: string, impression: string) {
  const text = `Findings\n${findings}\n\nImpression\n${impression}`;
  return {
    image_index: index,
    input_image_name: imageName,
    generated_report: text,
    edited_report: null,
    effective_report: text,
    edited: false,
    edited_at: null,
    sections: { findings, impression },
  };
}

function savedSession(requestId: string, studyLevel = false, reportCount = 2): InferenceHistoryDetail {
  return {
    request_id: requestId,
    provider: 'huggingface',
    model_ref: 'huggingface:e2e/draft-model',
    model_revision: 'e2e-revision',
    generation_profile: 'deterministic',
    generation_config: { provenance: { report_scope: studyLevel ? 'study' : 'image' } },
    clinical_context: null,
    status: 'succeeded',
    execution_time_seconds: 0.1,
    date: '2026-10-23T10:00:00Z',
    reports: studyLevel
      ? [report(0, 'study.png', 'Clear lungs.', 'No acute disease.')]
      : [
          report(0, 'view-one.png', 'Clear lungs.', 'No acute disease.'),
          report(1, 'view-two.png', 'Mild basilar opacity.', 'Follow-up recommended.'),
        ].slice(0, reportCount),
    output_sections: ['findings', 'impression'],
  };
}

function generationResult(requestId: string, session: InferenceHistoryDetail, studyLevel = false) {
  const rows = session.reports;
  return {
    request_id: requestId,
    reports: Object.fromEntries(rows.map((row) => [row.input_image_name, row.generated_report])),
    reports_ordered: rows.map((row) => row.generated_report),
    report_filenames: rows.map((row) => row.input_image_name),
    count: rows.length,
    display_sections: {},
    provenance: studyLevel ? { report_scope: 'study' } : {},
  };
}

function updatedSession(session: InferenceHistoryDetail, reports: { image_index: number; edited_report: string }[]): InferenceHistoryDetail {
  const updates = new Map(reports.map((value) => [value.image_index, value.edited_report]));
  return {
    ...session,
    reports: session.reports.map((saved) => {
      const edited = updates.get(saved.image_index);
      return edited === undefined
        ? saved
        : { ...saved, edited_report: edited, effective_report: edited, edited: edited !== saved.generated_report, edited_at: '2026-10-23T10:01:00Z' };
    }),
  };
}

const modelRef = 'huggingface:e2e/draft-model';
const model: ModelAvailability = {
  model_ref: modelRef,
  provider: 'huggingface',
  origin: 'public',
  display_name: 'E2E Draft Model',
  description: 'Test model',
  status: 'ready',
  enabled: true,
  validation_status: 'passed',
  validation_receipt_status: 'passed',
  category: 'radiology',
  recommended: false,
  research_only: true,
  gated: false,
  access_policy: 'open',
  anatomy_coverage: 'chest_xray',
  hardware_demand: 'moderate',
  input_semantics: 'multiple_current_views',
  capabilities: { clinical_context: false, multiple_current_views: true, findings: true, impression: true, grounding: false },
  trust_remote_code: false,
  remote_code_approved: false,
  output_sections: ['findings', 'impression'],
  max_current_images: 2,
  installation_state: 'installed',
  integrity_status: 'verified',
} as unknown as ModelAvailability;

async function createInferencePage(options: { imageCount?: number; studyLevel?: boolean; historyError?: string; updateResults?: { result: InferenceHistoryDetail | null; error: string | null }[] } = {}) {
  const requestId = 'persisted-e2e-request';
  const imageCount = options.studyLevel ? 2 : options.imageCount ?? 2;
  let session = savedSession(requestId, options.studyLevel, imageCount);
  const updateHistory = vi.fn(async (_id: string, request: { reports: { image_index: number; edited_report: string }[] }) => {
    const next = options.updateResults?.shift();
    if (next) {
      if (next.result) session = next.result;
      return next;
    }
    session = updatedSession(session, request.reports);
    return apiResult(session);
  });
  const api = {
    getModels: () => apiResult({ models: [model], providers: {} }),
    generateReports: () => apiResult({ job_id: 'generation-job', poll_interval: 0.1 }),
    getHistory: vi.fn(() => options.historyError ? Promise.resolve({ result: null, error: options.historyError }) : apiResult(session)),
    updateHistory,
  };

  await TestBed.configureTestingModule({
    imports: [InferencePage],
    providers: [
      provideRouter([]),
      { provide: InferenceApiService, useValue: api },
      { provide: JobsApiService, useValue: { get: vi.fn(), cancel: vi.fn() } },
      { provide: JobPollingService, useValue: { poll: () => of({ job_id: 'generation-job', job_type: 'inference', status: 'completed', poll_interval: 0.1, progress: 100, result: generationResult(requestId, session, options.studyLevel), error: null }) } },
      { provide: GuidanceService, useValue: { requestTour: vi.fn(), shouldShow: () => false } },
    ],
  }).compileComponents();

  const fixture = TestBed.createComponent(InferencePage);
  await fixture.whenStable();
  const appState = TestBed.inject(AppStateService);
  appState.updateInference((state) => ({
    ...state,
    selectedModelRef: modelRef,
    runScope: options.studyLevel ? 'study' : 'all',
    images: Array.from({ length: imageCount }, (_, index) =>
      new File([`image ${index + 1}`], `view-${index + 1}.png`, { type: 'image/png' }),
    ),
  }));
  await fixture.componentInstance.generate();
  await fixture.whenStable();
  fixture.detectChanges();
  return { fixture, page: fixture.componentInstance, api, updateHistory, appState, requestId };
}

describe('inference draft persistence', () => {
  it('keeps saving unavailable until the persisted report is confirmed', async () => {
    const { fixture, page, updateHistory } = await createInferencePage({ historyError: 'History is temporarily unavailable.' });

    expect(page.persistedRequestId()).toBe('persisted-e2e-request');
    expect(page.persistedHistory()).toBeNull();
    expect(page.saveError()).toContain('History is temporarily unavailable.');
    expect((fixture.nativeElement.querySelector('button.primary-button') as HTMLButtonElement).disabled).toBe(true);

    await page.saveEdits();

    expect(updateHistory).not.toHaveBeenCalled();
  });

  it('loads the saved baseline and does not write unchanged reports', async () => {
    const { page, updateHistory } = await createInferencePage({ imageCount: 1 });

    expect(page.persistedRequestId()).toBe('persisted-e2e-request');
    expect(page.dirty()).toBe(false);
    await page.saveEdits();

    expect(updateHistory).not.toHaveBeenCalled();
  });

  it('saves only changed canonical reports and keeps generated output intact', async () => {
    const { page, updateHistory } = await createInferencePage();
    page.updateDraft('findings', 'Clear lungs after review.');

    await page.saveEdits();

    expect(updateHistory).toHaveBeenCalledWith('persisted-e2e-request', {
      reports: [{ image_index: 0, edited_report: 'Findings\nClear lungs after review.\n\nImpression\nNo acute disease.' }],
    });
    expect(page.dirty()).toBe(false);
    expect(page.saveSuccess()).toBe(true);
    expect(page.persistedHistory()?.reports[0]?.generated_report).toBe(
      'Findings\nClear lungs.\n\nImpression\nNo acute disease.',
    );
  });

  it('retains failed edits and allows a retry', async () => {
    const retrySession = updatedSession(savedSession('persisted-e2e-request'), [
      { image_index: 0, edited_report: 'Findings\nReviewed lungs.\n\nImpression\nNo acute disease.' },
    ]);
    const { page, updateHistory } = await createInferencePage({
      updateResults: [
        { result: null, error: '503 database unavailable' },
        { result: retrySession, error: null },
      ],
    });
    page.updateDraft('findings', 'Reviewed lungs.');

    await page.saveEdits();

    expect(page.dirty()).toBe(true);
    expect(page.draft('findings')).toBe('Reviewed lungs.');
    expect(page.saveError()).toContain('503 database unavailable');
    expect(page.saveSuccess()).toBe(false);

    await page.saveEdits();

    expect(page.dirty()).toBe(false);
    expect(page.saveSuccess()).toBe(true);
    expect(updateHistory).toHaveBeenCalledTimes(2);
  });

  it('maps a study-level editor shared across images to its single persisted row', async () => {
    const { page, updateHistory, appState } = await createInferencePage({ studyLevel: true });
    appState.updateInference((state) => ({ ...state, currentIndex: 1 }));
    page.updateDraft('impression', 'Reviewed study is stable.');

    expect(page.dirty()).toBe(true);
    await page.saveEdits();

    expect(updateHistory).toHaveBeenCalledWith('persisted-e2e-request', {
      reports: [{ image_index: 0, edited_report: 'Findings\nClear lungs.\n\nImpression\nReviewed study is stable.' }],
    });
    expect(page.dirty()).toBe(false);
  });

  it('navigates per-image reports and saves each one independently', async () => {
    const { fixture, page, updateHistory } = await createInferencePage();
    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('.report-navigator')).not.toBeNull();
    expect(page.reportEntries().map((entry) => entry.name)).toEqual(['view-1.png', 'view-2.png']);

    page.updateDraft('findings', 'Edited first.');
    page.stepReport(1);
    expect(page.state().currentIndex).toBe(1);
    expect(page.draft('findings')).toBe('Mild basilar opacity.');
    page.updateDraft('impression', 'Edited second.');
    page.stepReport(-1);
    expect(page.draft('findings')).toBe('Edited first.');

    await page.saveEdits();

    expect(updateHistory).toHaveBeenCalledWith('persisted-e2e-request', {
      reports: [{ image_index: 0, edited_report: 'Findings\nEdited first.\n\nImpression\nNo acute disease.' }],
    });
    expect(page.currentDirty()).toBe(false);
    expect(page.dirty()).toBe(true);
    page.stepReport(1);
    expect(page.draft('impression')).toBe('Edited second.');
    expect(page.currentDirty()).toBe(true);
  });

  it('keeps the generation profile editable after a draft and falls back from an unsupported study scope', async () => {
    const { fixture, page, api, appState } = await createInferencePage();
    const profile = fixture.nativeElement.querySelector('#profile-select') as HTMLSelectElement;
    expect(page.hasDraft()).toBe(true);
    expect(profile.disabled).toBe(false);

    const generateReports = vi.spyOn(api, 'generateReports');
    appState.updateInference((state) => ({
      ...state,
      runScope: 'study',
      generationProfile: 'detailed',
      modelAvailability: [{ ...model, capabilities: { ...model.capabilities, multiple_current_views: false } }],
    }));
    expect(page.effectiveScope()).toBe('all');

    await page.generate();

    expect(generateReports).toHaveBeenCalledWith(expect.anything(), modelRef, 'detailed', expect.anything(), 'per_image');
  });

  it('keeps unsaved edits across page navigation and still guards browser unload', async () => {
    const { fixture, page, api } = await createInferencePage();
    page.updateDraft('findings', 'Unsaved finding.');
    const confirm = vi.spyOn(window, 'confirm');

    expect(page.confirmDiscardUnsavedChanges()).toBe(true);
    expect(confirm).not.toHaveBeenCalled();
    const event = new Event('beforeunload', { cancelable: true }) as BeforeUnloadEvent;
    page.protectUnsavedDraft(event);
    expect(event.defaultPrevented).toBe(true);

    fixture.destroy();
    const returned = TestBed.createComponent(InferencePage);
    await returned.whenStable();
    expect(returned.componentInstance.draft('findings')).toBe('Unsaved finding.');
    expect(returned.componentInstance.dirty()).toBe(true);
    expect(api.getHistory).toHaveBeenCalledOnce();

    confirm.mockRestore();
  });

  it('keeps an independent study per model across switches until it is cleared', async () => {
    const { page, appState, api, requestId } = await createInferencePage();
    const other = { ...model, model_ref: 'huggingface:e2e/other-model', display_name: 'Other model' } as ModelAvailability;
    appState.updateInference((state) => ({ ...state, modelAvailability: [model, other] }));
    page.updateDraft('findings', 'Unsaved finding.');
    page.setIndex(1);
    const firstImages = page.state().images.map((image) => image.name);

    page.selectModel(other);
    expect(page.state().selectedModelRef).toBe(other.model_ref);
    expect(page.state().images).toEqual([]);
    expect(page.hasDraft()).toBe(false);
    expect(page.persistedHistory()).toBeNull();
    page.addFiles([new File(['other'], 'other.png', { type: 'image/png' })] as unknown as FileList);

    page.selectModel(model);
    expect(page.state().images.map((image) => image.name)).toEqual(firstImages);
    expect(page.state().currentIndex).toBe(1);
    expect(page.persistedHistory()?.request_id).toBe(requestId);
    page.setIndex(0);
    expect(page.draft('findings')).toBe('Unsaved finding.');
    expect(page.dirty()).toBe(true);
    expect(api.getHistory).toHaveBeenCalledOnce();

    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    page.resetStudy();
    expect(page.state().images).toEqual([]);
    expect(page.hasDraft()).toBe(false);
    page.selectModel(other);
    expect(page.state().images.map((image) => image.name)).toEqual(['other.png']);
    confirm.mockRestore();
  });

  it('confirms before study replacement or regeneration discards a dirty draft', async () => {
    const { page, api } = await createInferencePage();
    page.updateDraft('findings', 'Unsaved finding.');
    const originalImages = page.state().images.map((image) => image.name);
    const generateReports = vi.spyOn(api, 'generateReports');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);

    page.addFiles([new File(['replacement'], 'replacement.png', { type: 'image/png' })] as unknown as FileList);
    page.resetStudy();
    await page.generate();

    expect(confirm).toHaveBeenCalledTimes(3);
    expect(page.dirty()).toBe(true);
    expect(page.draft('findings')).toBe('Unsaved finding.');
    expect(page.state().selectedModelRef).toBe(modelRef);
    expect(page.state().images.map((image) => image.name)).toEqual(originalImages);
    expect(generateReports).not.toHaveBeenCalled();

    confirm.mockRestore();
  });

  describe('navigation and batch runs', () => {
    const urlStub = { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() };

    async function createPage(options: { state?: Partial<ReturnType<AppStateService['inference']>>; poll?: () => Observable<unknown>; generateReports?: ReturnType<typeof vi.fn> } = {}) {
      const requestId = 'persisted-e2e-request';
      const session = savedSession(requestId, false, 1);
      const api = {
        getModels: () => apiResult({ models: [model], providers: {} }),
        generateReports: options.generateReports ?? vi.fn(() => apiResult({ job_id: 'generation-job', poll_interval: 0.1 })),
        getHistory: vi.fn(() => apiResult(session)),
        updateHistory: vi.fn(),
      };
      const completed = { job_id: 'generation-job', job_type: 'inference', status: 'completed', poll_interval: 0.1, progress: 100, result: generationResult(requestId, session), error: null };
      await TestBed.configureTestingModule({
        imports: [InferencePage],
        providers: [
          provideRouter([]),
          { provide: InferenceApiService, useValue: api },
          { provide: JobsApiService, useValue: { get: vi.fn(), cancel: vi.fn() } },
          { provide: JobPollingService, useValue: { poll: options.poll ?? (() => of(completed)) } },
          { provide: GuidanceService, useValue: { requestTour: vi.fn(), shouldShow: () => false } },
        ],
      }).compileComponents();
      const appState = TestBed.inject(AppStateService);
      appState.updateInference((state) => ({
        ...state,
        selectedModelRef: modelRef,
        images: [new File(['one'], 'view-1.png', { type: 'image/png' }), new File(['two'], 'view-2.png', { type: 'image/png' }), new File(['three'], 'view-3.png', { type: 'image/png' })],
        ...options.state,
      }));
      const fixture = TestBed.createComponent(InferencePage);
      await fixture.whenStable();
      return { fixture, page: fixture.componentInstance, api, appState, requestId };
    }

    beforeEach(() => { vi.stubGlobal('URL', Object.assign(URL, urlStub)); });
    afterEach(() => { vi.unstubAllGlobals(); });

    it('resumes a job that finished while the page was closed and restores its saved report', async () => {
      const { page, api, appState, requestId } = await createPage({ state: { isGenerating: true, activeJobId: 'generation-job', runIndices: [0] } });

      expect(appState.inference().isGenerating).toBe(false);
      expect(appState.inference().activeJobId).toBeNull();
      expect(api.getHistory).toHaveBeenCalledWith(requestId);
      expect(page.persistedHistory()?.request_id).toBe(requestId);
      expect(page.draft('findings')).toBe('Clear lungs.');
      expect(page.currentImageUrl()).toBe('blob:preview');
    });

    it('keeps tracking a job that is still running when the page is recreated', async () => {
      const { page, appState } = await createPage({
        state: { isGenerating: true, activeJobId: 'generation-job' },
        poll: () => of({ job_id: 'generation-job', job_type: 'inference', status: 'running', poll_interval: 0.1, progress: 42, result: { lifecycle: { message: 'Generating report 1 of 3' } }, error: null }, ),
      });

      expect(appState.inference().isGenerating).toBe(true);
      expect(page.progress()).toBe(42);
      expect(page.progressMessage()).toBe('Generating report 1 of 3');
    });

    it('reloads the saved report of a completed run when the page is recreated', async () => {
      const { page, api, requestId } = await createPage({ state: { requestId: 'persisted-e2e-request', runIndices: [0] }, poll: () => NEVER });

      expect(api.getHistory).toHaveBeenCalledWith(requestId);
      expect(page.persistedHistory()?.request_id).toBe(requestId);
    });

    it('runs only the selected image by default and maps its report back to that image', async () => {
      const generateReports = vi.fn(() => apiResult({ job_id: 'generation-job', poll_interval: 0.1 }));
      const { fixture, page, appState } = await createPage({ generateReports, state: { currentIndex: 2 } });

      await page.generate();
      await fixture.whenStable();

      const [files, , , , batchMode] = generateReports.mock.calls[0] as unknown as [File[], string, string, string, string];
      expect(files.map((file) => file.name)).toEqual(['view-3.png']);
      expect(batchMode).toBe('study');
      expect(Object.keys(appState.inference().reports)).toEqual(['2']);
      expect(page.draft('findings')).toBe('Clear lungs.');
    });

    it('runs every uploaded image independently when requested', async () => {
      const generateReports = vi.fn(() => apiResult({ job_id: 'generation-job', poll_interval: 0.1 }));
      const { fixture, page, appState } = await createPage({ generateReports, state: { runScope: 'all' } });

      await page.generate();
      await fixture.whenStable();

      const [files, , , , batchMode] = generateReports.mock.calls[0] as unknown as [File[], string, string, string, string];
      expect(files.map((file) => file.name)).toEqual(['view-1.png', 'view-2.png', 'view-3.png']);
      expect(batchMode).toBe('per_image');
      expect(appState.inference().runIndices).toEqual([0, 1, 2]);
    });
  });
});
