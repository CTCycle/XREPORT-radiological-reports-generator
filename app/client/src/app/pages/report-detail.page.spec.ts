import { TestBed } from '@angular/core/testing';
import { ActivatedRoute, Router, convertToParamMap } from '@angular/router';
import { of } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { InferenceApiService } from '../services/inference-api.service';
import type { InferenceHistoryDetail } from '../types/inferenceApi';
import { ReportDetailPage } from './report-detail.page';

type HistoryUpdate = Parameters<InferenceApiService['updateHistory']>;
type UpdateResponse = Awaited<ReturnType<InferenceApiService['updateHistory']>>;
type DeleteHistory = Parameters<InferenceApiService['deleteHistory']>;
type DeleteResponse = Awaited<ReturnType<InferenceApiService['deleteHistory']>>;

const requestId = 'report-detail-test';

function savedSession(findings = 'Clear lungs.', edited = false): InferenceHistoryDetail {
  const reportText = `Findings\n${findings}`;
  return {
    request_id: requestId,
    provider: 'huggingface',
    model_ref: 'huggingface:e2e/report-detail',
    model_revision: 'test-revision',
    generation_profile: 'deterministic',
    generation_config: { provenance: 'test' },
    clinical_context: null,
    status: 'succeeded',
    execution_time_seconds: 0.1,
    date: '2026-10-23T10:00:00Z',
    reports: [{
      image_index: 0,
      input_image_name: 'study.png',
      generated_report: 'Findings\nClear lungs.',
      edited_report: edited ? reportText : null,
      effective_report: reportText,
      edited,
      edited_at: edited ? '2026-10-23T10:01:00Z' : null,
      sections: { findings },
    }],
    output_sections: ['findings'],
  };
}

function updateResult(result: InferenceHistoryDetail | null, error: string | null): UpdateResponse {
  return { result, error };
}

function deleteResult(success: boolean, message: string): DeleteResponse {
  return { result: { success, message }, error: null };
}

async function createPage(options: {
  updateHistory?: (...args: HistoryUpdate) => ReturnType<InferenceApiService['updateHistory']>;
  deleteHistory?: (...args: DeleteHistory) => ReturnType<InferenceApiService['deleteHistory']>;
} = {}) {
  const session = savedSession();
  const api = {
    getHistory: vi.fn(async () => ({ result: session, error: null })),
    updateHistory: options.updateHistory ?? vi.fn(async () => updateResult(savedSession('Reviewed report.', true), null)),
    deleteHistory: options.deleteHistory ?? vi.fn(async () => deleteResult(true, 'Deleted.')),
  };
  const router = { navigateByUrl: vi.fn(async () => true) };

  TestBed.configureTestingModule({
    providers: [
      { provide: ActivatedRoute, useValue: { paramMap: of(convertToParamMap({ requestId })) } },
      { provide: Router, useValue: router },
      { provide: InferenceApiService, useValue: api },
    ],
  });

  const page = TestBed.runInInjectionContext(() => new ReportDetailPage());
  await Promise.resolve();
  await Promise.resolve();
  return { page, api, router };
}

afterEach(() => {
  vi.restoreAllMocks();
  TestBed.resetTestingModule();
});

describe('ReportDetailPage unsaved edits', () => {
  it('asks before leaving a dirty report and preserves the exact draft when the user stays', async () => {
    const { page } = await createPage();
    page.updateDraft(0, 'findings', 'Corrected wording with punctuation.');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);

    expect(page.confirmDiscardUnsavedChanges()).toBe(false);
    expect(confirm).toHaveBeenCalledOnce();
    expect(page.drafts()[0]?.findings).toBe('Corrected wording with punctuation.');
    expect(page.dirty()).toBe(true);
  });

  it('does not prompt when clean and protects a dirty draft from browser unload', async () => {
    const { page } = await createPage();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const cleanEvent = new Event('beforeunload', { cancelable: true }) as BeforeUnloadEvent;

    expect(page.confirmDiscardUnsavedChanges()).toBe(true);
    page.protectUnsavedDraft(cleanEvent);
    expect(cleanEvent.defaultPrevented).toBe(false);
    expect(confirm).not.toHaveBeenCalled();

    page.updateDraft(0, 'findings', 'Unsaved finding.');
    const dirtyEvent = new Event('beforeunload', { cancelable: true }) as BeforeUnloadEvent;
    page.protectUnsavedDraft(dirtyEvent);
    expect(dirtyEvent.defaultPrevented).toBe(true);
  });

  it('blocks navigation during save, then allows exit after a successful save without confirmation', async () => {
    let resolveSave!: (value: UpdateResponse) => void;
    const updateHistory = vi.fn(() => new Promise<UpdateResponse>((resolve) => { resolveSave = resolve; }));
    const { page } = await createPage({ updateHistory });
    page.updateDraft(0, 'findings', 'Reviewed report.');
    const confirm = vi.spyOn(window, 'confirm');

    const save = page.save();
    expect(page.saving()).toBe(true);
    expect(page.confirmDiscardUnsavedChanges()).toBe(false);
    expect(page.errorMessage()).toBe('Wait for the report operation to finish before leaving this page.');
    expect(confirm).not.toHaveBeenCalled();

    resolveSave(updateResult(savedSession('Reviewed report.', true), null));
    await save;

    expect(page.saving()).toBe(false);
    expect(page.dirty()).toBe(false);
    expect(page.confirmDiscardUnsavedChanges()).toBe(true);
    expect(confirm).not.toHaveBeenCalled();
  });

  it.each([
    ['an error response', async () => updateResult(null, 'The server rejected the edit.')],
    ['a rejected request', async () => { throw new Error('Network unavailable.'); }],
  ])('retains the draft and unlocks retry after %s', async (_failureKind, failUpdate) => {
    const updateHistory = vi.fn(failUpdate);
    const { page } = await createPage({ updateHistory });
    page.updateDraft(0, 'findings', 'Keep this exact draft.');

    await page.save();

    expect(page.saving()).toBe(false);
    expect(page.dirty()).toBe(true);
    expect(page.drafts()[0]?.findings).toBe('Keep this exact draft.');
    expect(page.errorMessage()).toBeTruthy();
  });

  it('bypasses the discard prompt only after session deletion succeeds', async () => {
    const { page, router } = await createPage();
    page.updateDraft(0, 'findings', 'Draft before deleting the session.');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);

    await page.deleteSession();

    expect(router.navigateByUrl).toHaveBeenCalledWith('/reports');
    expect(confirm).toHaveBeenCalledOnce();
    expect(page.confirmDiscardUnsavedChanges()).toBe(true);
    expect(confirm).toHaveBeenCalledOnce();
  });

  it('retains edits and keeps the discard guard active when deletion fails', async () => {
    const deleteHistory = vi.fn(async () => deleteResult(false, 'Deletion failed.'));
    const { page, router } = await createPage({ deleteHistory });
    page.updateDraft(0, 'findings', 'Keep this draft after failed deletion.');
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(true).mockReturnValue(false);

    await page.deleteSession();

    expect(page.deleting()).toBe(false);
    expect(page.dirty()).toBe(true);
    expect(page.drafts()[0]?.findings).toBe('Keep this draft after failed deletion.');
    expect(page.errorMessage()).toBe('Deletion failed.');
    expect(router.navigateByUrl).not.toHaveBeenCalled();
    expect(page.confirmDiscardUnsavedChanges()).toBe(false);
    expect(confirm).toHaveBeenCalledTimes(2);
  });
});
