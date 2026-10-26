import { CommonModule } from '@angular/common';
import { Component, DestroyRef, HostListener, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { GenerationMetadataComponent } from '../components/generation-metadata.component';
import { ReportDraftEditorComponent } from '../components/report-draft-editor.component';
import { SECTION_LABELS, formatReportDraft, isOutputSection, parseReportDraft } from '../common/report-sections';
import type { DraftSections } from '../common/report-sections';
import { diffOriginal } from '../common/text-diff';
import type { OriginalSegment } from '../common/text-diff';
import { InferenceApiService } from '../services/inference-api.service';
import type {
  InferenceHistoryDetail,
  InferenceHistoryReport,
  OutputSection,
} from '../types/inferenceApi';

type DraftMap = Record<number, DraftSections>;
interface OriginalPart { label: string; segments: OriginalSegment[]; }

@Component({
  standalone: true,
  selector: 'app-report-detail-page',
  imports: [CommonModule, RouterLink, GenerationMetadataComponent, ReportDraftEditorComponent],
  template: `
    <main class="report-detail-page page-container" aria-labelledby="report-detail-title">
      <a class="back-link" routerLink="/reports">← Back to Reports</a>

      @if (loading()) {
        <section class="detail-state" role="status">Loading saved report…</section>
      } @else if (error()) {
        <section class="detail-state detail-state-error" role="alert">
          <strong>Saved report could not be loaded</strong>
          <span>{{ error() }}</span>
          <a class="secondary-button" routerLink="/reports">Return to Reports</a>
        </section>
      } @else if (detail(); as session) {
        <header class="detail-header">
          <div>
            <h1 id="report-detail-title">{{ modelLabel(session.model_ref) }}</h1>
            <p>{{ formatDate(session.date) }} · {{ session.reports.length }} report{{ session.reports.length === 1 ? '' : 's' }}</p>
          </div>
          <span class="history-status" [class]="'history-status history-status-' + session.status">{{ pretty(session.status) }}</span>
        </header>

        <section class="detail-metadata" aria-label="Inference session metadata">
          <div><b>Model reference</b><span>{{ session.model_ref }}</span></div>
          <div><b>Revision</b><span>{{ session.model_revision || 'Not reported' }}</span></div>
          <div><b>Profile</b><span>{{ pretty(session.generation_profile) }}</span></div>
          <div><b>Duration</b><span>{{ duration(session.execution_time_seconds) }}</span></div>
          <div><b>Provider</b><span>{{ session.provider }}</span></div>
          <div><b>Clinical context</b><span>{{ session.clinical_context || 'None recorded' }}</span></div>
        </section>

        @if (session.status !== 'succeeded') {
          <div class="detail-notice" role="status">This session was not marked successful, so its saved text is read-only.</div>
        }

        @if (!session.reports.length) {
          <section class="detail-state" role="status">This session has no persisted report rows.</section>
        } @else {
          <section class="detail-reports" aria-label="Saved report drafts">
            @for (report of session.reports; track report.image_index) {
              <article class="detail-report">
                <header class="detail-report-header">
                  <div><span class="detail-report-index">Image {{ report.image_index + 1 }}</span><h2>{{ report.input_image_name }}</h2></div>
                  @if (report.edited) { <span class="edited-indicator">Edited draft</span> }
                </header>
                <app-report-draft-editor
                  [sections]="sectionsFor(report)"
                  [drafts]="drafts()[report.image_index] ?? {}"
                  [readOnly]="session.status !== 'succeeded'"
                  (draftChange)="updateDraft(report.image_index, $event.section, $event.value)"
                />
                <details class="original-output">
                  <summary>Show original model output</summary>
                  @if (originalParts(report); as parts) {
                    <div class="original-text">
                      @for (part of parts; track part.label) {
                        <p>
                          @if (part.label) { <strong class="original-label">{{ part.label }}</strong> }
                          <span>@for (segment of part.segments; track $index) {<span [class.diff-changed]="segment.changed">{{ segment.text }}</span>}</span>
                        </p>
                      }
                    </div>
                    @if (hasOriginalChanges(parts)) { <p class="original-legend">Highlighted text was changed or removed in the current report.</p> }
                  }
                </details>
              </article>
            }
          </section>
        }

        <details class="generation-metadata">
          <summary>Generation metadata and provenance</summary>
          <app-generation-metadata [config]="session.generation_config" />
        </details>

        <div class="detail-feedback" aria-live="polite">
          @if (saveMessage()) { <span class="detail-success">{{ saveMessage() }}</span> }
          @if (errorMessage()) { <span class="detail-error" role="alert">{{ errorMessage() }}</span> }
        </div>
        <footer class="detail-actions">
          <button type="button" class="danger-button" (click)="deleteSession()" [disabled]="deleting() || saving()">{{ deleting() ? 'Deleting…' : 'Delete session' }}</button>
          <button type="button" class="primary-button" (click)="save()" [disabled]="!dirty() || saving() || deleting() || session.status !== 'succeeded'">{{ saving() ? 'Saving…' : 'Save edits' }}</button>
        </footer>
      }
    </main>
  `,
  styleUrl: '../styles/ReportDetailPage.css',
})
export class ReportDetailPage {
  private readonly api = inject(InferenceApiService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly errorMessage = signal<string | null>(null);
  readonly saveMessage = signal<string | null>(null);
  readonly saving = signal(false);
  readonly deleting = signal(false);
  readonly detail = signal<InferenceHistoryDetail | null>(null);
  readonly drafts = signal<DraftMap>({});
  private readonly baselineDrafts = signal<DraftMap>({});
  readonly dirty = computed(() => JSON.stringify(this.drafts()) !== JSON.stringify(this.baselineDrafts()));
  private requestId: string | null = null;
  private allowNavigationAfterDeletion = false;

  constructor() {
    this.route.paramMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      this.requestId = params.get('requestId');
      void this.load(this.requestId);
    });
  }

  async load(requestId: string | null): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    this.errorMessage.set(null);
    this.saveMessage.set(null);
    if (!requestId) {
      this.error.set('The saved report request ID is missing.');
      this.loading.set(false);
      return;
    }
    const response = await this.api.getHistory(requestId);
    if (response.result) {
      this.applyDetail(response.result);
    } else {
      this.error.set(response.error ?? 'Unable to load saved report.');
    }
    this.loading.set(false);
  }

  private applyDetail(value: InferenceHistoryDetail): void {
    const next: DraftMap = {};
    for (const report of value.reports) {
      next[report.image_index] = parseReportDraft(report.effective_report, this.sectionsFor(report, value));
    }
    this.detail.set(value);
    this.drafts.set(next);
    this.baselineDrafts.set(structuredClone(next));
  }

  sectionsFor(report: InferenceHistoryReport, session = this.detail()): OutputSection[] {
    const declared = Object.keys(report.sections ?? {}).filter(isOutputSection);
    if (declared.length) return declared;
    const sessionSections = session?.output_sections.filter(isOutputSection) ?? [];
    return sessionSections.length ? sessionSections : ['raw_report'];
  }

  updateDraft(imageIndex: number, section: OutputSection, value: string): void {
    this.drafts.update((drafts) => ({
      ...drafts,
      [imageIndex]: { ...(drafts[imageIndex] ?? {}), [section]: value },
    }));
    this.errorMessage.set(null);
    this.saveMessage.set(null);
  }

  originalParts(report: InferenceHistoryReport): OriginalPart[] {
    const sections = this.sectionsFor(report);
    const draft = this.drafts()[report.image_index] ?? {};
    if (sections.includes('raw_report')) {
      return [{ label: '', segments: diffOriginal(report.generated_report, draft.raw_report ?? '') }];
    }
    const original = parseReportDraft(report.generated_report, sections);
    return sections.map((section) => ({
      label: SECTION_LABELS[section],
      segments: diffOriginal(original[section] ?? '', draft[section] ?? ''),
    }));
  }

  hasOriginalChanges(parts: OriginalPart[]): boolean {
    return parts.some((part) => part.segments.some((segment) => segment.changed));
  }

  private reportText(report: InferenceHistoryReport): string {
    return formatReportDraft(
      this.sectionsFor(report),
      this.drafts()[report.image_index] ?? {},
      report.effective_report,
    );
  }

  async save(): Promise<void> {
    const session = this.detail();
    if (!session || !this.requestId || session.status !== 'succeeded' || !this.dirty() || this.saving() || this.deleting()) return;
    const updates = session.reports
      .filter((report) => JSON.stringify(this.drafts()[report.image_index] ?? {}) !== JSON.stringify(this.baselineDrafts()[report.image_index] ?? {}))
      .map((report) => ({ image_index: report.image_index, edited_report: this.reportText(report) }));
    if (!updates.length) return;
    this.saving.set(true);
    this.errorMessage.set(null);
    this.saveMessage.set(null);
    try {
      const response = await this.api.updateHistory(this.requestId, { reports: updates });
      if (response.result) {
        this.applyDetail(response.result);
        this.errorMessage.set(null);
        this.saveMessage.set('Edits saved. The original model output remains available below.');
      } else {
        this.errorMessage.set(response.error ?? 'Unable to save report edits. Your changes are still available; try again.');
      }
    } catch (error) {
      this.errorMessage.set(error instanceof Error ? error.message : 'Unable to save report edits. Your changes are still available; try again.');
    } finally {
      this.saving.set(false);
    }
  }

  async deleteSession(): Promise<void> {
    const session = this.detail();
    if (!session || !this.requestId || this.deleting() || this.saving()) return;
    const name = session.reports[0]?.input_image_name ?? session.request_id;
    if (!confirm(`Delete the saved report session for ${name}? This removes the history entry but not model or dataset files.`)) return;
    this.deleting.set(true);
    this.errorMessage.set(null);
    try {
      const response = await this.api.deleteHistory(this.requestId);
      if (!response.result?.success) {
        this.errorMessage.set(response.error ?? response.result?.message ?? 'Unable to delete saved report.');
        this.deleting.set(false);
        return;
      }
    } catch (error) {
      this.errorMessage.set(error instanceof Error ? error.message : 'Unable to delete saved report.');
      this.deleting.set(false);
      return;
    }

    this.allowNavigationAfterDeletion = true;
    try {
      const navigated = await this.router.navigateByUrl('/reports');
      if (!navigated) {
        this.allowNavigationAfterDeletion = false;
        this.deleting.set(false);
        this.errorMessage.set('The session was deleted, but Reports could not be opened.');
      }
    } catch (error) {
      this.allowNavigationAfterDeletion = false;
      this.deleting.set(false);
      this.errorMessage.set(error instanceof Error ? error.message : 'The session was deleted, but Reports could not be opened.');
    }
  }

  confirmDiscardUnsavedChanges(): boolean {
    if (this.allowNavigationAfterDeletion) {
      this.allowNavigationAfterDeletion = false;
      return true;
    }
    if (this.saving() || this.deleting()) {
      this.errorMessage.set('Wait for the report operation to finish before leaving this page.');
      return false;
    }
    return !this.dirty() || window.confirm('You have unsaved report edits. Leave and discard them? The last saved report remains unchanged.');
  }

  @HostListener('window:beforeunload', ['$event'])
  protectUnsavedDraft(event: BeforeUnloadEvent): void {
    if (!this.dirty() && !this.saving() && !this.deleting()) return;
    event.preventDefault();
    event.returnValue = '';
  }

  modelLabel(modelRef: string): string {
    return modelRef.includes(':') ? modelRef.split(':').slice(1).join(':') : modelRef;
  }

  formatDate(value: string | null | undefined): string {
    if (!value) return 'Date not recorded';
    const parsed = new Date(value.includes('T') ? value : `${value.replace(' ', 'T')}Z`);
    return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
  }

  duration(value: number | null | undefined): string {
    return value === null || value === undefined ? 'Not recorded' : `${value.toFixed(1)} s`;
  }

  pretty(value: string): string {
    return value.replaceAll('_', ' ');
  }
}
