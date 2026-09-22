import { CommonModule } from '@angular/common';
import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { InferenceApiService } from '../services/inference-api.service';
import type {
  InferenceHistorySort,
  InferenceHistoryStatus,
  InferenceHistorySummary,
} from '../types/inferenceApi';

@Component({
  standalone: true,
  selector: 'app-reports-page',
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
    <main class="reports-page" aria-labelledby="reports-title">
      <header class="reports-header">
        <p class="reports-eyebrow">Durable inference history</p>
        <h1 id="reports-title">Reports</h1>
        <p>Review generated report sessions saved by XREPORT. Source radiographs are not retained here.</p>
      </header>

      <section class="reports-toolbar" aria-label="Report history filters">
        <label>
          <span>Model reference</span>
          <input [(ngModel)]="modelRef" name="model-ref" placeholder="Filter by model reference" (keyup.enter)="applyFilters()" />
        </label>
        <label>
          <span>Status</span>
          <select [(ngModel)]="status" name="status" (ngModelChange)="applyFilters()">
            <option value="">All statuses</option>
            @for (option of statusOptions; track option) { <option [value]="option">{{ pretty(option) }}</option> }
          </select>
        </label>
        <label>
          <span>Sort</span>
          <select [(ngModel)]="sort" name="sort" (ngModelChange)="applyFilters()">
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
        <button type="button" class="secondary-button" (click)="applyFilters()" [disabled]="loading()">Apply filters</button>
      </section>

      @if (loading()) {
        <section class="reports-state" role="status">Loading report history…</section>
      } @else if (error()) {
        <section class="reports-state reports-state-error" role="alert">
          <strong>Reports could not be loaded</strong>
          <span>{{ error() }}</span>
          <button type="button" class="secondary-button" (click)="load()">Try again</button>
        </section>
      } @else if (!items().length) {
        <section class="reports-state reports-empty" role="status">
          <strong>No saved reports yet</strong>
          <span>Successful inference runs will appear here after their generated reports are persisted.</span>
          <a class="primary-button" routerLink="/inference">Open Inference</a>
        </section>
      } @else {
        <section class="reports-results" aria-live="polite">
          <div class="reports-results-heading"><h2>Saved sessions</h2><span>{{ total() }} session{{ total() === 1 ? '' : 's' }}</span></div>
          <div class="report-list">
            @for (item of items(); track item.request_id) {
              <a class="report-card" [routerLink]="['/reports', item.request_id]">
                <div class="report-card-heading">
                  <div><strong>{{ modelLabel(item) }}</strong><span>{{ formatDate(item.date) }}</span></div>
                  <span class="history-status" [class]="'history-status history-status-' + item.status">{{ pretty(item.status) }}</span>
                </div>
                <div class="report-card-meta">
                  <span><b>Images</b>{{ item.image_names.join(', ') || 'No image names recorded' }}</span>
                  <span><b>Profile</b>{{ pretty(item.generation_profile) }}</span>
                  <span><b>Revision</b>{{ item.model_revision || 'Not reported' }}</span>
                  <span><b>Duration</b>{{ duration(item.execution_time_seconds) }}</span>
                </div>
                <div class="report-card-preview">
                  @if (item.reports[0]; as report) { <span>{{ report.preview || 'Report text is empty.' }}</span> }
                  @else { <span>This session has no persisted report rows.</span> }
                  @if (item.reports.some(isEdited)) { <em>Edited draft</em> }
                </div>
                <small class="report-request-id">Request {{ item.request_id }}</small>
              </a>
            }
          </div>
          <nav class="reports-pagination" aria-label="Report history pages">
            <button type="button" class="secondary-button" (click)="previousPage()" [disabled]="offset === 0 || loading()">Previous</button>
            <span>Showing {{ offset + 1 }}–{{ Math.min(offset + pageSize, total()) }} of {{ total() }}</span>
            <button type="button" class="secondary-button" (click)="nextPage()" [disabled]="offset + pageSize >= total() || loading()">Next</button>
          </nav>
        </section>
      }
    </main>
  `,
  styleUrl: '../styles/ReportsPage.css',
})
export class ReportsPage {
  private readonly api = inject(InferenceApiService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  private requestVersion = 0;

  readonly items = signal<InferenceHistorySummary[]>([]);
  readonly total = signal(0);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly pageSize = 10;
  readonly statusOptions: InferenceHistoryStatus[] = ['queued', 'running', 'succeeded', 'failed', 'cancelled'];
  readonly Math = Math;

  modelRef = '';
  status: InferenceHistoryStatus | '' = '';
  sort: InferenceHistorySort = 'newest';
  offset = 0;

  constructor() {
    this.route.queryParamMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      this.modelRef = params.get('model_ref') ?? '';
      const status = params.get('status') ?? '';
      this.status = this.statusOptions.includes(status as InferenceHistoryStatus)
        ? status as InferenceHistoryStatus
        : '';
      this.sort = params.get('sort') === 'oldest' ? 'oldest' : 'newest';
      const offset = Number(params.get('offset') ?? 0);
      this.offset = Number.isInteger(offset) && offset >= 0 ? offset : 0;
      void this.load();
    });
  }

  async load(): Promise<void> {
    const version = ++this.requestVersion;
    this.loading.set(true);
    this.error.set(null);
    const response = await this.api.listHistory({
      modelRef: this.modelRef.trim() || undefined,
      status: this.status || undefined,
      sort: this.sort,
      limit: this.pageSize,
      offset: this.offset,
    });
    if (version !== this.requestVersion) return;
    if (response.result) {
      this.items.set(response.result.items);
      this.total.set(response.result.total);
    } else {
      this.items.set([]);
      this.total.set(0);
      this.error.set(response.error ?? 'Unable to load report history.');
    }
    this.loading.set(false);
  }

  applyFilters(): void {
    this.offset = 0;
    void this.updateQuery();
  }

  previousPage(): void {
    if (this.offset === 0) return;
    this.offset = Math.max(0, this.offset - this.pageSize);
    void this.updateQuery();
  }

  nextPage(): void {
    if (this.offset + this.pageSize >= this.total()) return;
    this.offset += this.pageSize;
    void this.updateQuery();
  }

  async updateQuery(): Promise<void> {
    await this.router.navigate([], {
      relativeTo: this.route,
      queryParams: {
        model_ref: this.modelRef.trim() || null,
        status: this.status || null,
        sort: this.sort === 'newest' ? null : this.sort,
        offset: this.offset || null,
      },
      replaceUrl: true,
    });
  }

  modelLabel(item: InferenceHistorySummary): string {
    return item.model_ref.includes(':') ? item.model_ref.split(':').slice(1).join(':') : item.model_ref;
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

  isEdited(report: InferenceHistorySummary['reports'][number]): boolean {
    return report.edited;
  }
}
