import { CommonModule } from '@angular/common';
import { Component, DestroyRef, EventEmitter, Input, Output, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { CheckpointExchangeApiService } from '@xreport/shared/app/services/checkpoint-exchange-api.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import type { JobStatusResponse } from '@xreport/shared/app/types/jobs';
import { ModalFocusDirective } from '@xreport/shared/app/components/modal-focus.directive';

@Component({
  standalone: true,
  selector: 'app-checkpoint-export',
  imports: [CommonModule, FormsModule, ModalFocusDirective],
  template: `
    @if (checkpoint) {
      <div class="guidance-modal-backdrop" role="presentation">
        <section class="guidance-modal" role="dialog" aria-modal="true" aria-labelledby="checkpoint-export-title" appModalFocus (modalEscape)="close()">
          <header class="guidance-modal-header">
            <div>
              <h2 id="checkpoint-export-title">Export for XREPORT</h2>
              <p class="guidance-modal-subtitle">{{ checkpoint }}</p>
            </div>
          </header>
          <div class="guidance-modal-body">
            <p>Creates a portable archive with the weights, configuration and tokenizer. No dataset records or credentials are included.</p>
            <label class="guidance-checkbox">
              <input type="checkbox" [(ngModel)]="includeEncoder" [disabled]="busy()" />
              <span>Include the BEiT image encoder (about 350 MB) for machines without it installed</span>
            </label>
            @if (busy()) {
              <div role="progressbar" aria-label="Export progress" [attr.aria-valuenow]="progress()" aria-valuemin="0" aria-valuemax="100">Exporting — {{ progress() | number: '1.0-0' }}%</div>
            }
            @if (message(); as text) { <p [class.error-text]="failed()" role="status">{{ text }}</p> }
            @if (downloadUrl(); as url) { <p><a [href]="url" download>Download archive</a></p> }
          </div>
          <footer class="guidance-modal-footer">
            @if (busy()) {
              <button type="button" class="guidance-button" (click)="cancel()">Cancel export</button>
            } @else {
              <button type="button" class="guidance-button" (click)="close()">Close</button>
              <button type="button" class="guidance-button guidance-button-primary" (click)="start()">Export</button>
            }
          </footer>
        </section>
      </div>
    }
  `,
  styleUrl: '../../../../shared/src/app/styles/Guidance.css',
})
export class CheckpointExportComponent {
  private readonly api = inject(CheckpointExchangeApiService);
  private readonly jobs = inject(JobsApiService);
  private readonly polling = inject(JobPollingService);
  private readonly destroyRef = inject(DestroyRef);

  @Input() checkpoint: string | null = null;
  @Output() readonly closed = new EventEmitter<void>();

  includeEncoder = false;
  readonly busy = signal(false);
  readonly failed = signal(false);
  readonly message = signal<string | null>(null);
  readonly progress = signal(0);
  readonly downloadUrl = signal<string | null>(null);
  private jobId: string | null = null;

  async start(): Promise<void> {
    if (!this.checkpoint || this.busy()) return;
    this.busy.set(true);
    this.failed.set(false);
    this.message.set(null);
    this.downloadUrl.set(null);
    this.progress.set(0);
    const started = await this.api.startExport(this.checkpoint, this.includeEncoder);
    if (!started.result) {
      this.busy.set(false);
      this.failed.set(true);
      this.message.set(started.error ?? 'Export could not be started.');
      return;
    }
    this.jobId = started.result.job_id;
    this.polling.poll((id) => this.jobs.get(id), this.jobId, started.result.poll_interval ?? 1)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((status) => this.onStatus(status));
  }

  async cancel(): Promise<void> {
    if (this.jobId) await this.jobs.cancel(this.jobId);
  }

  close(): void {
    if (!this.busy()) this.closed.emit();
  }

  private onStatus(status: JobStatusResponse): void {
    this.progress.set(status.progress ?? 0);
    if (!['completed', 'failed', 'cancelled'].includes(status.status)) return;
    this.busy.set(false);
    this.jobId = null;
    if (status.status === 'cancelled') { this.message.set('Export cancelled.'); return; }
    if (status.status === 'failed') { this.failed.set(true); this.message.set(status.error ?? 'Export failed.'); return; }
    const exported = (status.result?.['export'] ?? {}) as { filename?: string };
    if (!exported.filename) { this.failed.set(true); this.message.set('Export finished without an archive.'); return; }
    this.message.set(`Archive ready: ${exported.filename}`);
    this.downloadUrl.set(this.api.exportUrl(exported.filename));
  }
}
