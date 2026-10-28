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
  selector: 'app-checkpoint-import',
  imports: [CommonModule, FormsModule, ModalFocusDirective],
  template: `
    @if (open) {
      <div class="guidance-modal-backdrop" role="presentation">
        <section class="guidance-modal" role="dialog" aria-modal="true" aria-labelledby="checkpoint-import-title" appModalFocus (modalEscape)="close()">
          <header class="guidance-modal-header">
            <div>
              <h2 id="checkpoint-import-title">Import checkpoint</h2>
              <p class="guidance-modal-subtitle">Install a model exported from XREPORT Studio (.xreportckpt).</p>
            </div>
          </header>
          <div class="guidance-modal-body">
            <p>
              Select the <code>.xreportckpt</code> archive exported from XREPORT Studio. It bundles the model,
              configuration and tokenizer with a manifest that is verified before anything is installed.
              A lone <code>.keras</code> file or a hand-made zip cannot be imported.
            </p>
            <label class="guidance-field">
              <span>Checkpoint archive</span>
              <input type="file" accept=".xreportckpt" [disabled]="busy()" (change)="onFile($event)" />
            </label>
            @if (conflict()) {
              <label class="guidance-field">
                <span>Name for the imported checkpoint</span>
                <input type="text" [(ngModel)]="renameTo" [disabled]="busy()" maxlength="128" />
              </label>
            }
            @if (busy()) {
              <div role="progressbar" aria-label="Import progress" [attr.aria-valuenow]="progress()" aria-valuemin="0" aria-valuemax="100">
                {{ phase() }} — {{ progress() | number: '1.0-0' }}%
              </div>
            }
            @if (message(); as text) { <p [class.error-text]="failed()" role="status">{{ text }}</p> }
          </div>
          <footer class="guidance-modal-footer">
            @if (busy()) {
              <button type="button" class="guidance-button" (click)="cancel()">Cancel import</button>
            } @else {
              <button type="button" class="guidance-button" (click)="close()">Close</button>
              <button type="button" class="guidance-button guidance-button-primary" [disabled]="!file" (click)="start()">Import</button>
            }
          </footer>
        </section>
      </div>
    }
  `,
  styleUrl: '../../../../shared/src/app/styles/Guidance.css',
})
export class CheckpointImportComponent {
  private readonly api = inject(CheckpointExchangeApiService);
  private readonly jobs = inject(JobsApiService);
  private readonly polling = inject(JobPollingService);
  private readonly destroyRef = inject(DestroyRef);

  @Input() open = false;
  @Output() readonly closed = new EventEmitter<void>();
  @Output() readonly imported = new EventEmitter<string>();

  file: File | null = null;
  renameTo = '';
  readonly busy = signal(false);
  readonly failed = signal(false);
  readonly conflict = signal(false);
  readonly message = signal<string | null>(null);
  readonly progress = signal(0);
  readonly phase = signal('Uploading');
  private jobId: string | null = null;

  onFile(event: Event): void {
    this.file = (event.target as HTMLInputElement).files?.[0] ?? null;
    this.message.set(null);
    this.conflict.set(false);
    this.renameTo = '';
  }

  async start(): Promise<void> {
    if (!this.file || this.busy()) return;
    this.busy.set(true);
    this.failed.set(false);
    this.message.set(null);
    this.progress.set(0);
    this.phase.set('Uploading');
    const started = await this.api.startImport(this.file, this.renameTo.trim() || undefined);
    if (!started.result) {
      this.busy.set(false);
      this.failed.set(true);
      this.conflict.set(started.error?.startsWith('409 ') ?? false);
      this.message.set(started.error ?? 'Import could not be started.');
      return;
    }
    this.jobId = started.result.job_id;
    this.phase.set('Verifying and installing');
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
    if (status.status === 'cancelled') {
      this.message.set('Import cancelled. Nothing was installed.');
      return;
    }
    if (status.status === 'failed') {
      this.failed.set(true);
      this.message.set(status.error ?? 'Import failed. Nothing was installed.');
      return;
    }
    const outcome = (status.result?.['import'] ?? {}) as { checkpoint?: string; status?: string };
    this.message.set(outcome.status === 'already_imported'
      ? `${outcome.checkpoint} is already installed.`
      : `Imported ${outcome.checkpoint}. It is now available in the model catalogue.`);
    this.file = null;
    this.imported.emit(outcome.checkpoint ?? '');
  }
}
