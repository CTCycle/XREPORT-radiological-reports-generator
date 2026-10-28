import { CommonModule } from '@angular/common';
import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideArrowLeft, lucideLoaderCircle, lucidePlay, lucideSettings } from '@ng-icons/lucide';
import { ValidationApiService } from '../services/validation-api.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import { TokenizerSelectorComponent } from '../components/tokenizer-selector.component';
import { HelpPopoverComponent } from '@xreport/shared/app/components/help-popover.component';
import { ValidationStatisticsComponent } from '../components/validation-statistics.component';

@Component({
  standalone: true,
  selector: 'app-dataset-validation-page',
  imports: [CommonModule, ReactiveFormsModule, NgIcon, HelpPopoverComponent, ValidationStatisticsComponent, TokenizerSelectorComponent],
  providers: [provideIcons({ lucideArrowLeft, lucideLoaderCircle, lucidePlay, lucideSettings })],
  template: `
    <main class="dataset-container page-container">
      <div class="header">
        <button type="button" class="btn-icon-text validation-back-button" (click)="back()"><ng-icon name="lucideArrowLeft"/><span>Back to Datasets</span></button>
        <h1>Validation Wizard</h1>
        <p>Validate and analyze: <strong>{{ datasetName }}</strong></p>
      </div>
      <section class="section">
        <div class="section-title"><ng-icon name="lucideSettings"/><span>Configuration</span><app-help-popover label="Explain validation configuration" title="Validation configuration" body="Use a fraction while exploring, then use 1 for final checks. Select at least one metric; each metric adds a different view of dataset quality." /></div>
        <form [formGroup]="validationForm" class="validation-config-panel validation-config-layout">
          <div class="validation-config-row">
            <div class="form-group validation-config-min-width"><label for="dataset-validation-sample-size" class="form-label">Evaluation Sample Size (0-1)</label><input id="dataset-validation-sample-size" type="number" step="0.05" min="0.01" max="1" class="form-input" formControlName="evalSampleSize"/></div>
            <div class="eval-checkboxes validation-checkboxes-row">
              <label class="checkbox-label"><input type="checkbox" formControlName="imgStats"/> Image statistics</label>
              <label class="checkbox-label"><input type="checkbox" formControlName="textStats"/> Text statistics</label>
              <label class="checkbox-label"><input type="checkbox" formControlName="pixDist"/> Pixel intensity dist.</label>
              <label class="checkbox-label"><input type="checkbox" formControlName="countTokens"/> Count tokens (needs text statistics)</label>
            </div>
          </div>
          @if (validationForm.controls.countTokens.value) { <app-tokenizer-selector idPrefix="validation-tokenizer" [value]="tokenizer()" (valueChange)="tokenizer.set($event)" (compatibleChange)="tokenizerCompatible.set($event)"/> }
          <div class="form-actions"><button type="button" class="btn btn-primary validation-run-button" (click)="runValidation()" [disabled]="isValidating()"><ng-icon [name]="isValidating() ? 'lucideLoaderCircle' : 'lucidePlay'" [class.spin]="isValidating()"/>{{ isValidating() ? 'Running Validation...' : 'Run Validation' }}</button></div>
        </form>
      </section>
      <section class="validation-results-spacing">
        <div class="validation-dashboard">
          <div class="dashboard-status-header"><h2>Dataset Validation Results</h2>@if (isValidating()) { <span class="status-badge" role="status">Running</span> } @else if (validationError()) { <span class="status-badge" role="status">Error</span> }</div>
          @if (isValidating()) {
            <div class="loading-container"><ng-icon name="lucideLoaderCircle" class="spin"/><span class="loading-text">Running validation analytics... {{ progress() | number:'1.0-0' }}%</span><div class="progress-track"><span [style.width.%]="progress()"></span></div></div>
          } @else if (validationError()) {
            <div class="idle-message error" role="alert">{{ validationError() }}</div>
          }
          <app-validation-statistics [datasetName]="datasetName" [refreshToken]="statisticsRefresh()"/>
        </div>
      </section>
    </main>
  `,
  styleUrl: '../styles/DatasetPage.css',
})
export class DatasetValidationPage {
  private readonly api = inject(ValidationApiService);
  private readonly polling = inject(JobPollingService);
  private readonly jobsApi = inject(JobsApiService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  readonly datasetName = this.route.snapshot.paramMap.get('datasetName') ?? '';
  readonly validationForm = inject(NonNullableFormBuilder).group({
    evalSampleSize: [0.2, [Validators.required, Validators.min(0.01), Validators.max(1)]],
    imgStats: true,
    textStats: true,
    pixDist: true,
    countTokens: false,
  });
  readonly tokenizer = signal('distilbert-base-uncased');
  readonly tokenizerCompatible = signal<boolean | null>(null);
  readonly isValidating = signal(false);
  readonly progress = signal(0);
  readonly validationError = signal<string | null>(null);
  readonly statisticsRefresh = signal(0);

  constructor() { if (!this.datasetName) void this.router.navigate(['/dataset']); }

  back() { void this.router.navigate(['/dataset']); }

  async runValidation() {
    const config = this.validationForm.getRawValue();
    if (this.validationForm.invalid) { this.validationForm.markAllAsTouched(); this.validationError.set('Enter a sample size between 0.01 and 1.'); return; }
    const metrics = [config.imgStats ? 'image_statistics' : '', config.textStats ? 'text_statistics' : '', config.pixDist ? 'pixels_distribution' : ''].filter(Boolean);
    if (!metrics.length) { this.validationError.set('Please select at least one validation metric'); return; }
    if (config.countTokens && !config.textStats) { this.validationError.set('Token counts need the text statistics metric.'); return; }
    if (config.countTokens && this.tokenizerCompatible() !== true) { this.validationError.set('Choose a compatible tokenizer to count tokens.'); return; }
    this.isValidating.set(true);
    this.validationError.set(null);
    const started = await this.api.run({ dataset_name: this.datasetName, metrics, sample_size: config.evalSampleSize, ...(config.countTokens ? { tokenizer: this.tokenizer() } : {}) });
    if (!started.result) { this.isValidating.set(false); this.validationError.set(started.error ?? 'Failed to start validation job'); return; }
    this.polling.poll((jobId) => this.jobsApi.get(jobId), started.result.job_id, started.result.poll_interval ?? 1).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => {
      this.progress.set(status.progress ?? 0);
      if (!['completed', 'failed', 'cancelled'].includes(status.status)) return;
      this.isValidating.set(false);
      if (status.status === 'completed') this.statisticsRefresh.update((value) => value + 1);
      else this.validationError.set(status.error ?? `Validation ${status.status}`);
    });
  }
}
