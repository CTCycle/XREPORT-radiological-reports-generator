import { CommonModule } from '@angular/common';
import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideAlertCircle, lucideBarChart2, lucideCheckCircle, lucideDatabase, lucideEye, lucideFileSpreadsheet, lucideFolderUp, lucideLoaderCircle, lucideRefreshCw, lucideSliders, lucideTrash2, lucideX } from '@ng-icons/lucide';
import { ApiService } from '../services/api.service';
import { AppStateService } from '../services/app-state.service';
import { JobPollingService } from '../services/job-polling.service';
import { StorageService } from '../services/storage.service';
import type { DatasetInfo, DirectoryItem } from '../types/trainingApi';
import type { DatasetProcessingConfig } from '../types';
import type { ValidationMetric, ValidationWizardConfirmPayload } from '../types/validationWizard';
import type { ValidationResponse } from '../types/validationApi';
import { ImageViewerComponent } from '../components/image-viewer.component';
import { ModalFocusDirective } from '../components/modal-focus.directive';
import { ValidationReportModalComponent } from '../components/validation-report-modal.component';
import { ValidationWizardComponent } from '../components/validation-wizard.component';

interface StoredValidationJob { jobId: string; status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'; progress?: number; metrics: string[]; sampleSize: number; }
const VALIDATION_STORAGE_KEY = 'xreport.validation.jobs';

@Component({
  standalone: true,
  selector: 'app-dataset-page',
  imports: [CommonModule, FormsModule, NgIcon, ImageViewerComponent, ModalFocusDirective, ValidationReportModalComponent, ValidationWizardComponent],
  providers: [provideIcons({ lucideAlertCircle, lucideBarChart2, lucideCheckCircle, lucideDatabase, lucideEye, lucideFileSpreadsheet, lucideFolderUp, lucideLoaderCircle, lucideRefreshCw, lucideSliders, lucideTrash2, lucideX })],
  template: `
    <main class="dataset-container"><div class="layout-rows">
      <div class="layout-row row-datasource"><section class="section"><div class="section-title"><ng-icon name="lucideDatabase"/><span>Data Source</span></div><div class="upload-row-content"><div class="upload-grid"><button type="button" class="upload-card" (click)="folderBrowserOpen.set(true)" [disabled]="!canBrowse()"><ng-icon name="lucideFolderUp" class="upload-icon"/><div class="upload-text">Upload Image Folder</div><div class="upload-hint">{{ canBrowse() ? 'DICOM, PNG, JPG' : 'Disabled by server configuration' }}</div><div class="upload-subtext">{{ state().imageFolderName || (canBrowse() ? 'Select directory' : 'Unavailable') }}</div>@if (state().imageValidation) { <div class="upload-status" [class.success]="state().imageValidation?.valid" [class.error]="!state().imageValidation?.valid"><ng-icon [name]="state().imageValidation?.valid ? 'lucideCheckCircle' : 'lucideAlertCircle'"/>{{ state().imageValidation?.valid ? state().imageValidation?.image_count + ' images' : state().imageValidation?.message }}</div> }</button><button type="button" class="upload-card" (click)="datasetInput.click()"><ng-icon name="lucideFileSpreadsheet" class="upload-icon"/><div class="upload-text">Upload Data File</div><div class="upload-hint">Reports &amp; Metadata</div><div class="upload-subtext">{{ state().datasetFile?.name || 'Select .csv or .xlsx' }}</div>@if (state().datasetUpload?.success) { <div class="upload-status success"><ng-icon name="lucideCheckCircle"/>{{ state().datasetUpload?.row_count }} rows, {{ state().datasetUpload?.column_count }} cols</div> }</button><input #datasetInput class="sr-only" type="file" accept=".csv,.xlsx" (change)="uploadFile(datasetInput.files)"/></div><div class="load-dataset-section"><p class="load-dataset-description">Load a dataset from the selected source to make it available for processing.</p><div class="load-dataset-actions"><button type="button" class="btn btn-secondary btn-sm" (click)="loadDataset()" [disabled]="state().isLoading"><ng-icon *ngIf="state().isLoading" name="lucideLoaderCircle" class="spin"/>{{ state().isLoading ? 'Loading...' : 'Load Dataset' }}</button>@if (state().uploadError) { <div class="upload-status error"><ng-icon name="lucideAlertCircle"/>{{ state().uploadError }}</div> }@if (state().loadResult?.success) { <div class="upload-status success"><ng-icon name="lucideCheckCircle"/>Loaded {{ state().loadResult?.matched_records }} records ({{ state().loadResult?.total_images }} images)</div> }</div></div></div></section></div>
      <div class="layout-row"><section class="section"><div class="section-title processing-section-title"><div class="section-title-label"><ng-icon name="lucideSliders"/><span>Dataset Processing</span></div>@if (state().processingResult?.success) { <div class="upload-status success processing-header-status"><ng-icon name="lucideCheckCircle"/>Processed: {{ state().processingResult?.train_samples }} train, {{ state().processingResult?.validation_samples }} val</div> }</div><div class="dataset-processing-split"><div class="dataset-grid-section"><div class="dataset-table-container"><div class="dataset-table-header-row"><span class="dataset-table-title">Available Datasets</span><button type="button" class="btn-icon-small dataset-refresh-button" title="Refresh datasets" (click)="refreshNames()"><ng-icon name="lucideRefreshCw"/></button></div><p class="dataset-table-hint">Lists all datasets currently stored in the database. Select one to configure and build.</p><div class="dataset-table"><div class="dataset-table-header"><span class="dataset-table-col-actions">Actions</span><span>Name</span><span>Source</span><span class="dataset-table-col-rows">Rows</span></div><div class="dataset-table-body">@if (!datasets().length) { <div class="dataset-table-empty">Please upload at least one dataset</div> }@for (dataset of datasets(); track dataset.name) { <div class="dataset-table-row" [class.selected]="state().selectedDatasets.includes(dataset.name)" role="button" tabindex="0" [attr.aria-pressed]="state().selectedDatasets.includes(dataset.name)" (click)="toggleDataset(dataset.name)" (keydown.enter)="toggleDataset(dataset.name)"><div class="dataset-row-actions"><button type="button" class="btn-icon-small" title="Run validation" (click)="$event.stopPropagation(); openValidationWizard(dataset)"><ng-icon name="lucideCheckCircle"/></button><button type="button" class="btn-icon-small" [class.active]="dataset.has_validation_report || !!validationJobs()[dataset.name]" [attr.aria-label]="dataset.has_validation_report ? 'View validation report' : 'Open validation route'" title="View validation report" (click)="$event.stopPropagation(); openReport(dataset)"><ng-icon name="lucideBarChart2"/></button><button type="button" class="btn-icon-small" title="View images" (click)="$event.stopPropagation(); openViewer(dataset.name)"><ng-icon name="lucideEye"/></button><button type="button" class="btn-icon-small danger" title="Delete dataset" (click)="$event.stopPropagation(); deleteDataset(dataset.name)"><ng-icon name="lucideTrash2"/></button></div><span>{{ dataset.name }}</span><span>{{ dataset.folder_path || 'Uploaded source' }}</span><span class="dataset-table-col-rows">{{ dataset.row_count }}</span></div> }</div></div></div><div class="processing-config-panel"><h3>Processing Configuration</h3><p class="processing-description">Configure how reports are cleaned, tokenized, split, and stored for training.</p><div class="form-grid"><div class="form-group"><label class="form-label" for="dataset-sample-size">Sample Size (0-1)</label><input id="dataset-sample-size" class="form-input" type="number" min="0.01" max="1" step="0.05" [ngModel]="config().sampleSize" (ngModelChange)="setConfig('sampleSize', +$event)"/></div><div class="form-group"><label class="form-label" for="dataset-validation-size">Val Split (0-1)</label><input id="dataset-validation-size" class="form-input" type="number" min="0" max="1" step="0.05" [ngModel]="config().validationSize" (ngModelChange)="setConfig('validationSize', +$event)"/></div><div class="form-group"><label class="form-label" for="dataset-max-report-size">Max Report Size</label><input id="dataset-max-report-size" class="form-input" type="number" [ngModel]="config().maxReportSize" (ngModelChange)="setConfig('maxReportSize', +$event)"/></div><div class="form-group"><label class="form-label" for="dataset-tokenizer">Tokenizer</label><select id="dataset-tokenizer" class="form-select" [ngModel]="config().tokenizer" (ngModelChange)="setConfig('tokenizer', $event)"><option value="distilbert-base-uncased">distilbert-base-uncased</option><option value="bert-base-uncased">bert-base-uncased</option><option value="roberta-base">roberta-base</option></select></div><div class="form-group"><label class="form-label" for="dataset-name">Dataset Name (Optional)</label><input id="dataset-name" class="form-input" placeholder="Default: source name" [ngModel]="config().datasetName" (ngModelChange)="setConfig('datasetName', $event)"/></div></div><div class="processing-actions"><button type="button" class="btn btn-primary" (click)="buildDataset()" [disabled]="state().isProcessing"><ng-icon name="lucideSliders"/>{{ state().isProcessing ? 'Processing...' : 'Build Dataset' }}</button></div></div></div></div></section></div>
      @if (state().dbStatus) { <div class="dataset-db-status" role="status">{{ state().dbStatus?.message }}</div> }
    </div></main>
    @if (folderBrowserOpen()) { <div class="modal-backdrop" role="presentation" (click)="folderBrowserOpen.set(false)"><section class="modal folder-browser-modal" role="dialog" aria-modal="true" aria-labelledby="folder-title" appModalFocus (modalEscape)="folderBrowserOpen.set(false)" (click)="$event.stopPropagation()"><div class="modal-header"><h2 id="folder-title">Select image folder</h2><button type="button" class="btn-icon-small" aria-label="Close" (click)="folderBrowserOpen.set(false)"><ng-icon name="lucideX"/></button></div><div class="folder-path"><input class="form-input" [(ngModel)]="browsePath" placeholder="Server folder path"/><button type="button" class="btn btn-secondary" (click)="browse()">Browse</button></div>@if (browseError()) { <div class="upload-status error">{{ browseError() }}</div> }<div class="folder-items">@for (item of browseItems(); track item.path) { <button type="button" class="folder-item" (click)="item.is_dir ? navigateFolder(item.path) : null"><span>{{ item.is_dir ? '📁' : '🖼️' }}</span><span>{{ item.name }}</span><small>{{ item.is_dir ? item.image_count + ' images' : '' }}</small></button> }@if (!browseItems().length) { <p>No folders returned. Enter a server path and browse.</p> }</div><div class="modal-footer"><button type="button" class="btn btn-secondary" (click)="folderBrowserOpen.set(false)">Cancel</button><button type="button" class="btn btn-primary" [disabled]="!browsePath" (click)="selectFolder()">Use this folder</button></div></section></div> }
    <app-image-viewer [open]="!!viewerDataset()" [datasetName]="viewerDataset()" (closed)="closeViewer()"/>
    <app-validation-wizard [open]="validationWizardOpen()" [row]="validationWizardDataset()" [initialSelected]="selectedValidationMetrics()" (closed)="validationWizardOpen.set(false)" (confirmed)="confirmValidation($event)"/>
    <app-validation-report-modal [open]="reportOpen()" [datasetName]="reportDataset()?.name ?? null" [loading]="reportLoading()" [result]="reportResult()" [error]="reportError()" [progress]="reportProgress()" [status]="reportStatus()" [metadata]="reportMetadata()" (closed)="reportOpen.set(false)"/>
  `,
  styleUrl: '../styles/DatasetPage.css',
})
export class DatasetPage {
  private readonly api = inject(ApiService);
  private readonly appState = inject(AppStateService);
  private readonly storage = inject(StorageService);
  private readonly polling = inject(JobPollingService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  readonly state = this.appState.dataset;
  readonly folderBrowserOpen = signal(false);
  readonly viewerDataset = signal<string | null>(null);
  readonly validationWizardOpen = signal(false);
  readonly validationWizardDataset = signal<DatasetInfo | null>(null);
  readonly reportOpen = signal(false);
  readonly reportDataset = signal<DatasetInfo | null>(null);
  readonly reportLoading = signal(false);
  readonly reportResult = signal<ValidationResponse | null>(null);
  readonly reportError = signal<string | null>(null);
  readonly reportProgress = signal<number | null>(null);
  readonly reportStatus = signal<StoredValidationJob['status'] | null>(null);
  readonly reportMetadata = signal<{ date?: string | null; sampleSize?: number | null; metrics?: string[] } | null>(null);
  private readonly browsePathState = signal('');
  readonly browseItems = signal<DirectoryItem[]>([]);
  readonly browseError = signal<string | null>(null);
  readonly validationJobs = signal<Record<string, StoredValidationJob>>(this.storage.readRecord<StoredValidationJob>(VALIDATION_STORAGE_KEY));
  readonly datasets = computed(() => this.state().datasetNames?.datasets ?? []);
  readonly config = computed(() => this.state().config);
  constructor() { void this.refresh(); this.restoreValidationJobs(); }
  readonly canBrowse = computed(() => this.state().dbStatus?.allow_server_browse ?? true);
  get browsePath() { return this.browsePathState(); }
  set browsePath(value: string) { this.browsePathState.set(value); }
  async refresh() { const [status, names] = await Promise.all([this.api.getDatasetStatus(), this.api.getDatasetNames()]); if (status.result) this.appState.updateDataset((state) => ({ ...state, dbStatus: status.result })); if (names.result) this.appState.updateDataset((state) => ({ ...state, datasetNames: names.result })); }
  async refreshNames() { const result = await this.api.getDatasetNames(); if (result.result) this.appState.updateDataset((state) => ({ ...state, datasetNames: result.result })); }
  setConfig<K extends keyof DatasetProcessingConfig>(key: K, value: DatasetProcessingConfig[K]) { this.appState.updateDatasetConfig(key, value); }
  toggleDataset(name: string) { this.appState.updateDataset((state) => ({ ...state, selectedDatasets: state.selectedDatasets.includes(name) ? state.selectedDatasets.filter((item) => item !== name) : [...state.selectedDatasets, name] })); }
  async uploadFile(files: FileList | null) { const file = files?.[0]; if (!file) return; this.appState.updateDataset((state) => ({ ...state, datasetFile: file, datasetUpload: null, uploadError: null })); const result = await this.api.uploadDataset(file); if (result.result) this.appState.updateDataset((state) => ({ ...state, datasetUpload: result.result })); else this.appState.updateDataset((state) => ({ ...state, uploadError: result.error })); }
  async browse(path = this.browsePath) { const result = await this.api.browseDirectory(path); if (result.result) { this.browsePath = result.result.current_path; this.browseItems.set(result.result.items); this.browseError.set(null); } else this.browseError.set(result.error); }
  navigateFolder(path: string) { this.browsePath = path; void this.browse(path); }
  async selectFolder() { const path = this.browsePath; const result = await this.api.validateImagePath(path); this.appState.updateDataset((state) => ({ ...state, imageFolderPath: path, imageFolderName: path.split(/[\\/]/).filter(Boolean).pop() ?? path, imageValidation: result.result, uploadError: result.error })); if (result.result?.valid) this.folderBrowserOpen.set(false); }
  async loadDataset() { const current = this.state(); if (!current.imageValidation?.valid) { this.appState.updateDataset((state) => ({ ...state, uploadError: 'Please select an image folder first' })); return; } if (!current.datasetUpload?.success) { this.appState.updateDataset((state) => ({ ...state, uploadError: 'Please upload a dataset file first' })); return; } this.appState.updateDataset((state) => ({ ...state, isLoading: true, uploadError: null })); const result = await this.api.loadDataset({ image_folder_path: current.imageFolderPath, sample_size: current.config.sampleSize }); this.appState.updateDataset((state) => ({ ...state, loadResult: result.result, isLoading: false, uploadError: result.error })); await this.refresh(); }
  async buildDataset() { const current = this.state(); const datasetName = current.selectedDatasets[0]; if (!datasetName || current.selectedDatasets.length !== 1) { this.appState.updateDataset((state) => ({ ...state, uploadError: 'Select exactly one dataset to process.' })); return; } this.appState.updateDataset((state) => ({ ...state, isProcessing: true, uploadError: null })); const started = await this.api.processDataset({ dataset_name: datasetName, custom_name: current.config.datasetName || undefined, sample_size: current.config.sampleSize, validation_size: current.config.validationSize, tokenizer: current.config.tokenizer, max_report_size: current.config.maxReportSize }); if (!started.result) { this.appState.updateDataset((state) => ({ ...state, isProcessing: false, uploadError: started.error })); return; } this.polling.poll((jobId) => this.api.getPreparationJobStatus(jobId), started.result.job_id, 2).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => { if (['completed', 'failed', 'cancelled'].includes(status.status)) { this.appState.updateDataset((state) => ({ ...state, isProcessing: false, uploadError: status.status === 'completed' ? null : status.error ?? 'Processing failed' })); if (status.status === 'completed') { const parsed = this.api.parseProcessDatasetResponse(status.result ?? {}); this.appState.updateDataset((state) => ({ ...state, processingResult: parsed })); void this.refresh(); } } }); }
  async deleteDataset(name: string) { if (!confirm(`Delete dataset ${name}?`)) return; const result = await this.api.deleteDataset(name); if (!result.result) this.appState.updateDataset((state) => ({ ...state, uploadError: result.error })); await this.refresh(); }
  openViewer(datasetName: string) { this.viewerDataset.set(datasetName); }
  closeViewer() { this.viewerDataset.set(null); }
  selectedValidationMetrics(): ValidationMetric[] { const config = this.config(); return [config.pixDist ? 'pixels_distribution' : '', config.textStats ? 'text_statistics' : '', config.imgStats ? 'image_statistics' : ''].filter(Boolean) as ValidationMetric[]; }
  openValidationWizard(dataset: DatasetInfo) { this.validationWizardDataset.set(dataset); this.validationWizardOpen.set(true); }
  async confirmValidation(payload: ValidationWizardConfirmPayload) {
    this.validationWizardOpen.set(false);
    const dataset = payload.row;
    if (!dataset || !payload.metrics.length) return;
    this.reportDataset.set(dataset); this.reportOpen.set(true); this.reportLoading.set(true); this.reportResult.set(null); this.reportError.set(null); this.reportProgress.set(0); this.reportStatus.set('pending'); this.reportMetadata.set({ sampleSize: payload.sampleFraction, metrics: payload.metrics });
    const started = await this.api.runValidation({ dataset_name: dataset.name, metrics: payload.metrics, sample_size: payload.sampleFraction });
    if (!started.result) { this.reportLoading.set(false); this.reportStatus.set('failed'); this.reportError.set(started.error ?? 'Failed to start validation job'); return; }
    const job: StoredValidationJob = { jobId: started.result.job_id, metrics: payload.metrics, sampleSize: payload.sampleFraction, status: 'pending', progress: 0 };
    this.validationJobs.update((jobs) => ({ ...jobs, [dataset.name]: job })); this.storage.writeRecord(VALIDATION_STORAGE_KEY, this.validationJobs());
    this.pollValidationJob(dataset.name, job);
  }
  async openReport(dataset: DatasetInfo) {
    this.reportDataset.set(dataset); this.reportOpen.set(true); this.reportResult.set(null); this.reportError.set(null); this.reportMetadata.set(null); this.reportProgress.set(null); this.reportStatus.set(null); this.reportLoading.set(true);
    const active = this.validationJobs()[dataset.name];
    if (active && active.status !== 'completed' && active.status !== 'failed' && active.status !== 'cancelled') { this.reportMetadata.set({ sampleSize: active.sampleSize, metrics: active.metrics }); this.reportProgress.set(active.progress ?? 0); this.reportStatus.set(active.status ?? 'running'); this.pollValidationJob(dataset.name, active); return; }
    const result = await this.api.getValidationReport(dataset.name);
    if (!result.result) { this.reportLoading.set(false); this.reportError.set(result.error ?? 'Failed to load validation report'); return; }
    this.reportMetadata.set({ date: result.result.date, sampleSize: result.result.sample_size, metrics: result.result.metrics }); this.reportResult.set({ success: true, message: 'Validation report loaded', pixel_distribution: result.result.pixel_distribution, image_statistics: result.result.image_statistics, text_statistics: result.result.text_statistics }); this.reportProgress.set(100); this.reportStatus.set('completed'); this.reportLoading.set(false);
  }
  private pollValidationJob(datasetName: string, job: StoredValidationJob) {
    this.polling.poll((jobId) => this.api.getValidationJobStatus(jobId), job.jobId, 2).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => {
      const next = { ...job, status: status.status, progress: status.progress };
      this.validationJobs.update((jobs) => ({ ...jobs, [datasetName]: next })); this.storage.writeRecord(VALIDATION_STORAGE_KEY, this.validationJobs());
      if (this.reportDataset()?.name === datasetName) { this.reportProgress.set(status.progress ?? 0); this.reportStatus.set(status.status); this.reportLoading.set(!['completed', 'failed', 'cancelled'].includes(status.status)); }
      if (status.status === 'completed' && status.result && this.reportDataset()?.name === datasetName) { this.reportResult.set(this.api.parseValidationResponse(status.result)); this.reportLoading.set(false); }
      if (status.status !== 'completed' && ['failed', 'cancelled'].includes(status.status) && this.reportDataset()?.name === datasetName) { this.reportLoading.set(false); this.reportError.set(status.error ?? `Validation ${status.status}`); }
    });
  }
  private restoreValidationJobs() { for (const [datasetName, job] of Object.entries(this.validationJobs())) { if (job.status !== 'completed' && job.status !== 'failed' && job.status !== 'cancelled') this.pollValidationJob(datasetName, job); } }
  openValidation(dataset: DatasetInfo) { this.viewerDataset.set(null); void this.router.navigate(['/dataset/validate', dataset.name]); }
}
