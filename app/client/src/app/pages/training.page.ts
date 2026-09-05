import { CommonModule } from '@angular/common';
import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideActivity, lucideBarChart2, lucideChevronDown, lucideChevronUp, lucideInfo, lucideLoaderCircle, lucidePlay, lucideRefreshCw, lucideRotateCcw, lucideTrash2, lucideX } from '@ng-icons/lucide';
import { asRecord, readNumber, readNumberArray, readStringArray } from '../common/parsers';
import { DatasetApiService } from '../services/dataset-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import { AppStateService } from '../services/app-state.service';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import type { CheckpointInfo, DatasetInfo, StartTrainingConfig } from '../types/trainingApi';
import type { CheckpointEvaluationReport } from '../types/inferenceApi';
import type { JobLifecycleStatus } from '../types/jobs';
import type { ChartDataPoint, TrainingDashboardState } from '../types';
import { CheckpointEvaluationReportModalComponent } from '../components/checkpoint-evaluation-report-modal.component';
import { EvaluationWizardComponent, EvaluationWizardConfirmPayload } from '../components/evaluation-wizard.component';
import { ModalFocusDirective } from '../components/modal-focus.directive';
import { NewTrainingWizardComponent } from '../components/new-training-wizard.component';
import { TrainingDashboardComponent } from '../components/training-dashboard.component';
import { HelpPopoverComponent } from '../components/help-popover.component';

interface StoredEvaluationJob { jobId: string; metrics: string[]; metricConfigs?: Record<string, { dataFraction: number }>; status?: JobLifecycleStatus; progress?: number; }
type TrainingCheckpoint = Omit<CheckpointInfo, 'epochs' | 'loss' | 'val_loss'> & { epochs: number; loss: number; val_loss: number };
@Component({
  standalone: true,
  selector: 'app-training-page',
  imports: [CommonModule, ReactiveFormsModule, NgIcon, CheckpointEvaluationReportModalComponent, EvaluationWizardComponent, ModalFocusDirective, NewTrainingWizardComponent, TrainingDashboardComponent, HelpPopoverComponent],
  providers: [provideIcons({ lucideActivity, lucideBarChart2, lucideChevronDown, lucideChevronUp, lucideInfo, lucideLoaderCircle, lucidePlay, lucideRefreshCw, lucideRotateCcw, lucideTrash2, lucideX })],
  template: `
        <main class="training-container"><div class="header"><h1>XREPORT Transformer</h1><p>Configure and monitor your training sessions</p></div><div class="training-panels">
    <section class="training-panel" [class.collapsed]="collapsedNew()"><div class="panel-left"><div class="panel-header"><div><h3>New Training Session</h3><app-help-popover label="Explain starting a new run" title="Start a new training run" body="Select a processed dataset, open the five-step wizard, review the model, schedule, device, and checkpoint-saving choices, then start the background run." /><p>Select a processed dataset to configure your next run.</p></div><button type="button" class="panel-collapse-toggle" [attr.aria-expanded]="!collapsedNew()" (click)="collapsedNew.set(!collapsedNew())"><ng-icon [name]="collapsedNew() ? 'lucideChevronDown' : 'lucideChevronUp'"/></button><button type="button" class="panel-refresh" title="Refresh datasets" (click)="loadDatasets()"><ng-icon name="lucideRefreshCw"/></button></div>@if (!collapsedNew()) { <div class="panel-collapsible-content"><div class="panel-list" data-guidance-target="training-dataset-list">@if (!datasets().length) { <div class="panel-empty">No datasets available yet.</div> }@for (dataset of datasets(); track dataset.name) { <div class="panel-row" [class.selected]="selectedDataset()?.name === dataset.name"><button type="button" class="panel-row-main panel-row-main-button" [attr.aria-pressed]="selectedDataset()?.name === dataset.name" (click)="selectDataset(dataset)"><span class="panel-row-title">{{ dataset.name }}</span><span class="panel-row-count">{{ dataset.row_count | number }} rows</span></button><div class="panel-row-actions"><button type="button" class="icon-button" title="Show metadata" (click)="showDatasetMetadata(dataset)"><ng-icon name="lucideInfo"/></button><button type="button" class="icon-button danger" title="Delete dataset" (click)="deleteDataset(dataset)"><ng-icon name="lucideTrash2"/></button></div></div> }</div></div> }</div><div class="panel-right"><div class="panel-card" data-guidance-target="training-new-action"><div class="panel-card-header"><div class="panel-card-title-row"><ng-icon name="lucidePlay"/><h4>Initialize Training</h4></div><p>Launch the configuration wizard to set up your training run.</p></div><div class="panel-card-summary"><span>Selected Dataset</span><strong>{{ selectedDataset()?.name || 'None selected' }}</strong><span>Samples</span><strong>{{ selectedDataset() ? (selectedDataset()!.row_count | number) : 'N/A' }}</strong></div><button type="button" class="btn btn-primary" (click)="newWizardOpen.set(true)" [disabled]="!selectedDataset()"><ng-icon name="lucidePlay"/>Configure Training</button></div></div></section>
    <section class="training-panel" [class.collapsed]="collapsedResume()"><div class="panel-left"><div class="panel-header"><div class="panel-heading-copy"><div class="panel-title-row"><h3>Resume Training</h3><app-help-popover label="Explain resume training" title="Resume training" body="Resume continues a saved training state. Select a checkpoint, add epochs, and continue without starting from scratch." /></div><p>Pick a checkpoint to continue training from a saved state.</p></div><button type="button" class="panel-collapse-toggle" [attr.aria-expanded]="!collapsedResume()" (click)="collapsedResume.set(!collapsedResume())"><ng-icon [name]="collapsedResume() ? 'lucideChevronDown' : 'lucideChevronUp'"/></button><button type="button" class="panel-refresh" title="Refresh checkpoints" (click)="loadCheckpoints()"><ng-icon name="lucideRefreshCw"/></button></div>@if (!collapsedResume()) { <div class="panel-collapsible-content"><div class="panel-list" data-guidance-target="training-resume-list">@if (!checkpoints().length) { <div class="panel-empty">No checkpoints available yet.</div> }@for (checkpoint of checkpoints(); track checkpoint.name) { <div class="panel-row" [class.selected]="selectedCheckpoint() === checkpoint.name"><button type="button" class="panel-row-main panel-row-main-button" [attr.aria-pressed]="selectedCheckpoint() === checkpoint.name" (click)="selectedCheckpoint.set(checkpoint.name)"><span class="panel-row-title">{{ checkpoint.name }}</span><span class="panel-row-meta">{{ checkpoint.epochs }} epochs · loss {{ checkpoint.loss | number:'1.0-4' }}</span></button><div class="panel-row-actions"><button type="button" class="icon-button" title="Show metadata" (click)="showCheckpointMetadata(checkpoint)"><ng-icon name="lucideInfo"/></button><button type="button" class="icon-button" title="View evaluation report" (click)="openEvaluationReport(checkpoint)"><ng-icon name="lucideBarChart2"/></button><button type="button" class="icon-button danger" title="Delete checkpoint" (click)="deleteCheckpoint(checkpoint)"><ng-icon name="lucideTrash2"/></button></div></div> }</div></div> }</div><div class="panel-right"><div class="panel-card" data-guidance-target="training-resume-action"><div class="panel-card-header"><div class="panel-card-title-row"><ng-icon name="lucideRotateCcw"/><h4>Checkpoint Actions</h4></div><p>Resume training or evaluate the performance of the selected checkpoint.</p></div><div class="panel-card-summary"><span>Selected Checkpoint</span><strong>{{ selectedCheckpoint() || 'None selected' }}</strong><span>Epochs</span><strong>{{ selectedCheckpointInfo()?.epochs || 'N/A' }}</strong></div><div class="panel-card-actions"><button type="button" class="btn btn-primary panel-card-action-btn" (click)="resumeWizardOpen.set(true)" [disabled]="!selectedCheckpointInfo()"><ng-icon name="lucideRotateCcw"/>Resume Training</button><button type="button" class="btn btn-secondary panel-card-action-btn" (click)="openEvaluationWizard(selectedCheckpointInfo()!)" [disabled]="!selectedCheckpointInfo()"><ng-icon name="lucideActivity"/>Evaluate Model</button></div></div></div></section>
    </div><app-training-dashboard [dashboardState]="dashboard()" [error]="trainingError()" (stopRequested)="stopTraining()"/></main>
    <app-new-training-wizard [open]="newWizardOpen()" [datasetLabel]="selectedDataset()?.name ?? ''" [form]="trainingForm" [isLoading]="isLoading()" [error]="trainingError()" (closed)="newWizardOpen.set(false)" (submitted)="startTraining()"/>
    @if (resumeWizardOpen()) { <div class="training-modal-backdrop" role="presentation"><section class="modal training-wizard-modal" role="dialog" aria-modal="true" aria-labelledby="resume-training-title" appModalFocus (modalEscape)="resumeWizardOpen.set(false)"><div class="modal-header"><h2 id="resume-training-title">Resume Training</h2><button type="button" class="btn-icon-small" aria-label="Close" (click)="resumeWizardOpen.set(false)" [disabled]="isLoading()"><ng-icon name="lucideX"/></button></div><p>Configure the continuation for: <strong>{{ selectedCheckpoint() }}</strong></p><div class="training-wizard-body" [formGroup]="resumeForm"><div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideActivity"/><span>Training Schedule</span></div><div class="wizard-compact-grid"><label class="form-group"><span class="form-label">Additional Epochs</span><input class="form-input" type="number" min="1" formControlName="additionalEpochs"/></label></div>@if (selectedCheckpointInfo(); as checkpoint) { <div class="wizard-summary"><div><span>Starting Epoch</span><strong>{{ checkpoint.epochs }}</strong></div><div><span>Total Epochs</span><strong>{{ checkpoint.epochs + (resumeForm.get('additionalEpochs')?.value || 0) }}</strong></div></div> }</div></div>@if (trainingError()) { <div class="upload-status error">{{ trainingError() }}</div> }<div class="modal-footer training-wizard-footer"><div class="wizard-actions"><button type="button" class="btn btn-secondary" (click)="resumeWizardOpen.set(false)" [disabled]="isLoading()">Cancel</button><button type="button" class="btn btn-primary" (click)="resumeTraining()" [disabled]="isLoading() || resumeForm.invalid"><ng-icon name="lucideRotateCcw"/>{{ isLoading() ? 'Resuming…' : 'Resume Training' }}</button></div></div></section></div> }
    <app-evaluation-wizard [open]="evaluationOpen()" [checkpointName]="evaluationCheckpoint()?.name ?? ''" (closed)="closeEvaluationWizard()" (confirmed)="runEvaluation($event)"/>
    @if (metadataModal()) { <div class="metadata-backdrop" role="presentation" (click)="metadataModal.set(null)"><section class="metadata-modal" role="dialog" aria-modal="true" aria-labelledby="metadata-title" appModalFocus (modalEscape)="metadataModal.set(null)" (click)="$event.stopPropagation()"><header class="metadata-header"><div><h3 id="metadata-title">{{ metadataModal()?.title }}</h3><p class="metadata-subtitle">{{ metadataModal()?.subtitle }}</p></div><button type="button" class="metadata-close" aria-label="Close metadata dialog" (click)="metadataModal.set(null)"><ng-icon name="lucideX"/></button></header><div class="metadata-body"><pre class="metadata-value">{{ metadataModal()?.body }}</pre></div><footer class="metadata-footer"><button type="button" class="btn btn-secondary" (click)="metadataModal.set(null)">Close</button></footer></section></div> }
    <app-checkpoint-evaluation-report-modal [open]="evaluationReportOpen()" [checkpoint]="evaluationReportCheckpoint()?.name ?? null" [loading]="evaluationReportLoading()" [error]="evaluationReportError()" [progress]="evaluationReportProgress()" [report]="evaluationReport()" (closed)="closeEvaluationReport()"/>
  `,
  styleUrls: ['../styles/TrainingPage.css', '../styles/MetadataModal.css'],
})
export class TrainingPage {
  private readonly api = inject(TrainingApiService);
  private readonly datasetApi = inject(DatasetApiService);
  private readonly validationApi = inject(ValidationApiService);
  private readonly appState = inject(AppStateService);
  private readonly polling = inject(JobPollingService);
  private readonly jobsApi = inject(JobsApiService);
  private readonly destroyRef = inject(DestroyRef);
  readonly state = this.appState.training;
  readonly datasets = signal<DatasetInfo[]>([]);
  readonly checkpoints = signal<TrainingCheckpoint[]>([]);
  readonly selectedDataset = signal<DatasetInfo | null>(null);
  readonly selectedCheckpoint = signal('');
  readonly collapsedNew = signal(false);
  readonly collapsedResume = signal(false);
  readonly newWizardOpen = signal(false);
  readonly resumeWizardOpen = signal(false);
  readonly evaluationOpen = signal(false);
  readonly metadataModal = signal<{ title: string; subtitle: string; body: string } | null>(null);
  readonly isLoading = signal(false);
  readonly trainingError = signal<string | null>(null);
  readonly evaluationCheckpoint = signal<CheckpointInfo | null>(null);
  readonly evaluationReportCheckpoint = signal<CheckpointInfo | null>(null);
  readonly evaluationReportOpen = signal(false);
  readonly evaluationReportLoading = signal(false);
  readonly evaluationReportError = signal<string | null>(null);
  readonly evaluationReportProgress = signal<number | null>(null);
  readonly evaluationReport = signal<CheckpointEvaluationReport | null>(null);
  readonly evaluationJobs = signal<Record<string, StoredEvaluationJob>>({});
  private activeJobId: string | null = null;
  readonly trainingForm = inject(NonNullableFormBuilder).group({
    epochs: [this.state().config.epochs, [Validators.required, Validators.min(1)]],
    batchSize: [this.state().config.batchSize, [Validators.required, Validators.min(1)]],
    numEncoders: [this.state().config.numEncoders, [Validators.required, Validators.min(1)]],
    numDecoders: [this.state().config.numDecoders, [Validators.required, Validators.min(1)]],
    embeddingDims: [this.state().config.embeddingDims, [Validators.required, Validators.min(1)]],
    attnHeads: [this.state().config.attnHeads, [Validators.required, Validators.min(1)]],
    trainTemp: [this.state().config.trainTemp, [Validators.required, Validators.min(0)]],
    freezeImgEncoder: [this.state().config.freezeImgEncoder],
    useImgAugment: [this.state().config.useImgAugment],
    shuffleWithBuffer: [this.state().config.shuffleWithBuffer],
    shuffleBufferSize: [this.state().config.shuffleBufferSize, [Validators.required, Validators.min(1)]],
    saveCheckpoints: [this.state().config.saveCheckpoints],
    useScheduler: [this.state().config.useScheduler],
    targetLR: [this.state().config.targetLR, [Validators.required, Validators.min(0)]],
    warmupSteps: [this.state().config.warmupSteps, [Validators.required, Validators.min(0)]],
    useGpu: [this.state().config.useGpu],
    gpuId: [this.state().config.gpuId, [Validators.required, Validators.min(0)]],
    jitCompile: [this.state().config.jitCompile],
    jitBackend: [this.state().config.jitBackend],
    useMixedPrecision: [this.state().config.useMixedPrecision],
    dataloaderWorkers: [this.state().config.dataloaderWorkers, [Validators.required, Validators.min(0)]],
    prefetchFactor: [this.state().config.prefetchFactor, [Validators.required, Validators.min(1)]],
    pinMemory: [this.state().config.pinMemory],
    persistentWorkers: [this.state().config.persistentWorkers],
    realTimePlot: [this.state().config.realTimePlot],
  });
  readonly resumeForm = inject(NonNullableFormBuilder).group({ additionalEpochs: [50, [Validators.required, Validators.min(1)]] });
  trainingConfig = { ...this.state().config };
  readonly dashboard = computed(() => this.state().dashboardState);
  readonly selectedCheckpointInfo = computed(() => this.checkpoints().find((checkpoint) => checkpoint.name === this.selectedCheckpoint()) ?? null);
  constructor() { void this.loadDatasets(); void this.loadCheckpoints(); void this.restoreActiveTraining(); }
  async loadDatasets() { const result = await this.datasetApi.getProcessedNames(); if (result.result) { this.datasets.set(result.result.datasets); const selected = this.selectedDataset(); if (selected && !result.result.datasets.some((dataset) => dataset.name === selected.name)) { this.selectedDataset.set(null); this.newWizardOpen.set(false); this.metadataModal.set(null); } } }
  async loadCheckpoints() { const result = await this.api.getCheckpoints(); if (result.result) { const checkpoints = result.result.checkpoints.map((checkpoint) => ({ ...checkpoint, epochs: checkpoint.epochs ?? 0, loss: checkpoint.loss ?? 0, val_loss: checkpoint.val_loss ?? 0 })); this.checkpoints.set(checkpoints); const selected = this.selectedCheckpoint(); if (selected && !checkpoints.some((checkpoint) => checkpoint.name === selected)) { this.selectedCheckpoint.set(''); this.resumeWizardOpen.set(false); this.evaluationOpen.set(false); this.evaluationCheckpoint.set(null); this.evaluationReportCheckpoint.set(null); this.evaluationReportOpen.set(false); this.evaluationReport.set(null); this.evaluationReportError.set(null); } } }
  selectDataset(dataset: DatasetInfo) { this.selectedDataset.set(dataset); }
  async showDatasetMetadata(dataset: DatasetInfo) { const result = await this.datasetApi.getProcessingMetadata(dataset.name); this.metadataModal.set({ title: 'Dataset Metadata', subtitle: dataset.name, body: result.result ? JSON.stringify(result.result.metadata, null, 2) : result.error ?? 'No metadata found' }); }
  async deleteDataset(dataset: DatasetInfo) { if (!confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) return; const result = await this.datasetApi.deleteDataset(dataset.name); if (!result.result?.success) this.trainingError.set(result.error ?? result.result?.message ?? 'Failed to delete dataset'); await this.loadDatasets(); }
  async showCheckpointMetadata(checkpoint: CheckpointInfo) { const result = await this.api.getCheckpointMetadata(checkpoint.name); this.metadataModal.set({ title: 'Checkpoint Metadata', subtitle: checkpoint.name, body: result.result ? JSON.stringify(result.result, null, 2) : result.error ?? 'No metadata found' }); }
  async deleteCheckpoint(checkpoint: CheckpointInfo) { if (!confirm(`Delete checkpoint "${checkpoint.name}"? This cannot be undone.`)) return; const result = await this.api.deleteCheckpoint(checkpoint.name); if (!result.result?.success) this.trainingError.set(result.error ?? result.result?.message ?? 'Failed to delete checkpoint'); await this.loadCheckpoints(); }
  private startConfig(checkpointId?: string): StartTrainingConfig { const config = { ...this.trainingConfig, ...this.trainingForm.getRawValue() }; return { dataset_name: this.selectedDataset()?.name ?? '', epochs: config.epochs, batch_size: config.batchSize, num_encoders: config.numEncoders, num_decoders: config.numDecoders, embedding_dims: config.embeddingDims, attention_heads: config.attnHeads, train_temp: config.trainTemp, freeze_img_encoder: config.freezeImgEncoder, use_img_augmentation: config.useImgAugment, shuffle_with_buffer: config.shuffleWithBuffer, shuffle_size: config.shuffleBufferSize, save_checkpoints: config.saveCheckpoints, checkpoint_id: checkpointId, use_device_GPU: config.useGpu, device_ID: config.gpuId, jit_compile: config.jitCompile, jit_backend: config.jitBackend, use_mixed_precision: config.useMixedPrecision, dataloader_workers: config.dataloaderWorkers, prefetch_factor: config.prefetchFactor, pin_memory: config.pinMemory, persistent_workers: config.persistentWorkers, plot_training_metrics: config.realTimePlot, use_scheduler: config.useScheduler, target_LR: config.targetLR, warmup_steps: config.warmupSteps }; }
  async startTraining() { if (!this.selectedDataset()) return; if (this.trainingForm.invalid) { this.trainingForm.markAllAsTouched(); this.trainingError.set('Enter positive values for all training parameters.'); return; } this.isLoading.set(true); this.trainingError.set(null); const started = await this.api.start(this.startConfig()); if (!started.result) { this.isLoading.set(false); this.trainingError.set(started.error ?? 'Training failed'); return; } this.newWizardOpen.set(false); this.isLoading.set(false); this.pollTraining(started.result.job_id); }
  async resumeTraining() { const checkpoint = this.selectedCheckpointInfo(); if (!checkpoint) return; if (this.resumeForm.invalid) { this.resumeForm.markAllAsTouched(); this.trainingError.set('Additional epochs must be at least 1.'); return; } this.isLoading.set(true); const started = await this.api.resume(checkpoint.name, this.resumeForm.getRawValue().additionalEpochs); if (!started.result) { this.isLoading.set(false); this.trainingError.set(started.error ?? 'Resume training failed'); return; } this.resumeWizardOpen.set(false); this.isLoading.set(false); this.pollTraining(started.result.job_id); }
  private pollTraining(jobId: string) {
    this.activeJobId = jobId;
    this.appState.updateDashboard({ isTraining: true, currentEpoch: 0, progressPercent: 0, chartData: [], availableMetrics: [], epochBoundaries: [], logEntries: [`Training job started (${jobId}).`] });
    this.polling.poll((id) => this.jobsApi.get(id), jobId, 2).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => {
      const result = asRecord(status.result);
      if (result) {
        const current = this.state().dashboardState;
        const dashboardUpdate: Partial<TrainingDashboardState> = {
          isTraining: !['completed', 'failed', 'cancelled'].includes(status.status),
          currentEpoch: readNumber(result['current_epoch']) ?? current.currentEpoch,
          totalEpochs: readNumber(result['total_epochs']) ?? current.totalEpochs,
          loss: readNumber(result['loss']) ?? current.loss,
          valLoss: readNumber(result['val_loss']) ?? current.valLoss,
          accuracy: readNumber(result['accuracy']) ?? current.accuracy,
          valAccuracy: readNumber(result['val_accuracy']) ?? current.valAccuracy,
          progressPercent: readNumber(result['progress_percent']) ?? status.progress ?? current.progressPercent,
          elapsedSeconds: readNumber(result['elapsed_seconds']) ?? current.elapsedSeconds,
        };
        const chartData = this.parseChartData(result['chart_data']);
        const availableMetrics = readStringArray(result['available_metrics']);
        const epochBoundaries = readNumberArray(result['epoch_boundaries']);
        if (chartData) dashboardUpdate.chartData = chartData;
        if (availableMetrics) dashboardUpdate.availableMetrics = availableMetrics;
        if (epochBoundaries) dashboardUpdate.epochBoundaries = epochBoundaries;
        this.appState.updateDashboard(dashboardUpdate);
      }
      if (['completed', 'failed', 'cancelled'].includes(status.status)) {
        const terminalMessage = status.status === 'completed' ? 'Training completed successfully.' : `Training ${status.status}: ${status.error ?? 'Unknown error'}`;
        this.appState.updateDashboard((current) => ({ ...current, isTraining: false, logEntries: [...current.logEntries, terminalMessage].slice(-200) }));
        if (status.status !== 'completed') this.trainingError.set(status.error ?? `Training ${status.status}`);
        this.activeJobId = null;
        void this.loadCheckpoints();
      }
    });
  }
  private parseChartData(value: unknown): ChartDataPoint[] | undefined { if (!Array.isArray(value)) return undefined; const points = value.filter((entry): entry is Record<string, unknown> => Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry)).map((entry) => { const point: ChartDataPoint = { batch: readNumber(entry['batch']) ?? 0 }; for (const [key, raw] of Object.entries(entry)) { if (key !== 'batch' && typeof raw === 'number') point[key] = raw; } return point; }); return points.length ? points : undefined; }
  async stopTraining() { if (!this.activeJobId) return; const response = await this.jobsApi.cancel(this.activeJobId); if (!response.result?.success) this.trainingError.set(response.error ?? response.result?.message ?? 'Unable to cancel training.'); }
  openEvaluationWizard(checkpoint: CheckpointInfo) { if (!checkpoint) return; this.evaluationCheckpoint.set(checkpoint); this.evaluationOpen.set(true); }
  closeEvaluationWizard() { this.evaluationOpen.set(false); this.evaluationCheckpoint.set(null); }
  async openEvaluationReport(checkpoint: CheckpointInfo) { if (!checkpoint) return; this.evaluationReportCheckpoint.set(checkpoint); this.evaluationReport.set(null); this.evaluationReportError.set(null); this.evaluationReportProgress.set(null); this.evaluationReportLoading.set(true); this.evaluationReportOpen.set(true); const report = await this.validationApi.getCheckpointEvaluationReport(checkpoint.name); this.evaluationReportLoading.set(false); if (report.result) this.evaluationReport.set(report.result); else this.evaluationReportError.set(report.error ?? 'No evaluation report is available for this checkpoint.'); }
  closeEvaluationReport() { this.evaluationReportOpen.set(false); this.evaluationReportCheckpoint.set(null); }
  async runEvaluation(payload: EvaluationWizardConfirmPayload) { const checkpoint = this.evaluationCheckpoint(); if (!checkpoint) return; const metricConfigsForApi = Object.fromEntries(Object.entries(payload.metricConfigs).map(([key, value]) => [key, { data_fraction: value.dataFraction }])); this.closeEvaluationWizard(); this.evaluationReportCheckpoint.set(checkpoint); this.evaluationReport.set({ checkpoint: checkpoint.name, metrics: payload.metrics, metric_configs: metricConfigsForApi }); this.evaluationReportError.set(null); this.evaluationReportProgress.set(0); this.evaluationReportLoading.set(true); this.evaluationReportOpen.set(true); const started = await this.validationApi.evaluateCheckpoint(checkpoint.name, payload.metrics, 10, metricConfigsForApi); if (!started.result) { this.evaluationReportLoading.set(false); this.evaluationReportError.set(started.error ?? 'Failed to start evaluation job'); return; } this.evaluationJobs.update((jobs) => ({ ...jobs, [checkpoint.name]: { jobId: started.result!.job_id, metrics: payload.metrics, metricConfigs: payload.metricConfigs, status: 'pending', progress: 0 } })); this.polling.poll((jobId) => this.jobsApi.get(jobId), started.result.job_id, 2).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(async (status) => { if (!['completed', 'failed', 'cancelled'].includes(status.status)) { this.evaluationReportProgress.set(status.progress ?? 0); return; } this.evaluationJobs.update((jobs) => ({ ...jobs, [checkpoint.name]: { ...(jobs[checkpoint.name] ?? { jobId: started.result!.job_id, metrics: payload.metrics, metricConfigs: payload.metricConfigs }), status: status.status, progress: status.progress } })); this.evaluationReportProgress.set(status.progress ?? 100); this.evaluationReportLoading.set(false); if (status.status !== 'completed') { this.evaluationReportError.set(status.error ?? `Evaluation ${status.status}`); return; } const report = await this.validationApi.getCheckpointEvaluationReport(checkpoint.name); if (report.result) this.evaluationReport.set(report.result); else this.evaluationReportError.set(report.error ?? 'Failed to load evaluation report'); }); }

  private async restoreActiveTraining(): Promise<void> {
    const response = await this.jobsApi.list('training');
    const active = response.result?.jobs.find((job) => job.status === 'pending' || job.status === 'running');
    if (active) this.pollTraining(active.job_id);
  }
}
