import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, computed, DestroyRef, effect, ElementRef, HostListener, inject, signal, untracked } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideAlertTriangle, lucideCheck, lucideChevronLeft, lucideChevronRight,
  lucideCopy, lucideDownload, lucideFileImage, lucideImagePlus, lucideLoaderCircle,
  lucideRefreshCw, lucideSearch, lucideSparkles, lucideTrash2, lucideX,
} from '@ng-icons/lucide';
import { asRecord, readNumber, readString, readStringArray } from '@xreport/shared/app/common/parsers';
import { SECTION_LABELS, formatReportDraft, isOutputSection, parseReportDraft } from '@xreport/shared/app/common/report-sections';
import type { DraftSections } from '@xreport/shared/app/common/report-sections';
import { InferenceApiService } from '@xreport/shared/app/services/inference-api.service';
import { AppStateService } from '@xreport/shared/app/services/app-state.service';
import { NotificationService } from '@xreport/shared/app/services/notification.service';
import { JobPollingService } from '@xreport/shared/app/services/job-polling.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';
import { GuidanceService, INFERENCE_TOUR_ID } from '@xreport/shared/app/services/guidance.service';
import { markStartupPhase } from '@xreport/shared/app/services/startup-timing';
import type { GenerationProfile, InferenceHistoryDetail, InferenceHistoryReport, ModelAvailability, OutputSection } from '@xreport/shared/app/types/inferenceApi';
import type { InferenceStudy } from '@xreport/shared/app/types';
import { CheckpointImportComponent } from '../components/checkpoint-import.component';
import { FeatureTipComponent } from '@xreport/shared/app/components/feature-tip.component';
import { HelpPopoverComponent } from '@xreport/shared/app/components/help-popover.component';
import { ReportDraftEditorComponent } from '../components/report-draft-editor.component';

@Component({
  standalone: true,
  selector: 'app-inference-page',
  imports: [CommonModule, FormsModule, NgIcon, RouterLink, CheckpointImportComponent, FeatureTipComponent, HelpPopoverComponent, ReportDraftEditorComponent],
  providers: [provideIcons({ lucideAlertTriangle, lucideCheck, lucideChevronLeft, lucideChevronRight, lucideCopy, lucideDownload, lucideFileImage, lucideImagePlus, lucideLoaderCircle, lucideRefreshCw, lucideSearch, lucideSparkles, lucideTrash2, lucideX })],
  template: `
    <main class="inference-workspace page-container">
      <header class="workspace-heading"><div class="workspace-title"><h1>Turn a radiograph into a draft report</h1><p>Select a local model, provide the study, then review the generated text before it leaves the workspace.</p></div><div class="research-warning" role="alert"><ng-icon name="lucideAlertTriangle" aria-hidden="true"/><div><strong>Research use only</strong><span>Models and generated drafts are not clinically approved. Qualified review and independent verification are required.</span></div></div></header>
      <app-feature-tip guidanceId="inference-first-use" [version]="1" title="A quick way to get started" message="Choose a model, add a study image, then review the editable draft. The walkthrough is optional." [showAction]="true" actionLabel="Show me" (action)="startInferenceTour()" />
      <section class="workflow-stack" aria-label="Inference workflow">
        <section class="workflow-step model-step" aria-labelledby="model-step-title">
          <div class="step-header"><div class="step-title"><span class="step-number">1</span><div><h2 id="model-step-title">Choose or prepare a model</h2><p>Only models exposed by the local catalog can be selected.</p></div></div></div>
          <div class="model-selection">
            <aside class="catalog-panel" aria-label="Model catalog" data-guidance-target="inference-model-catalog">
              <div class="catalog-heading"><div><span class="catalog-title"><strong>Model catalogue</strong><app-help-popover label="About model rows" title="About model rows" body="Start with Ready or Download available. Each row reports the model's readiness, size or validation, and recommended or selected state; the details panel shows anatomy, image limit, context support and output sections." /></span><span>{{ filteredModels().length }} models · {{ publicModels().length }} public · {{ customModels().length }} custom</span></div><button type="button" class="secondary-button" (click)="importOpen.set(true)">Import checkpoint</button></div>
      <app-checkpoint-import [open]="importOpen()" (closed)="importOpen.set(false)" (imported)="onCheckpointImported()" />
              <label class="search-field"><ng-icon name="lucideSearch" aria-hidden="true"/><span class="sr-only">Filter models</span><input [(ngModel)]="modelFilterValue" placeholder="Filter by model, anatomy, or origin"/></label>
              @if (state().isLoadingModels) { <div class="catalog-state"><ng-icon name="lucideLoaderCircle" class="spin"/>Discovering local models…</div> }
              @if (catalogError()) { <div class="catalog-state error" role="alert">{{ catalogError() }}</div> }
              <div class="model-groups" role="region" aria-label="Scrollable model catalogue" tabindex="0">
                <section class="model-group"><div class="model-group-heading"><strong>Custom checkpoints</strong><span class="group-count">{{ customModels().length }}</span></div><div class="model-list">
                  @for (model of customModels(); track model.model_ref) { <button type="button" class="model-card" [class.selected]="state().selectedModelRef === model.model_ref" [attr.aria-pressed]="state().selectedModelRef === model.model_ref" [disabled]="state().isGenerating || saving() || loadingPersistedHistory() || maintenanceBusy()" (click)="selectModel(model)"><span class="status-dot" [class]="'status-dot ' + model.status"></span><span class="model-card-body"><strong>{{ model.display_name }}</strong><small>{{ model.validation_receipt_status === 'passed' ? 'Validated' : 'Artifact ' + pretty(model.validation_receipt_status) }} · {{ model.max_current_images }} images · {{ outputLabels(model.output_sections ?? []) }}</small></span><span class="model-card-end"><span class="status-label" [class]="'status-label ' + model.status">{{ pretty(model.status) }}</span>@if (state().selectedModelRef === model.model_ref) { <span class="model-card-selection">Selected</span> } @else if (model.recommended) { <span class="model-card-badge">Recommended</span> }</span></button> }
                  @if (!state().isLoadingModels && !customModels().length) { <div class="catalog-state compact">Import a checkpoint exported from XREPORT Studio to make it available here.</div> }
                </div></section>
                <section class="model-group"><div class="model-group-heading"><strong>Public models</strong><span class="group-count">{{ publicModels().length }}</span></div><div class="model-list">
                  @for (model of publicModels(); track model.model_ref) { <button type="button" class="model-card" [class.selected]="state().selectedModelRef === model.model_ref" [attr.aria-pressed]="state().selectedModelRef === model.model_ref" [disabled]="state().isGenerating || saving() || loadingPersistedHistory() || maintenanceBusy()" (click)="selectModel(model)"><span class="status-dot" [class]="'status-dot ' + model.status"></span><span class="model-card-body"><strong>{{ model.display_name }}</strong><small>{{ model.parameter_label || model.parameter_size || 'Public model' }} · {{ pretty(model.anatomy_coverage) }}</small></span><span class="model-card-end"><span class="status-label" [class]="'status-label ' + model.status">{{ pretty(model.status) }}</span>@if (state().selectedModelRef === model.model_ref) { <span class="model-card-selection">Selected</span> } @else if (model.recommended) { <span class="model-card-badge">Recommended</span> }</span></button> }
                  @if (!state().isLoadingModels && !publicModels().length) { <div class="catalog-state compact">No public models match this filter.</div> }
                </div></section>
              </div>
            </aside>
            @if (selectedModel(); as model) {
              <div class="model-details"><div class="model-details-top"><div><span class="provider-pill">{{ model.origin === 'custom' ? 'Custom checkpoint' : 'Public model' }}</span>@if (model.recommended) { <span class="recommended-pill">Recommended</span> }</div></div><h3>{{ model.display_name }}</h3><p class="model-description">{{ model.description }}</p>@if (modelNotice(); as notice) { <div class="model-info-banner" role="status"><ng-icon name="lucideAlertTriangle" aria-hidden="true"/><span>{{ notice }}</span><button type="button" class="model-info-dismiss" aria-label="Dismiss model information" title="Dismiss model information" (click)="dismissModelNotice()"><ng-icon name="lucideX" aria-hidden="true"/></button></div> }<dl class="model-meta-grid"><div><dt>Status</dt><dd>{{ pretty(model.status) }}</dd></div><div><dt>Installation</dt><dd>{{ installationLabel(model) }}</dd></div><div><dt>Validation</dt><dd>{{ pretty(model.validation_status) }}</dd></div><div><dt>Evidence</dt><dd>{{ pretty(model.validation_receipt_status) }}</dd></div><div><dt>Anatomy</dt><dd>{{ pretty(model.anatomy_coverage) }}</dd></div><div><dt>Hardware</dt><dd>{{ pretty(model.hardware_demand) }}</dd></div><div><dt>Input</dt><dd>{{ pretty(model.input_semantics) }}</dd></div><div><dt>Output</dt><dd>{{ outputLabels(model.output_sections ?? []) }}</dd></div>@if (model.license) { <div><dt>Licence</dt><dd>{{ model.license }}</dd></div> }<div class="meta-wide"><dt>Revision</dt><dd class="wrap-value" [attr.title]="modelRevision(model)">{{ modelRevision(model) | slice:0:12 }}</dd></div></dl><div class="capability-list">@for (capability of capabilities(model); track capability) { <span>{{ capability }}</span> }</div>@if (model.origin === 'public') { <div class="maintenance-controls">@if (model.access_policy === 'gated' && model.access_url) { <a class="text-button" [href]="model.access_url" target="_blank" rel="noreferrer">Open model access page</a> }@if (hasAction(model, 'download') || hasAction(model, 'repair')) { <button type="button" class="secondary-button" (click)="maintain(model, modelMaintenanceAction(model))" [disabled]="maintenanceBusy() || !(hasAction(model, 'download') || hasAction(model, 'repair'))">{{ maintenanceJobId() ? 'Downloading…' : modelMaintenanceLabel(model) }}</button> }<button type="button" class="secondary-button" (click)="checkUpdate(model)" [disabled]="maintenanceBusy()" [attr.aria-busy]="updateChecking()">{{ updateChecking() ? 'Checking…' : 'Check for updates' }}</button>@if (updateChecking()) { <div class="maintenance-progress" role="status" aria-live="polite"><div class="progress-heading"><span>Checking Hugging Face for a newer revision…</span></div><div class="progress-track indeterminate"><span></span></div></div> }@if (maintenanceJobId()) { <div class="maintenance-progress" role="status" aria-live="polite"><div class="progress-heading"><span>{{ maintenanceProgressMessage() }}</span><strong>{{ maintenanceProgress() | number:'1.0-0' }}%</strong></div><div class="progress-track"><span [style.width.%]="maintenanceProgress()"></span></div><small>{{ maintenanceProgressDetail() }}</small><button type="button" class="secondary-button cancel-button" (click)="cancelMaintenance()" [disabled]="maintenanceCancellationRequested()">{{ maintenanceCancellationRequested() ? 'Stopping download…' : 'Cancel download' }}</button></div> }@if (maintenanceMessage()) { <span role="status">{{ maintenanceMessage() }}</span> }</div> }</div>
            } @else { <div class="model-details model-details-empty"><strong>Select a model to inspect its contract.</strong><span>The local catalog reports readiness, installation, validation, and supported inputs here.</span></div> }
          </div>
        </section>
        <section class="workflow-step study-step" aria-labelledby="study-step-title"><div class="step-header"><div class="step-title"><span class="step-number">2</span><div><h2 id="study-step-title">Add a radiograph and run inference</h2><p>Use a supported image, optional clinical context, and a generation profile.</p></div></div>@if (state().images.length) { <span class="study-image-count">{{ state().images.length }} image{{ state().images.length === 1 ? '' : 's' }} ready</span> }</div><div class="study-layout"><div class="study-input"><div class="upload-zone" (dragover)="$event.preventDefault()" (drop)="onDrop($event)">@if (currentImageUrl()) { <div class="image-stage"><img [src]="currentImageUrl()" [alt]="'Study image ' + (state().currentIndex + 1)"/><span>{{ currentImage()?.name }}</span></div> } @else { <button type="button" class="upload-prompt" (click)="fileInput.click()" [disabled]="state().isGenerating || !selectedModel()"><ng-icon name="lucideImagePlus"/><strong>{{ selectedModel() ? 'Add study image' : 'Select a model first' }}</strong><span>Drop an image here or browse local files</span></button> }<input #fileInput class="sr-only" type="file" accept="image/*" [multiple]="maxImages() > 1" (change)="addFiles(fileInput.files)"/></div><div class="study-toolbar"><button type="button" class="secondary-button" (click)="fileInput.click()" [disabled]="!selectedModel() || state().isGenerating"><ng-icon name="lucideImagePlus"/>{{ state().images.length ? 'Replace / add image' : 'Browse images' }}</button><span>{{ maxImages() > 1 ? 'Up to ' + maxImages() + ' independent images' : 'Up to 1 current image' }}</span>@if (state().images.length) { <button type="button" class="text-button danger" (click)="resetStudy()" [disabled]="state().isGenerating"><ng-icon name="lucideTrash2"/>Clear study</button> }</div>@if (state().images.length > 1) { <div class="image-navigation"><button type="button" aria-label="Previous image" (click)="setIndex(state().currentIndex - 1)" [disabled]="state().currentIndex === 0"><ng-icon name="lucideChevronLeft"/></button><span>{{ state().currentIndex + 1 }} / {{ state().images.length }}</span><button type="button" aria-label="Next image" (click)="setIndex(state().currentIndex + 1)" [disabled]="state().currentIndex === state().images.length - 1"><ng-icon name="lucideChevronRight"/></button></div> }</div><div class="study-settings"><div class="settings-heading"><strong>Generation settings</strong><span>{{ selectedModel()?.capabilities?.clinical_context ? 'Context supported' : 'Context unavailable' }}</span></div><label class="field-label" for="clinical-context"><span>Clinical context</span><small>{{ selectedModel()?.capabilities?.clinical_context ? 'Optional' : 'Not supported by selected model' }}</small></label><textarea id="clinical-context" class="context-input" [(ngModel)]="clinicalContext" [disabled]="!selectedModel()?.capabilities?.clinical_context || state().isGenerating" placeholder="Indication, relevant history, comparison details…"></textarea><div class="generation-controls"><label for="profile-select">Generation profile</label><select id="profile-select" [(ngModel)]="generationProfile" [disabled]="state().isGenerating"><option value="deterministic">Deterministic</option><option value="concise">Concise</option><option value="detailed">Detailed</option></select></div>@if (profileNote(); as note) { <small class="control-note">{{ note }}</small> }@if (state().images.length > 1) { <div class="generation-controls"><label for="scope-select">Run on</label><select id="scope-select" [ngModel]="effectiveScope()" (ngModelChange)="setRunScope($event)" [disabled]="state().isGenerating"><option value="selected">Selected image only ({{ state().currentIndex + 1 }} of {{ state().images.length }})</option><option value="all">Each image separately · {{ state().images.length }} reports</option><option value="study" [disabled]="!studySupported()">All images as one study · 1 report{{ studySupported() ? '' : ' (not supported by this model)' }}</option></select></div><small class="control-note" data-testid="scope-description">{{ scopeDescription() }}</small> }<button type="button" class="generate-button" (click)="generate()" [disabled]="!canGenerate() || !state().images.length || state().isGenerating"><ng-icon name="lucideSparkles"/>{{ state().isGenerating ? 'Preparing and generating…' : selectedModel()?.status === 'ready' ? 'Generate draft' : selectedModel()?.origin === 'public' && (selectedModel()?.gated || selectedModel()?.access_policy === 'gated') ? 'Access required' : 'Download and generate' }}</button>@if (state().isGenerating) { <button type="button" class="secondary-button cancel-button" (click)="cancelGeneration()">Cancel generation</button> }@if (state().isGenerating) { <div class="generation-progress" role="status" aria-live="polite"><div class="progress-heading"><span>{{ progressMessage() }}</span><strong>{{ progress() | number:'1.0-0' }}%</strong></div><div class="progress-track"><span [style.width.%]="progress()"></span></div></div> }@if (generationError()) { <div class="generation-error" role="alert"><strong>{{ generationCancelled() ? 'Generation cancelled' : 'Generation could not complete' }}</strong><span>{{ generationError() }}</span></div> }</div></div></section>
        <section class="workflow-step report-step" aria-labelledby="report-step-title">
          <div class="step-header"><div class="step-title"><span class="step-number">3</span><div><h2 id="report-step-title">Review the generated draft</h2><p>Edit the declared output sections, then copy or export for qualified review.</p></div></div><div class="report-header-actions"><span class="step-state" [class.ready]="hasDraft()">{{ state().isGenerating ? 'Generating' : hasDraft() ? 'Draft ready' : 'No draft yet' }}</span><div class="draft-actions"><button type="button" aria-label="Regenerate draft" title="Regenerate" (click)="generate()" [disabled]="!canGenerate() || !state().images.length || state().isGenerating || saving() || loadingPersistedHistory()"><ng-icon name="lucideRefreshCw"/></button><button type="button" aria-label="Copy draft" title="Copy" (click)="copyDraft()" [disabled]="!hasDraft()">@if (state().isCopied) { <ng-icon name="lucideCheck"/> } @else { <ng-icon name="lucideCopy"/> }</button><button type="button" aria-label="Export draft" title="Export text" (click)="exportDraft()" [disabled]="!hasDraft()"><ng-icon name="lucideDownload"/></button></div></div></div>
          @if (!hasDraft() && !state().isGenerating) { <div class="draft-empty"><ng-icon name="lucideFileImage"/><strong>No draft yet</strong><span>The report will appear here after a model and radiograph are ready.</span></div> }
          @else {
            @if (reportEntries().length > 1) {
              <div class="report-navigator" role="group" aria-label="Generated reports">
                <button type="button" aria-label="Previous report" (click)="stepReport(-1)" [disabled]="reportPosition() <= 0"><ng-icon name="lucideChevronLeft"/></button>
                <select aria-label="Select generated report" [ngModel]="state().currentIndex" (ngModelChange)="setIndex(+$event)">
                  @for (entry of reportEntries(); track entry.index; let position = $index) { <option [ngValue]="entry.index">Report {{ position + 1 }} of {{ reportEntries().length }} · {{ entry.name }}{{ entry.dirty ? ' · unsaved edits' : '' }}{{ entry.hasReport ? '' : ' · no report' }}</option> }
                </select>
                <button type="button" aria-label="Next report" (click)="stepReport(1)" [disabled]="reportPosition() >= reportEntries().length - 1"><ng-icon name="lucideChevronRight"/></button>
              </div>
            } @else if (studyReport() && state().runIndices.length > 1) { <div class="report-scope-chip">Study report · {{ state().runIndices.length }} images read together</div> }
            <app-report-draft-editor [sections]="outputSections()" [drafts]="drafts()[state().currentIndex] ?? {}" [readOnly]="state().isGenerating || saving() || loadingPersistedHistory() || (hasDraft() && !persistedHistory())" (draftChange)="updateDraft($event.section, $event.value)" /> }
          <div class="draft-persistence" aria-live="polite">
            <div class="draft-persistence-status" role="status">
              @if (loadingPersistedHistory()) { <span>Confirming the saved report…</span> }
              @else if (saving()) { <span>Saving edits…</span> }
              @else if (currentDirty()) { <span>Unsaved changes</span> }
              @else if (dirty()) { <span>Other reports have unsaved changes</span> }
              @else if (persistedHistory()) { <span>{{ saveSuccess() ? 'Edits saved.' : 'Generated report saved.' }}</span> }
            </div>
            <div class="draft-persistence-actions">
              <button type="button" class="primary-button" (click)="saveEdits()" [disabled]="!currentDirty() || !persistedHistory() || saving() || loadingPersistedHistory() || state().isGenerating">{{ saving() ? 'Saving…' : 'Save edits' }}</button>
              @if (persistedRequestId() && !persistedHistory() && !loadingPersistedHistory()) { <button type="button" class="secondary-button" (click)="retryLoadPersistedHistory()">Retry history lookup</button> }
              @if (persistedHistory() && persistedRequestId()) { <a class="secondary-button" [routerLink]="['/reports', persistedRequestId()]">View in Reports</a> }
            </div>
            @if (saveError()) { <div class="draft-persistence-error" role="alert">{{ saveError() }}</div> }
          </div>
          @if (selectedModel() || hasDraft()) { <div class="runtime-metadata"><span><strong>Model</strong>{{ selectedModel()?.display_name || 'Not selected' }}</span><span><strong>Provider</strong>{{ selectedModel()?.provider || '—' }}</span><span><strong>Revision</strong>{{ selectedModel()?.active_revision || selectedModel()?.model_revision || 'Not reported' }}</span><span><strong>Profile</strong>{{ profileLabel() }}</span><span><strong>Output</strong>{{ outputLabels(outputSections()) }}</span></div> }
        </section>
      </section>
    </main>
  `,
  styleUrl: '../styles/InferencePage.css',
})
export class InferencePage implements AfterViewInit {
  private readonly element = inject(ElementRef<HTMLElement>);
  private readonly api = inject(InferenceApiService);
  private readonly appState = inject(AppStateService);
  private readonly polling = inject(JobPollingService);
  private readonly jobsApi = inject(JobsApiService);
  private readonly notifications = inject(NotificationService);
  private readonly guidance = inject(GuidanceService);
  private readonly destroyRef = inject(DestroyRef);
  readonly state = this.appState.inference;
  readonly modelFilter = signal('');
  readonly catalogError = signal<string | null>(null);
  readonly generationError = signal<string | null>(null);
  readonly generationCancelled = signal(false);
  readonly progress = signal(0);
  readonly progressMessage = signal('Preparing and generating…');
  readonly maintenanceBusy = signal(false);
  readonly updateChecking = signal(false);
  readonly maintenanceMessage = signal<string | null>(null);
  readonly maintenanceJobId = signal<string | null>(null);
  readonly maintenanceProgress = signal(0);
  readonly maintenanceProgressMessage = signal('Preparing model download…');
  readonly maintenanceProgressDetail = signal('');
  readonly maintenanceCancellationRequested = signal(false);
  readonly modelNoticeDismissedFor = signal<string | null>(null);
  private readonly currentUrl = signal<string | null>(null);
  // Study fields live in the selected model's study, so they survive navigation and model switches
  readonly drafts = this.studyField('drafts');
  readonly baselineDrafts = this.studyField('baselineDrafts');
  readonly persistedRequestId = this.studyField('requestId');
  readonly persistedHistory = this.studyField('persistedHistory');
  readonly loadingPersistedHistory = signal(false);
  readonly saving = signal(false);
  readonly saveError = signal<string | null>(null);
  readonly saveSuccess = signal(false);
  readonly studyReport = this.studyField('studyReport');
  private watchedJobId: string | null = null;
  private pollInterval = 1;
  readonly dirty = computed(() => {
    const session = this.persistedHistory();
    return Boolean(
      session?.status === 'succeeded' &&
      session.reports.some((report) => this.isPersistedReportDirty(report, session)),
    );
  });
  readonly currentDirty = computed(() => {
    const session = this.persistedHistory();
    return Boolean(session?.status === 'succeeded' && session.reports.some((report) => this.isCurrentReportDirty(report, session)));
  });
  readonly selectedModel = computed(() => this.state().modelAvailability.find((model) => model.model_ref === this.state().selectedModelRef) ?? null);
  readonly filteredModels = computed(() => { const query = this.modelFilter().trim().toLowerCase(); return this.state().modelAvailability.filter((model) => !query || `${model.display_name} ${model.description} ${model.provider} ${model.origin} ${model.anatomy_coverage}`.toLowerCase().includes(query)); });
  readonly publicModels = computed(() => this.filteredModels().filter((model) => model.origin === 'public'));
  readonly customModels = computed(() => this.filteredModels().filter((model) => model.origin === 'custom'));
  readonly currentImage = computed(() => this.state().images[this.state().currentIndex] ?? null);
  readonly currentImageUrl = computed(() => this.currentUrl());
  readonly maxImages = computed(() => this.selectedModel()?.max_current_images ?? 1);
  readonly outputSections = computed(() => this.selectedModel()?.output_sections ?? []);
  readonly canGenerate = computed(() => { const model = this.selectedModel(); const status = model?.status; return Boolean(status && !(model?.origin === 'public' && model.validation_status === 'degraded') && !(model?.origin === 'public' && (model.gated || model.access_policy === 'gated') && status !== 'ready') && ['ready', 'not_installed', 'unvalidated', 'runtime_unavailable'].includes(status)); });
  readonly modelNotice = computed(() => {
    const model = this.selectedModel();
    if (!model || this.modelNoticeDismissedFor() === model.model_ref) return null;
    const notices: string[] = [];
    if (model.origin === 'public' && model.validation_status === 'degraded') {
      notices.push('This model failed qualification and cannot generate drafts.');
    }
    if (model.access_policy === 'gated' || model.gated) {
      notices.push('Access is required before this model can be downloaded.');
    } else if (model.installation_state === 'not_installed' || model.status === 'not_installed' || model.status_message?.toLowerCase().includes('download')) {
      notices.push('Downloads locally on first Generate.');
    }
    if (model.validation_status !== 'passed' || model.validation_receipt_status !== 'passed') {
      if (model.validation_message || model.validation_receipt_message || model.validation_status !== 'passed') {
        notices.push('Local files are checked before readiness is promoted by real image-to-report validation.');
      }
    }
    if (!notices.length) {
      const fallback = [model.status_message, model.validation_message, model.validation_receipt_message]
        .map((value) => value?.trim())
        .filter((value): value is string => Boolean(value));
      return [...new Set(fallback)].join(' ') || null;
    }
    return notices.join(' ');
  });
  readonly hasDraft = computed(() => Object.values(this.drafts()[this.state().currentIndex] ?? {}).some((value) => Boolean(value?.trim())) || Boolean(this.state().generatedReport.trim()));
  readonly studySupported = computed(() => Boolean(this.selectedModel()?.capabilities?.multiple_current_views));
  // A scope kept from a previous model may no longer be supported; fall back to independent reports
  readonly effectiveScope = computed(() => { const scope = this.state().runScope; return scope === 'study' && !this.studySupported() ? 'all' : scope; });
  readonly scopeDescription = computed(() => {
    const { images, currentIndex } = this.state();
    const scope = this.effectiveScope();
    if (scope === 'selected') return `One report for ${images[currentIndex]?.name ?? 'the selected image'}; the other images are not processed.`;
    if (scope === 'all') return `${images.length} independent reports, one per image, each saved separately. Browse them in step 3.${this.studySupported() ? '' : ' This model reads one image at a time, so a single combined study report is unavailable.'}`;
    return `One consolidated report from all ${images.length} images read together as views of the same study.`;
  });
  readonly profileNote = computed(() => this.selectedModel()?.provider === 'xreport' ? 'Custom checkpoints decode Deterministic and Concise identically (greedy search); Detailed uses beam search.' : null);
  readonly profileLabel = computed(() => {
    const selected = this.generationProfile;
    const used = this.hasDraft() ? this.persistedHistory()?.generation_profile : undefined;
    return used && used !== selected ? `${used} (selected: ${selected}; regenerate to apply)` : used ?? selected;
  });
  readonly reportEntries = computed(() => {
    if (this.studyReport() || (this.state().isGenerating && this.effectiveScope() === 'study')) return [];
    const drafts = this.drafts(); const { images } = this.state(); const session = this.persistedHistory();
    return this.runIndices().map((index) => ({
      index,
      name: images[index]?.name ?? `Image ${index + 1}`,
      hasReport: Object.values(drafts[index] ?? {}).some((value) => Boolean(value?.trim())),
      dirty: Boolean(session?.reports.some((report) => this.displayIndexForHistoryReport(report, session) === index && this.isPersistedReportDirty(report, session))),
    }));
  });
  readonly reportPosition = computed(() => this.reportEntries().findIndex((entry) => entry.index === this.state().currentIndex));
  get modelFilterValue() { return this.modelFilter(); }
  set modelFilterValue(value: string) { this.modelFilter.set(value); }
  get clinicalContext() { return this.state().clinicalContext; }
  set clinicalContext(value: string) { this.appState.updateInference((state) => ({ ...state, clinicalContext: value })); }
  get generationProfile() { return this.state().generationProfile; }
  set generationProfile(value: GenerationProfile) { this.appState.updateInference((state) => ({ ...state, generationProfile: value })); }

  readonly importOpen = signal(false);
  onCheckpointImported() { void this.loadModels(); }

  readonly runIndices = computed(() => { const { runIndices, images } = this.state(); return runIndices.length ? runIndices : images.map((_, index) => index); });
  constructor() {
    this.loadModels();
    this.setPreview(this.currentImage());
    const { isGenerating, requestId, persistedHistory } = this.state();
    if (!isGenerating && requestId && !persistedHistory) void this.loadPersistedHistory(requestId);
    // The job id lives in shared state, so a page recreated mid-run resumes tracking it (and picks up completed jobs on the first poll)
    effect(() => { const jobId = this.state().activeJobId; if (jobId && jobId !== this.watchedJobId) untracked(() => this.watchJob(jobId)); });
    this.destroyRef.onDestroy(() => { const url = this.currentUrl(); if (url) URL.revokeObjectURL(url); });
  }
  ngAfterViewInit() {
    const targets: Record<string, string> = {
      '.upload-zone': 'inference-study-upload',
      '.study-settings': 'inference-generation-settings',
      '.report-step': 'inference-draft-review',
    };
    const root = this.element.nativeElement as HTMLElement;
    for (const [selector, target] of Object.entries(targets)) (root.querySelector(selector) as HTMLElement | null)?.setAttribute('data-guidance-target', target);
  }
  startInferenceTour() { this.guidance.requestTour(INFERENCE_TOUR_ID); }
  private studyField<K extends keyof InferenceStudy>(key: K) {
    return Object.assign(computed(() => this.state()[key]), {
      set: (value: InferenceStudy[K]) => this.appState.updateInference((state) => ({ ...state, [key]: value })),
      update: (updater: (value: InferenceStudy[K]) => InferenceStudy[K]) => this.appState.updateInference((state) => ({ ...state, [key]: updater(state[key]) })),
    });
  }
  private async loadModels() { this.appState.updateInference((state) => ({ ...state, isLoadingModels: true })); this.catalogError.set(null); markStartupPhase('catalog_request_started'); const response = await this.api.getModels(); if (response.result) { const models = response.result.models; const current = this.state().selectedModelRef; const next = models.some((model) => model.model_ref === current) ? current : models[0]?.model_ref ?? ''; this.appState.updateInference((state) => ({ ...state, modelAvailability: models, isLoadingModels: false })); if (next !== current) { this.appState.switchInferenceModel(next); this.setPreview(this.currentImage()); } this.appState.retainInferenceStudies(new Set(models.map((model) => model.model_ref))); markStartupPhase('catalog_loaded'); } else { this.catalogError.set(response.error ?? 'Unable to load the local model catalog.'); this.appState.updateInference((state) => ({ ...state, isLoadingModels: false })); markStartupPhase('catalog_failed'); } }
  // Switching keeps the current study for this model and restores the target model's own study; nothing is discarded
  selectModel(model: ModelAvailability) { if (this.maintenanceBusy() || this.state().isGenerating || this.saving() || this.loadingPersistedHistory() || model.model_ref === this.state().selectedModelRef) return; this.appState.switchInferenceModel(model.model_ref); this.setPreview(this.currentImage()); this.modelNoticeDismissedFor.set(null); this.generationError.set(null); this.generationCancelled.set(false); this.maintenanceMessage.set(null); this.saveError.set(null); this.saveSuccess.set(false); const { requestId, persistedHistory } = this.state(); if (requestId && !persistedHistory) void this.loadPersistedHistory(requestId); }
  dismissModelNotice() { const model = this.selectedModel(); if (model) this.modelNoticeDismissedFor.set(model.model_ref); }
  onDrop(event: DragEvent) { event.preventDefault(); this.addFiles(event.dataTransfer?.files ?? null); }
  addFiles(files: FileList | null) { if (!files?.length || this.state().isGenerating || this.saving() || this.loadingPersistedHistory()) return; const accepted = Array.from(files).filter((file) => file.type.startsWith('image/')); if (!accepted.length || !this.confirmDiscardIfDirty()) return; const images = this.maxImages() === 1 ? accepted.slice(0, 1) : [...this.state().images, ...accepted].slice(0, Math.min(this.maxImages(), 16)); this.clearPersistedDraftState(); this.appState.updateInference((state) => ({ ...state, images, currentIndex: 0, runIndices: [], reports: {}, generatedReport: '', isCopied: false })); this.setPreview(images[0] ?? null); this.drafts.set({}); this.studyReport.set(false); }
  private setPreview(file: File | null) { const old = this.currentUrl(); if (old) URL.revokeObjectURL(old); this.currentUrl.set(file ? URL.createObjectURL(file) : null); }
  setIndex(index: number) { const next = Math.max(0, Math.min(this.state().images.length - 1, index)); this.appState.updateInference((state) => ({ ...state, currentIndex: next, generatedReport: state.reports[next] ?? '' })); this.setPreview(this.state().images[next] ?? null); }
  stepReport(delta: number) { const target = this.reportEntries()[this.reportPosition() + delta]; if (target) this.setIndex(target.index); }
  resetStudy() { if (this.state().isGenerating || this.saving() || this.loadingPersistedHistory() || !this.confirmDiscardIfDirty()) return; this.clearPersistedDraftState(); this.appState.clearInferenceStudy(); this.setPreview(null); this.generationError.set(null); }
  async generate() {
    const model = this.selectedModel(); const all = this.state().images;
    if (!model || !all.length || this.state().isGenerating || this.saving() || this.loadingPersistedHistory() || !this.confirmDiscardIfDirty()) return;
    const scope = this.effectiveScope();
    const indices = scope === 'selected' ? [this.state().currentIndex] : all.map((_, index) => index);
    this.appState.updateInference((state) => ({ ...state, isGenerating: true, isCopied: false, runIndices: indices }));
    this.progress.set(0); this.progressMessage.set('Preparing and generating…'); this.generationError.set(null); this.generationCancelled.set(false);
    const started = await this.api.generateReports(indices.map((index) => all[index]), model.model_ref, this.generationProfile, this.clinicalContext, scope === 'all' ? 'per_image' : 'study');
    if (!started.result) { this.appState.updateInference((state) => ({ ...state, isGenerating: false, runIndices: [] })); this.generationError.set(started.error ?? 'Unable to start generation.'); this.notifications.notify({ type: 'error', message: `Unable to start generation: ${this.generationError()}` }); return; }
    this.clearPersistedDraftState(); this.drafts.set({}); this.studyReport.set(false);
    this.pollInterval = started.result.poll_interval ?? 1;
    this.appState.updateInference((state) => ({ ...state, reports: {}, generatedReport: '', isCopied: false, activeJobId: started.result!.job_id }));
  }
  setRunScope(scope: 'selected' | 'all' | 'study') { this.appState.updateInference((state) => ({ ...state, runScope: scope })); }
  private watchJob(jobId: string) {
    this.watchedJobId = jobId;
    this.polling.poll((id) => this.jobsApi.get(id), jobId, this.pollInterval).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => {
      this.progress.set(status.progress ?? 0);
      const lifecycle = asRecord(status.result?.['lifecycle']);
      this.progressMessage.set(readString(lifecycle?.['message']) ?? status.status);
      if (status.result) this.consumeReports(status.result);
      if (!['completed', 'failed', 'cancelled'].includes(status.status)) return;
      this.appState.updateInference((state) => ({ ...state, isGenerating: false, activeJobId: null }));
      this.watchedJobId = null;
      if (status.status === 'completed') {
        const requestId = readString(status.result?.['request_id'])?.trim();
        if (requestId) void this.loadPersistedHistory(requestId); else this.saveError.set('Generation completed, but its saved report could not be linked. Check Reports and try again.');
        const failures = Object.entries(asRecord(asRecord(status.result?.['provenance'])?.['image_errors']) ?? {});
        if (failures.length) this.generationError.set(`${failures.length} image(s) failed: ${failures.map(([name, message]) => `${name} (${String(message)})`).join('; ')}`);
        this.notifications.notify(failures.length
          ? { type: 'warning', message: `Report generation finished with ${failures.length} failed image(s).` }
          : { type: 'success', message: 'Report generation completed.' });
      } else {
        const cancelled = status.status === 'cancelled';
        this.generationCancelled.set(cancelled);
        this.generationError.set(status.error ?? (cancelled ? 'Generation cancelled.' : 'Generation did not complete.'));
        this.notifications.notify({ type: cancelled ? 'info' : 'error', message: cancelled ? 'Report generation cancelled.' : `Report generation failed: ${this.generationError()}` });
      }
      void this.loadModels();
    });
  }
  private consumeReports(result: Record<string, unknown>) {
    const images = this.state().images; const run = this.runIndices();
    const ordered = readStringArray(result['reports_ordered']); const reports = asRecord(result['reports']);
    const names = readStringArray(result['report_filenames']) ?? run.map((index) => images[index]?.name ?? '');
    const studyScoped = readString(asRecord(result['provenance'])?.['report_scope']) === 'study';
    const entries: [string, string][] = ordered?.length ? ordered.map((report, position) => [names[position] ?? '', report]) : reports ? Object.entries(reports).flatMap(([name, report]) => readString(report) === undefined ? [] : [[name, readString(report)!] as [string, string]]) : [];
    if (!entries.length) return;
    const byIndex: Record<number, string> = {}; const used = new Set<number>();
    entries.forEach(([name, report], position) => { const target = run.find((index) => !used.has(index) && images[index]?.name === name) ?? run[position]; if (target === undefined) return; used.add(target); byIndex[target] = report; });
    this.studyReport.set(studyScoped);
    if (studyScoped && entries.length === 1 && run.length > 1) { const report = entries[0][1]; run.forEach((index) => byIndex[index] = report); }
    const nextDrafts = { ...this.drafts() };
    Object.entries(byIndex).forEach(([index, report]) => { nextDrafts[Number(index)] = this.parseDraft(report); });
    this.drafts.set(nextDrafts);
    this.appState.updateInference((state) => ({ ...state, reports: byIndex, generatedReport: byIndex[state.currentIndex] ?? '' }));
  }
  private parseDraft(report: string): DraftSections { return parseReportDraft(report, this.outputSections()); }
  draft(section: OutputSection) { return this.drafts()[this.state().currentIndex]?.[section] ?? ''; }
  updateDraft(section: OutputSection, value: string) { const currentIndex = this.state().currentIndex; this.drafts.update((drafts) => { const next = { ...drafts, [currentIndex]: { ...(drafts[currentIndex] ?? {}), [section]: value } }; if (this.studyReport()) this.runIndices().forEach((index) => { next[index] = { ...(next[index] ?? {}), [section]: value }; }); return next; }); const report = this.outputSections().map((item) => `${SECTION_LABELS[item]}\n${this.draft(item).trim()}`).join('\n\n').trim(); this.appState.updateInference((state) => ({ ...state, generatedReport: report, reports: this.studyReport() ? this.runIndices().reduce<Record<number, string>>((all, index) => ({ ...all, [index]: report }), {}) : { ...state.reports, [currentIndex]: report } })); this.saveError.set(null); this.saveSuccess.set(false); }
  private historySections(report: InferenceHistoryReport, session: InferenceHistoryDetail): OutputSection[] { const declared = Object.keys(report.sections ?? {}).filter(isOutputSection); if (declared.length) return declared; const sessionSections = session.output_sections.filter(isOutputSection); return sessionSections.length ? sessionSections : this.outputSections().length ? this.outputSections() : ['raw_report']; }
  private displayIndexForHistoryReport(report: InferenceHistoryReport, session: InferenceHistoryDetail): number | null { const run = this.runIndices(); if (this.studyReport() && session.reports.length === 1 && run.length > 1) return run[0]; return run[report.image_index] ?? null; }
  private draftForHistoryReport(report: InferenceHistoryReport, session: InferenceHistoryDetail): DraftSections { const displayIndex = this.displayIndexForHistoryReport(report, session); return displayIndex === null ? this.baselineDrafts()[report.image_index] ?? {} : this.drafts()[displayIndex] ?? {}; }
  private isPersistedReportDirty(report: InferenceHistoryReport, session: InferenceHistoryDetail): boolean { const displayIndex = this.displayIndexForHistoryReport(report, session); if (displayIndex === null) return false; return JSON.stringify(this.drafts()[displayIndex] ?? {}) !== JSON.stringify(this.baselineDrafts()[report.image_index] ?? {}); }
  // A study report is shared by every submitted image, so it is always "current"; per-image reports only for their own image
  private isCurrentReportDirty(report: InferenceHistoryReport, session: InferenceHistoryDetail): boolean { return (this.studyReport() || this.displayIndexForHistoryReport(report, session) === this.state().currentIndex) && this.isPersistedReportDirty(report, session); }
  private clearPersistedDraftState(): void { this.appState.updateInference((state) => ({ ...state, requestId: null, persistedHistory: null, baselineDrafts: {} })); this.loadingPersistedHistory.set(false); this.saveError.set(null); this.saveSuccess.set(false); }
  private async loadPersistedHistory(requestId: string): Promise<void> { this.persistedRequestId.set(requestId); this.loadingPersistedHistory.set(true); this.saveError.set(null); this.saveSuccess.set(false); try { const response = await this.api.getHistory(requestId); if (this.persistedRequestId() !== requestId) return; const session = response.result; if (!session || session.request_id !== requestId || session.status !== 'succeeded' || !session.reports.length) { this.saveError.set(response.error ?? 'The saved report could not be confirmed. Retry loading its history before saving edits.'); return; } this.applyPersistedHistory(session); } catch (error) { this.saveError.set(error instanceof Error ? error.message : 'Unable to load the saved report. Retry loading its history.'); } finally { if (this.persistedRequestId() === requestId) this.loadingPersistedHistory.set(false); } }
  // `saved` limits the refresh to the reports just written, so unsaved edits on the other reports survive
  private applyPersistedHistory(session: InferenceHistoryDetail, saved?: ReadonlySet<number>): void { const baseline: Record<number, DraftSections> = {}; const nextDrafts = { ...this.drafts() }; const nextReports = { ...this.state().reports }; const reports = [...session.reports].sort((left, right) => left.image_index - right.image_index); const run = this.runIndices(); const studyLevel = this.studyReport() && reports.length === 1 && run.length > 1; for (const report of reports) { if (saved && !saved.has(report.image_index) && this.isPersistedReportDirty(report, session)) { baseline[report.image_index] = this.baselineDrafts()[report.image_index] ?? {}; continue; } const draft = parseReportDraft(report.effective_report, this.historySections(report, session)); baseline[report.image_index] = draft; const formatted = formatReportDraft(this.historySections(report, session), draft, report.effective_report); if (studyLevel) { run.forEach((index) => { nextDrafts[index] = { ...draft }; nextReports[index] = formatted; }); } else if (run[report.image_index] !== undefined) { nextDrafts[run[report.image_index]] = draft; nextReports[run[report.image_index]] = formatted; } } this.persistedHistory.set({ ...session, reports }); this.persistedRequestId.set(session.request_id); this.baselineDrafts.set(baseline); this.drafts.set(nextDrafts); this.appState.updateInference((state) => ({ ...state, reports: nextReports, generatedReport: nextReports[state.currentIndex] ?? state.generatedReport })); }
  async retryLoadPersistedHistory(): Promise<void> { const requestId = this.persistedRequestId(); if (!requestId || this.loadingPersistedHistory() || this.saving()) return; await this.loadPersistedHistory(requestId); }
  async saveEdits(): Promise<void> { const session = this.persistedHistory(); const requestId = this.persistedRequestId(); if (!session || !requestId || session.status !== 'succeeded' || !this.currentDirty() || this.saving() || this.loadingPersistedHistory() || this.state().isGenerating) return; const updates = [...session.reports].sort((left, right) => left.image_index - right.image_index).filter((report) => this.isCurrentReportDirty(report, session)).map((report) => ({ image_index: report.image_index, edited_report: formatReportDraft(this.historySections(report, session), this.draftForHistoryReport(report, session), report.effective_report) })); if (!updates.length) return; this.saving.set(true); this.saveError.set(null); this.saveSuccess.set(false); try { const response = await this.api.updateHistory(requestId, { reports: updates }); if (response.result?.request_id === requestId && response.result.status === 'succeeded') { this.applyPersistedHistory(response.result, new Set(updates.map((update) => update.image_index))); this.saveSuccess.set(true); } else { this.saveError.set(response.error ?? 'Unable to save report edits. Your changes are still available; try again.'); } } catch (error) { this.saveError.set(error instanceof Error ? error.message : 'Unable to save report edits. Your changes are still available; try again.'); } finally { this.saving.set(false); } }
  // Unsaved edits stay in the model's study while navigating, so leaving only waits for an in-flight report operation
  confirmDiscardUnsavedChanges(): boolean { if (this.saving() || this.loadingPersistedHistory()) { this.saveError.set('Wait for the report operation to finish before leaving this page.'); return false; } return true; }
  private confirmDiscardIfDirty(): boolean { return !this.dirty() || window.confirm('You have unsaved report edits. Leave and discard them? The generated report remains available in Reports.'); }
  @HostListener('window:beforeunload', ['$event']) protectUnsavedDraft(event: BeforeUnloadEvent): void { if (!this.dirty() && !this.saving()) return; event.preventDefault(); event.returnValue = ''; }
  async copyDraft() { if (!this.hasDraft()) return; try { await navigator.clipboard.writeText(this.formattedDraft()); this.appState.updateInference((state) => ({ ...state, isCopied: true })); setTimeout(() => this.appState.updateInference((state) => ({ ...state, isCopied: false })), 1600); } catch { this.generationError.set('Clipboard access is unavailable in this browser.'); } }
  exportDraft() { if (!this.hasDraft()) return; const blob = new Blob([this.formattedDraft()], { type: 'text/plain;charset=utf-8' }); const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `xreport-${this.currentImage()?.name?.replace(/\.[^.]+$/, '') || 'draft'}.txt`; anchor.click(); URL.revokeObjectURL(url); }
  private formattedDraft() { return formatReportDraft(this.outputSections(), this.drafts()[this.state().currentIndex] ?? {}, this.state().generatedReport); }
  async cancelGeneration() { const jobId = this.state().activeJobId; if (!jobId) return; const response = await this.jobsApi.cancel(jobId); if (!response.result?.success) { this.generationError.set(response.error ?? response.result?.message ?? 'Unable to cancel generation.'); return; } this.progressMessage.set('Cancellation requested…'); }
  async checkUpdate(model: ModelAvailability) {
    if (this.maintenanceBusy()) return;
    this.maintenanceBusy.set(true);
    this.updateChecking.set(true);
    this.maintenanceMessage.set(null);
    try {
      const response = await this.api.checkModelUpdate(model.model_ref);
      const result = response.result;
      const short = (revision?: string | null) => revision?.slice(0, 12) ?? 'unknown';
      this.maintenanceMessage.set(
        response.error ? `Update check failed: ${response.error}`
        : !result ? 'Update check returned no result.'
        : result.error ? `Update check failed: ${result.error}`
        : result.update_available ? `Update available: ${short(result.installed_revision)} → ${short(result.latest_revision)}`
        : result.installed_revision ? `Up to date (${short(result.installed_revision)}).`
        : `No active local revision. Latest published revision: ${short(result.latest_revision)}.`,
      );
    } finally {
      this.updateChecking.set(false);
      this.maintenanceBusy.set(false);
    }
  }
  async maintain(model: ModelAvailability, action: 'download' | 'repair' | 'reinstall' | 'download_update' | 'delete_local') { if (this.maintenanceBusy()) return; if (action === 'delete_local' && !confirm(`Delete the local files for ${model.display_name}? The catalogue entry will remain available.`)) return; this.maintenanceBusy.set(true); this.maintenanceJobId.set(null); this.maintenanceProgress.set(0); this.maintenanceProgressMessage.set(`${action.replace('_', ' ')} starting…`); this.maintenanceProgressDetail.set(''); this.maintenanceCancellationRequested.set(false); const started = await this.api.maintainModel(model.model_ref, action); if (!started.result) { this.maintenanceMessage.set(started.error ?? 'Unable to start maintenance.'); this.maintenanceBusy.set(false); return; } this.maintenanceJobId.set(started.result.job_id); this.maintenanceMessage.set(`${action.replace('_', ' ')} started…`); this.polling.poll((jobId) => this.jobsApi.get(jobId), started.result.job_id, (started.result.poll_interval ?? 1)).pipe(takeUntilDestroyed(this.destroyRef)).subscribe((status) => { this.maintenanceProgress.set(status.progress ?? 0); const lifecycle = asRecord(status.result?.['lifecycle']); this.maintenanceProgressMessage.set(readString(lifecycle?.['message']) ?? status.status); const currentFile = readString(lifecycle?.['current_file']); const filesCompleted = readNumber(lifecycle?.['files_completed']); const totalFiles = readNumber(lifecycle?.['total_files']); this.maintenanceProgressDetail.set(currentFile ? `${currentFile}${filesCompleted !== undefined && totalFiles !== undefined ? ` · ${filesCompleted}/${totalFiles} files` : ''}` : ''); if (['completed', 'failed', 'cancelled'].includes(status.status)) { this.maintenanceMessage.set(status.error ?? (status.status === 'completed' ? 'Model maintenance completed.' : status.status === 'cancelled' ? 'Download cancelled; partial files remain resumable.' : 'Model maintenance did not complete.')); this.maintenanceBusy.set(false); this.maintenanceJobId.set(null); this.maintenanceCancellationRequested.set(false); void this.loadModels(); } }); }
  async cancelMaintenance() { const jobId = this.maintenanceJobId(); if (!jobId || this.maintenanceCancellationRequested()) return; this.maintenanceCancellationRequested.set(true); this.maintenanceMessage.set('Cancellation requested…'); const response = await this.jobsApi.cancel(jobId); if (!response.result?.success) { this.maintenanceCancellationRequested.set(false); this.maintenanceMessage.set(response.error ?? response.result?.message ?? 'Unable to cancel the model download.'); } }
  hasAction(model: ModelAvailability, action: string) { return model.available_actions?.includes(action) ?? false; }
  modelMaintenanceAction(model: ModelAvailability): 'download' | 'repair' { return this.hasAction(model, 'download') ? 'download' : 'repair'; }
  modelMaintenanceLabel(model: ModelAvailability) { return this.hasAction(model, 'repair') ? 'Retry / resume download' : 'Download model'; }
  pretty(value: string | null | undefined) { const text = value?.replaceAll('_', ' '); return text ? text.charAt(0).toUpperCase() + text.slice(1) : 'Not specified'; }
  installationLabel(model: ModelAvailability) { const state = this.pretty(model.installation_state); const integrity = model.integrity_status; return integrity && integrity !== 'unknown' ? `${state} · ${this.pretty(integrity)}` : state; }
  modelRevision(model: ModelAvailability) { return model.active_revision || model.model_revision || 'Not configured'; }
  sectionLabel(section: OutputSection) { return SECTION_LABELS[section]; }
  outputLabels(sections: OutputSection[]) { return sections.map((section) => SECTION_LABELS[section]).join(', ') || 'Not declared'; }
  capabilities(model: ModelAvailability) { return [model.capabilities.clinical_context && 'Clinical context', model.capabilities.multiple_current_views && 'Multiple views', model.capabilities.findings && 'Findings', model.capabilities.impression && 'Impression', model.capabilities.grounding && 'Grounding'].filter((value): value is string => Boolean(value)); }
}
