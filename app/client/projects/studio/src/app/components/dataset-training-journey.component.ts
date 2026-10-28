import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Output, inject, signal, computed } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideArrowRight,
  lucideCheckCircle,
  lucideChevronDown,
  lucideDatabase,
  lucideLoaderCircle,
  lucidePlay,
  lucideRefreshCw,
  lucideRotateCcw,
  lucideSliders,
  lucideUpload,
} from '@ng-icons/lucide';
import { DatasetApiService } from '../services/dataset-api.service';
import { TrainingApiService } from '../services/training-api.service';
import { JobsApiService } from '@xreport/shared/app/services/jobs-api.service';

type JourneyStepStatus = 'completed' | 'current' | 'upcoming' | 'available';

interface JourneyStep {
  id: string;
  icon: string;
  title: string;
  detail: string;
  route: '/dataset' | '/training';
  routeLabel: string;
}

interface JourneyStatus {
  loading: boolean;
  unknown: boolean;
  error: string | null;
  sourceAvailable: boolean;
  sourceCount: number;
  processedAvailable: boolean;
  processedCount: number;
  checkpointCount: number;
  trainingActive: boolean;
  currentEpoch: number;
}

interface VisibleJourneyStep extends JourneyStep {
  status: JourneyStepStatus;
}

const initialStatus: JourneyStatus = {
  loading: true,
  unknown: false,
  error: null,
  sourceAvailable: false,
  sourceCount: 0,
  processedAvailable: false,
  processedCount: 0,
  checkpointCount: 0,
  trainingActive: false,
  currentEpoch: 0,
};

@Component({
  standalone: true,
  selector: 'app-dataset-training-journey',
  imports: [CommonModule, NgIcon],
  providers: [provideIcons({ lucideArrowRight, lucideCheckCircle, lucideChevronDown, lucideDatabase, lucideLoaderCircle, lucidePlay, lucideRefreshCw, lucideRotateCcw, lucideSliders, lucideUpload })],
  template: `
    <section class="guidance-workflow-journey" aria-labelledby="dataset-training-journey-title">
      <header class="guidance-journey-header">
        <h3 id="dataset-training-journey-title">Dataset to training</h3>
        <button type="button" class="guidance-journey-refresh" (click)="refresh()" [disabled]="status().loading" [attr.aria-label]="status().loading ? 'Refreshing workflow status' : 'Refresh workflow status'">
          <ng-icon name="lucideRefreshCw" [class.spin]="status().loading" aria-hidden="true" />
          <span>{{ status().loading ? 'Refreshing…' : 'Refresh' }}</span>
        </button>
      </header>

      <div class="guidance-journey-status" role="status" aria-live="polite">
        @if (status().unknown) {
          <span class="guidance-journey-status-dot"></span>
          <strong>Live status unavailable</strong>
          <span class="guidance-journey-count" [attr.title]="status().error">Steps are shown without progress.</span>
        } @else {
          <span class="guidance-journey-status-dot" [class.is-loading]="status().loading" [class.is-ready]="!status().loading"></span>
          <strong>{{ summary() }}</strong>
          <span class="guidance-journey-count">{{ completedRequiredCount() }}/4 done</span>
        }
      </div>

      <ol class="guidance-journey-steps">
        @for (step of visibleSteps(); track step.id; let index = $index) {
          <li class="guidance-journey-step" [class.is-expanded]="activeStep() === index" [class.is-completed]="step.status === 'completed'" [class.is-current]="step.status === 'current'" [class.is-available]="step.status === 'available'">
            <button type="button" class="guidance-journey-step-toggle" [attr.aria-expanded]="activeStep() === index" [attr.aria-controls]="'journey-step-content-' + step.id" [attr.aria-current]="step.status === 'current' ? 'step' : null" (click)="selectStep(index)">
              <span class="guidance-journey-marker">
                @if (step.status === 'completed') {
                  <ng-icon name="lucideCheckCircle" aria-hidden="true" />
                } @else {
                  <ng-icon [name]="step.icon" aria-hidden="true" />
                }
              </span>
              <strong class="guidance-journey-step-title">{{ step.title }}</strong>
              <span class="guidance-journey-chip">{{ statusLabel(step.status) }}</span>
              <ng-icon name="lucideChevronDown" class="guidance-journey-chevron" aria-hidden="true" />
            </button>

            @if (activeStep() === index) {
              <div class="guidance-journey-step-content" [id]="'journey-step-content-' + step.id">
                <p class="guidance-journey-summary">{{ step.detail }}</p>
                <div class="guidance-journey-actions">
                  <button type="button" class="guidance-button guidance-button-primary" (click)="requestRoute(step, index)" [disabled]="isRouteLocked(step, index)">
                    <ng-icon [name]="isRouteLocked(step, index) ? 'lucideLoaderCircle' : 'lucideArrowRight'" [class.spin]="isRouteLocked(step, index) && status().loading" aria-hidden="true" />
                    {{ actionLabel(step, index) }}
                  </button>
                  @if (index === 4 && !status().checkpointCount) {
                    <span class="guidance-journey-hint">Enable checkpoint saving when starting a run.</span>
                  }
                </div>
              </div>
            }
          </li>
        }
      </ol>

      <footer class="guidance-journey-footer">
        <span>Want to see the controls in context?</span>
        <button type="button" class="guidance-button" (click)="walkthroughRequested.emit()">Walk me through it</button>
      </footer>
    </section>
  `,
  styleUrl: '../../../../shared/src/app/styles/Guidance.css',
})
export class DatasetTrainingJourneyComponent {
  private readonly datasetApi = inject(DatasetApiService);
  private readonly trainingApi = inject(TrainingApiService);
  private readonly jobsApi = inject(JobsApiService);
  private refreshSequence = 0;

  @Output() readonly routeRequested = new EventEmitter<string>();
  @Output() readonly walkthroughRequested = new EventEmitter<void>();

  readonly status = signal<JourneyStatus>({ ...initialStatus });
  readonly activeStep = signal(0);
  readonly steps: readonly JourneyStep[] = [
    { id: 'source', icon: 'lucideUpload', title: 'Load source data', detail: 'Pick the image folder and its CSV/XLSX reports file, then click Load Dataset.', route: '/dataset', routeLabel: 'Open Dataset' },
    { id: 'process', icon: 'lucideSliders', title: 'Process the source', detail: 'Select one source dataset, set sample size, split, tokenizer and report length, then click Build Dataset.', route: '/dataset', routeLabel: 'Open Processing' },
    { id: 'training-dataset', icon: 'lucideDatabase', title: 'Pick a processed dataset', detail: 'Processed datasets are listed under New Training Session on the Training page.', route: '/training', routeLabel: 'Open Training' },
    { id: 'new-training', icon: 'lucidePlay', title: 'Start a training run', detail: 'Click Configure Training and review model, epochs, batch size, device and checkpoint saving.', route: '/training', routeLabel: 'Open New Training' },
    { id: 'resume', icon: 'lucideRotateCcw', title: 'Resume from a checkpoint', detail: 'Select a checkpoint under Resume Training, or inspect it first with Evaluate Model.', route: '/training', routeLabel: 'Open Checkpoints' },
  ];

  readonly visibleSteps = computed<VisibleJourneyStep[]>(() => this.steps.map((step, index) => ({ ...step, status: this.stepStatus(index) })));
  readonly completedRequiredCount = computed(() => this.visibleSteps().filter((step, index) => index < 4 && step.status === 'completed').length);
  readonly summary = computed(() => {
    const state = this.status();
    if (state.loading) return 'Checking your workflow status…';
        if (!state.sourceAvailable) return 'Next: upload and load your source data';
    if (!state.processedAvailable) return 'Next: process the loaded source';
    if (state.trainingActive) return `Training is in progress at epoch ${state.currentEpoch || 0}`;
    if (!state.currentEpoch && !state.checkpointCount) return 'Next: start a new training run';
    if (state.checkpointCount) return 'Your checkpoint branch is ready to resume';
    return 'Your processed dataset is ready for training';
  });

  constructor() {
    void this.refresh();
  }

  async refresh(): Promise<void> {
    const sequence = ++this.refreshSequence;
    this.status.set({ ...initialStatus });

    const [datasetStatus, sourceDatasets, processedDatasets, checkpoints, trainingJobs] = await Promise.all([
      this.datasetApi.getStatus(),
      this.datasetApi.getNames(),
      this.datasetApi.getProcessedNames(),
      this.trainingApi.getCheckpoints(),
      this.jobsApi.list('training'),
    ]);

    if (sequence !== this.refreshSequence) return;

    const errors = [datasetStatus.error, sourceDatasets.error, processedDatasets.error, checkpoints.error, trainingJobs.error].filter((error): error is string => Boolean(error));
    const sourceCount = sourceDatasets.result?.count ?? 0;
    const processedCount = processedDatasets.result?.count ?? 0;
    const checkpointCount = checkpoints.result?.checkpoints.length ?? 0;
    const training = trainingJobs.result?.jobs.find((job) => job.status === 'pending' || job.status === 'running');
    const trainingResult = training?.result;

    this.status.set({
      loading: false,
      unknown: errors.length > 0,
      error: errors[0] ?? null,
      sourceAvailable: Boolean(datasetStatus.result?.has_data || sourceCount > 0),
      sourceCount: sourceCount || (datasetStatus.result?.has_data ? 1 : 0),
      processedAvailable: processedCount > 0,
      processedCount,
      checkpointCount,
      trainingActive: Boolean(training),
      currentEpoch: typeof trainingResult?.['current_epoch'] === 'number' ? trainingResult['current_epoch'] : 0,
    });

    this.activeStep.set(this.recommendedStepIndex());
  }

  selectStep(index: number): void {
    this.activeStep.set(index);
  }

  requestRoute(step: JourneyStep, index: number): void {
    if (this.isRouteLocked(step, index)) return;
    this.routeRequested.emit(step.route);
  }

  statusLabel(status: JourneyStepStatus): string {
    if (status === 'completed') return 'Done';
    if (status === 'current') return 'Next';
    if (status === 'available') return 'Available';
    return 'Locked';
  }

  actionLabel(step: JourneyStep, index: number): string {
    if (this.isRouteLocked(step, index)) {
      return index === 4 ? 'Waiting for checkpoint' : 'Complete previous step';
    }
    return step.routeLabel;
  }

  isRouteLocked(_step: JourneyStep, index: number): boolean {
    const step = this.visibleSteps()[index];
    return step.status === 'upcoming';
  }

  private recommendedStepIndex(): number {
    const current = this.visibleSteps().findIndex((step) => step.status === 'current');
    if (current >= 0) return current;
    const available = this.visibleSteps().findIndex((step) => step.status === 'available');
    if (available >= 0) return available;
    return Math.max(0, this.visibleSteps().length - 2);
  }

  private stepStatus(index: number): JourneyStepStatus {
    const state = this.status();
    if (state.loading || state.unknown) return index === 0 ? 'current' : 'upcoming';

    if (index === 0) return state.sourceAvailable ? 'completed' : 'current';
    if (index === 1) return state.processedAvailable ? 'completed' : state.sourceAvailable ? 'current' : 'upcoming';
    if (index === 2) return state.processedAvailable ? 'completed' : 'upcoming';
    if (index === 3) return state.trainingActive || state.currentEpoch > 0 || state.checkpointCount > 0 ? 'completed' : state.processedAvailable ? 'current' : 'upcoming';
    return state.checkpointCount > 0 ? 'available' : 'upcoming';
  }
}
