import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Output, inject, signal, computed } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideArrowRight,
  lucideBrainCircuit,
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
import { JobsApiService } from '../services/jobs-api.service';

type JourneyStepStatus = 'completed' | 'current' | 'upcoming' | 'available';

interface JourneyStep {
  id: string;
  icon: string;
  title: string;
  summary: string;
  doThis: string;
  whatHappens: string;
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
  providers: [provideIcons({ lucideArrowRight, lucideBrainCircuit, lucideCheckCircle, lucideChevronDown, lucideDatabase, lucideLoaderCircle, lucidePlay, lucideRefreshCw, lucideRotateCcw, lucideSliders, lucideUpload })],
  template: `
    <section class="guidance-workflow-journey" aria-labelledby="dataset-training-journey-title">
      <header class="guidance-journey-header">
        <div>
          <span class="guidance-journey-eyebrow">Dataset to training</span>
          <h3 id="dataset-training-journey-title">Build, train, and continue</h3>
          <p>Follow the five stages in order. Live status shows what is already available in this workspace and what to do next.</p>
        </div>
        <button type="button" class="guidance-journey-refresh" (click)="refresh()" [disabled]="status().loading" [attr.aria-label]="status().loading ? 'Refreshing workflow status' : 'Refresh workflow status'">
          <ng-icon name="lucideRefreshCw" [class.spin]="status().loading" aria-hidden="true" />
          <span>{{ status().loading ? 'Refreshing…' : 'Refresh status' }}</span>
        </button>
      </header>

      <div class="guidance-journey-flow" aria-label="Workflow relationship">
        <span>Source data</span>
        <ng-icon name="lucideArrowRight" aria-hidden="true" />
        <span>Processed dataset</span>
        <ng-icon name="lucideArrowRight" aria-hidden="true" />
        <span>Training run</span>
        <ng-icon name="lucideArrowRight" aria-hidden="true" />
        <span>Checkpoint</span>
      </div>

      <div class="guidance-journey-status" role="status" aria-live="polite">
        <span class="guidance-journey-status-dot" [class.is-loading]="status().loading" [class.is-ready]="!status().loading && !status().unknown"></span>
        <strong>{{ summary() }}</strong>
        <span class="guidance-journey-count">{{ completedRequiredCount() }} of 4 required stages complete</span>
      </div>

      @if (status().error) {
        <p class="guidance-journey-error" role="status">{{ status().error }} You can still read the walkthrough and open the relevant page.</p>
      }

      <ol class="guidance-journey-steps">
        @for (step of visibleSteps(); track step.id; let index = $index) {
          <li class="guidance-journey-step" [class.is-expanded]="activeStep() === index" [class.is-completed]="step.status === 'completed'" [class.is-current]="step.status === 'current'" [class.is-available]="step.status === 'available'">
            <button type="button" class="guidance-journey-step-toggle" [attr.aria-expanded]="activeStep() === index" [attr.aria-controls]="'journey-step-content-' + step.id" (click)="selectStep(index)">
              <span class="guidance-journey-marker">
                @if (step.status === 'completed') {
                  <ng-icon name="lucideCheckCircle" aria-hidden="true" />
                } @else {
                  <span>{{ index + 1 }}</span>
                }
              </span>
              <span class="guidance-journey-step-copy">
                <span class="guidance-journey-step-label">Step {{ index + 1 }} of {{ visibleSteps().length }}</span>
                <strong>{{ step.title }}</strong>
                <span class="guidance-journey-step-status">{{ statusLabel(step.status, index) }}</span>
              </span>
              <ng-icon name="lucideChevronDown" class="guidance-journey-chevron" aria-hidden="true" />
            </button>

            @if (activeStep() === index) {
              <div class="guidance-journey-step-content" [id]="'journey-step-content-' + step.id">
                <p class="guidance-journey-summary">{{ step.summary }}</p>
                <dl class="guidance-journey-details">
                  <div>
                    <dt>Do this</dt>
                    <dd>{{ step.doThis }}</dd>
                  </div>
                  <div>
                    <dt>What happens</dt>
                    <dd>{{ step.whatHappens }}</dd>
                  </div>
                </dl>
                <div class="guidance-journey-actions">
                  <button type="button" class="guidance-button guidance-button-primary" (click)="requestRoute(step, index)" [disabled]="isRouteLocked(step, index)">
                    <ng-icon [name]="isRouteLocked(step, index) ? 'lucideLoaderCircle' : step.icon" [class.spin]="isRouteLocked(step, index) && status().loading" aria-hidden="true" />
                    {{ actionLabel(step, index) }}
                  </button>
                  @if (index === 0) {
                    <span class="guidance-journey-hint">The image folder and data file are loaded together before processing.</span>
                  } @else if (index === 4 && !status().checkpointCount) {
                    <span class="guidance-journey-hint">Start a run with checkpoint saving enabled to make Resume available.</span>
                  }
                </div>
              </div>
            }
          </li>
        }
      </ol>

      <footer class="guidance-journey-footer">
        <span>Want to see the controls in context?</span>
        <button type="button" class="guidance-button guidance-button-primary" (click)="walkthroughRequested.emit()">Walk me through it</button>
      </footer>
    </section>
  `,
  styleUrl: '../styles/Guidance.css',
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
    {
      id: 'source',
      icon: 'lucideUpload',
      title: 'Upload and load source data',
      summary: 'Bring the image folder and its reports or metadata into the application as one matched source.',
      doThis: 'Choose an image folder containing JPG/JPEG, PNG, BMP, TIFF, or GIF files. Desktop uses the native system picker; browser mode uses the server fallback. Upload the CSV or XLSX reports/metadata file, then click Load Dataset.',
      whatHappens: 'XREPORT matches report image identifiers to available images, reports matched and unmatched records, and stores the loaded source rows for processing.',
      route: '/dataset',
      routeLabel: 'Open Dataset',
    },
    {
      id: 'process',
      icon: 'lucideSliders',
      title: 'Process the uploaded source',
      summary: 'Turn the loaded source into clean, tokenized, train/validation-ready material.',
      doThis: 'Select exactly one source dataset, choose the sample size, validation split, tokenizer, and maximum report size, then click Build Dataset.',
      whatHappens: 'A background job sanitizes reports, tokenizes text, creates training and validation splits, and saves the processing result.',
      route: '/dataset',
      routeLabel: 'Open Processing',
    },
    {
      id: 'training-dataset',
      icon: 'lucideDatabase',
      title: 'Use the processed training dataset',
      summary: 'The processed output is the separate training dataset that the model can consume.',
      doThis: 'Open Training and select a processed dataset from the New Training Session list. Inspect its row count or metadata when needed.',
      whatHappens: 'The processed dataset contains the prepared samples and is kept conceptually separate from the raw source rows.',
      route: '/training',
      routeLabel: 'Open Training',
    },
    {
      id: 'new-training',
      icon: 'lucidePlay',
      title: 'Start a new training run',
      summary: 'Configure a fresh run from a processed dataset and monitor it from the training dashboard.',
      doThis: 'Select a processed dataset, click Configure Training, and review Model, Dataset, Training, Device, and Summary. Pay attention to epochs, batch size, device, and checkpoint saving.',
      whatHappens: 'Training starts as a background job. The dashboard reports progress, losses, accuracy, elapsed time, charts, and logs while checkpoints are saved when enabled.',
      route: '/training',
      routeLabel: 'Open New Training',
    },
    {
      id: 'resume',
      icon: 'lucideRotateCcw',
      title: 'Resume from an existing checkpoint',
      summary: 'Continue a saved run instead of rebuilding the model state from the beginning.',
      doThis: 'Under Resume Training, select a checkpoint, click Resume Training, and enter the number of additional epochs. Use Evaluate Model when you want to inspect it first.',
      whatHappens: 'The saved checkpoint supplies the existing training state, so the next run continues from that point and produces updated progress and checkpoints.',
      route: '/training',
      routeLabel: 'Open Checkpoints',
    },
  ];

  readonly visibleSteps = computed<VisibleJourneyStep[]>(() => this.steps.map((step, index) => ({ ...step, status: this.stepStatus(index) })));
  readonly completedRequiredCount = computed(() => this.visibleSteps().filter((step, index) => index < 4 && step.status === 'completed').length);
  readonly summary = computed(() => {
    const state = this.status();
    if (state.loading) return 'Checking your workflow status…';
    if (state.unknown) return 'Workflow status is partially unavailable';
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
      error: errors.length ? `Live workflow status could not be fully loaded (${errors[0]}).` : null,
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

  statusLabel(status: JourneyStepStatus, index: number): string {
    if (status === 'completed') return 'Completed';
    if (status === 'current') return 'Next recommended';
    if (status === 'available') return 'Available now';
    if (index === 4) return 'Waiting for a checkpoint';
    return 'After the previous step';
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
