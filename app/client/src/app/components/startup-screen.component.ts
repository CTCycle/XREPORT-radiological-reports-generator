import { Component, EventEmitter, Input, OnChanges, OnDestroy, OnInit, Output, SimpleChanges, signal } from '@angular/core';
import type { StartupPhase } from '../services/startup-readiness.service';

const READY_TRANSITION_MS = 320;

// The backend does not expose per-step percentages; it only reports `ok`
// once fully ready. The frontend drives the bar from the real elapsed
// preparation time (the same window the readiness service uses for its
// slow/unavailable thresholds), advancing monotonically and reaching 100%
// only when the health poll actually succeeds.
const PROGRESS_TICK_MS = 180;
const PROGRESS_CAP = 90;
const PROGRESS_TAU_MS = 22_000;

// These mirror the actual startup pipeline logged by the backend:
// database validated, runtime resources prepared, services started, health reachable.
const STARTUP_STEPS = [
  { label: 'Validating local data store', min: 0 },
  { label: 'Preparing runtime resources', min: 25 },
  { label: 'Starting local services', min: 50 },
  { label: 'Connecting to XREPORT service', min: 80 },
];

function progressForElapsed(elapsedMs: number): number {
  const progress = PROGRESS_CAP * (1 - Math.exp(-elapsedMs / PROGRESS_TAU_MS));
  return Math.round(Math.max(0, Math.min(PROGRESS_CAP, progress)));
}

function stepForProgress(progress: number): string {
  let label = STARTUP_STEPS[0].label;
  for (const step of STARTUP_STEPS) {
    if (progress >= step.min) label = step.label;
  }
  return label;
}

@Component({
  standalone: true,
  selector: 'app-startup-screen',
  template: `
    <main
      class="startup-screen"
      [class.startup-ready]="phase === 'ready'"
      [class.startup-unavailable]="phase === 'unavailable'"
      aria-labelledby="startup-title"
    >
      <div class="startup-shell">
        <header class="startup-brand">
          <img class="startup-logo" src="favicon.png" alt="" aria-hidden="true" />
          <p class="startup-kicker" id="startup-title">XREPORT</p>
        </header>

        <p class="startup-status-line" role="status" aria-live="polite">{{ statusLine }}</p>

        <div
          class="startup-progress"
          role="progressbar"
          aria-label="Initializing XREPORT"
          aria-valuemin="0"
          aria-valuemax="100"
          [attr.aria-valuenow]="progress()"
        >
          <div class="startup-progress-fill" [style.width.%]="progress()"></div>
        </div>

        @if (phase === 'unavailable') {
          <button type="button" class="startup-retry" (click)="retryRequested.emit()">Retry connection</button>
        }
      </div>
    </main>
  `,
})
export class StartupScreenComponent implements OnInit, OnChanges, OnDestroy {
  @Input() phase: StartupPhase = 'starting';
  @Input() startedAt: number | null = null;
  @Output() readonly retryRequested = new EventEmitter<void>();
  @Output() readonly transitionComplete = new EventEmitter<void>();

  readonly progress = signal(0);
  private readyTransitionScheduled = false;
  private readyTransitionTimer: ReturnType<typeof setTimeout> | null = null;
  private progressTimer: ReturnType<typeof setInterval> | null = null;

  get statusLine(): string {
    if (this.phase === 'unavailable') return 'Waiting for the local service';
    if (this.phase === 'ready') return 'Opening your workspace';
    return stepForProgress(this.progress());
  }

  ngOnInit(): void {
    this.progressTimer = setInterval(() => this.tickProgress(), PROGRESS_TICK_MS);
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['phase']?.currentValue !== 'ready' || this.readyTransitionScheduled) return;

    this.readyTransitionScheduled = true;
    this.progress.set(100);
    this.stopProgressTimer();
    this.readyTransitionTimer = setTimeout(() => {
      this.readyTransitionTimer = null;
      this.transitionComplete.emit();
    }, READY_TRANSITION_MS);
  }

  ngOnDestroy(): void {
    if (this.readyTransitionTimer !== null) clearTimeout(this.readyTransitionTimer);
    this.stopProgressTimer();
  }

  private tickProgress(): void {
    if (this.phase === 'ready' || this.phase === 'unavailable') return;
    if (this.startedAt === null) return;

    const elapsed = Math.max(0, Date.now() - this.startedAt);
    this.progress.set(progressForElapsed(elapsed));
  }

  private stopProgressTimer(): void {
    if (this.progressTimer === null) return;
    clearInterval(this.progressTimer);
    this.progressTimer = null;
  }
}
