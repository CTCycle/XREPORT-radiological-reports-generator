import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges } from '@angular/core';
import type { StartupPhase } from '../services/startup-readiness.service';

const READY_TRANSITION_MS = 320;

const PHASE_COPY: Record<StartupPhase, { title: string; message: string }> = {
  starting: {
    title: 'Preparing XREPORT',
    message: 'Initializing radiology report generation',
  },
  slow: {
    title: 'XREPORT is still initializing',
    message: 'Local services and application data are still being prepared.',
  },
  unavailable: {
    title: 'XREPORT could not reach the local backend service',
    message: 'Initialization may still be in progress. XREPORT will keep checking automatically.',
  },
  ready: {
    title: 'XREPORT is ready',
    message: 'Opening your workspace',
  },
};

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
          <p class="startup-kicker">XREPORT</p>
          <h1 id="startup-title">Radiological Reports Generator</h1>
        </header>

        <div class="startup-layout">
          <div class="startup-visual" aria-hidden="true">
            <div class="startup-xray-frame">
              <svg class="startup-xray" viewBox="0 0 360 420" focusable="false" aria-hidden="true">
                <path class="startup-xray-thorax" d="M180 34C129 36 82 70 65 122c-17 51-12 124 19 183 21 39 57 67 96 77 39-10 75-38 96-77 31-59 36-132 19-183C278 70 231 36 180 34Z" />
                <path class="startup-xray-lung startup-xray-lung--left" d="M170 87c-35 5-59 35-65 78-7 50 8 105 39 146 9 12 18 20 27 26 6-39 8-78 7-115-1-47-3-91-8-135Z" />
                <path class="startup-xray-lung startup-xray-lung--right" d="M190 87c35 5 59 35 65 78 7 50-8 105-39 146-9 12-18 20-27 26-6-39-8-78-7-115 1-47 3-91 8-135Z" />
                <path class="startup-xray-spine" d="M180 62v288" />
                <path class="startup-xray-sternum" d="M180 82c-8 31-9 68-3 105 4 23 5 47 3 73" />
                <path class="startup-xray-diaphragm" d="M91 302c28 15 57 22 89 22s61-7 89-22" />
                <line class="startup-xray-scan" x1="55" x2="305" y1="52" y2="52" />
              </svg>
            </div>
          </div>

          <section class="startup-report" aria-label="Structured report preview">
            <div class="startup-report-heading"><span>REPORT STRUCTURE</span><span class="startup-report-pulse">ANALYZING</span></div>
            <div class="startup-report-section">
              <h2>FINDINGS</h2>
              <span class="startup-placeholder startup-placeholder--wide"></span>
              <span class="startup-placeholder startup-placeholder--medium"></span>
              <span class="startup-placeholder startup-placeholder--short"></span>
            </div>
            <div class="startup-report-section">
              <h2>IMPRESSION</h2>
              <span class="startup-placeholder startup-placeholder--wide"></span>
              <span class="startup-placeholder startup-placeholder--medium"></span>
            </div>
          </section>
        </div>

        <section
          class="startup-status"
          [attr.role]="phase === 'unavailable' ? 'alert' : 'status'"
          aria-live="polite"
          aria-atomic="true"
        >
          <h2 class="startup-status-title">{{ copy.title }}</h2>
          <p class="startup-status-copy">{{ copy.message }}</p>
          @if (phase === 'unavailable') {
            <button type="button" class="startup-retry" (click)="retryRequested.emit()">Retry connection</button>
          }
        </section>
      </div>
    </main>
  `,
})
export class StartupScreenComponent implements OnChanges, OnDestroy {
  @Input() phase: StartupPhase = 'starting';
  @Output() readonly retryRequested = new EventEmitter<void>();
  @Output() readonly transitionComplete = new EventEmitter<void>();

  private readyTransitionScheduled = false;
  private readyTransitionTimer: ReturnType<typeof setTimeout> | null = null;

  get copy(): { title: string; message: string } {
    return PHASE_COPY[this.phase];
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['phase']?.currentValue !== 'ready' || this.readyTransitionScheduled) return;

    this.readyTransitionScheduled = true;
    this.readyTransitionTimer = setTimeout(() => {
      this.readyTransitionTimer = null;
      this.transitionComplete.emit();
    }, READY_TRANSITION_MS);
  }

  ngOnDestroy(): void {
    if (this.readyTransitionTimer !== null) clearTimeout(this.readyTransitionTimer);
  }
}
