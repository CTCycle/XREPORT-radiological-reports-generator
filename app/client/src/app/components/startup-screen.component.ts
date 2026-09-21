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
              <svg class="startup-xray" viewBox="0 0 420 470" focusable="false" aria-hidden="true">
                <path class="startup-xray-rib startup-xray-rib--left" d="M210 74C153 77 105 115 94 184c-8 49 10 117 55 166 21 23 42 37 61 43" />
                <path class="startup-xray-rib startup-xray-rib--right" d="M210 74c57 3 105 41 116 110 8 49-10 117-55 166-21 23-42 37-61 43" />
                <path class="startup-xray-spine" d="M210 87v292" />
                <path class="startup-xray-lung startup-xray-lung--left" d="M202 119c-42 7-69 41-72 91-3 48 15 105 48 145 10 12 18 18 24 21Z" />
                <path class="startup-xray-lung startup-xray-lung--right" d="M218 119c42 7 69 41 72 91 3 48-15 105-48 145-10 12-18 18-24 21Z" />
                <path class="startup-xray-mediastinum" d="M207 119c-9 38-10 69-1 94 6 16 9 35 9 58" />
                <line class="startup-scan-line" x1="91" x2="329" y1="84" y2="84" />
                <g class="startup-xray-markers">
                  <circle cx="151" cy="181" r="7" />
                  <circle cx="268" cy="253" r="7" />
                  <circle cx="185" cy="326" r="7" />
                </g>
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
