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
              <img class="startup-xray-image" src="startup-radiograph.png" alt="" aria-hidden="true" />
              <span class="startup-xray-scan" aria-hidden="true"></span>
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
