import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideBarChart2, lucideCalendar, lucideListChecks, lucideLoaderCircle, lucideX } from '@ng-icons/lucide';
import type { CheckpointEvaluationReport } from '../types/inferenceApi';
import { ModalFocusDirective } from './modal-focus.directive';

@Component({
  standalone: true,
  selector: 'app-checkpoint-evaluation-report-modal',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideBarChart2, lucideCalendar, lucideListChecks, lucideLoaderCircle, lucideX })],
  template: `
    @if (open) {
      <div class="modal-backdrop" role="presentation" (click)="closed.emit()"><section class="report-modal" role="dialog" aria-modal="true" aria-labelledby="checkpoint-report-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
        <header class="report-header"><div><h3 id="checkpoint-report-title">Checkpoint Evaluation Report</h3><p class="report-subtitle">Checkpoint: <strong>{{ checkpoint || 'Unknown' }}</strong></p></div><button type="button" class="report-close" aria-label="Close evaluation report" (click)="closed.emit()"><ng-icon name="lucideX"/></button></header>
        <div class="report-meta"><span class="report-chip"><ng-icon name="lucideCalendar"/>{{ report?.date ? 'Generated: ' + report?.date : 'Generated: N/A' }}</span><span class="report-chip"><ng-icon name="lucideListChecks"/>{{ report?.metrics?.length || 0 }} metrics</span></div>
        <div class="report-metrics">@for (metric of report?.metrics || []; track metric) { <span class="report-metric-pill">{{ metricLabel(metric) }}</span> }</div>
        <div class="report-body"><div class="validation-dashboard"><div class="dashboard-header"><div class="dashboard-title"><ng-icon name="lucideBarChart2"/>Checkpoint Evaluation Results</div><span class="dashboard-status" [class.success]="hasResults()" [class.running]="loading" [class.error]="!!error">{{ statusLabel() }}</span></div>@if (loading) { <div class="loading-container"><ng-icon name="lucideLoaderCircle" class="spin"/><span class="loading-text">Running checkpoint evaluation... {{ progress ?? 0 }}%</span><div class="progress-bar"><div class="progress-fill" [style.width.%]="progress ?? 0"></div></div></div> } @else if (error) { <div class="idle-message error">{{ error }}</div> } @else if (hasResults()) { <div class="validation-content"><div class="stats-grid"><div class="stats-section"><div class="stats-section-title">Metrics</div><div class="stats-row"><span class="stat-label">Loss</span><span class="stat-value">{{ formatMetric(report?.results?.loss) }}</span></div><div class="stats-row"><span class="stat-label">Accuracy</span><span class="stat-value">{{ formatMetric(report?.results?.accuracy) }}</span></div><div class="stats-row"><span class="stat-label">BLEU Score</span><span class="stat-value">{{ formatMetric(report?.results?.bleu_score) }}</span></div></div></div></div> } @else { <div class="idle-message">Select evaluation options and run the checkpoint evaluation to see results.</div> }</div></div>
      </section></div>
    }
  `,
  styleUrls: ['../styles/ValidationReportModal.css', '../styles/ValidationDashboard.css'],
})
export class CheckpointEvaluationReportModalComponent {
  @Input() open = false;
  @Input() checkpoint: string | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;
  @Input() progress: number | null = null;
  @Input() report: CheckpointEvaluationReport | null = null;
  @Output() readonly closed = new EventEmitter<void>();

  hasResults() { const results = this.report?.results; return typeof results?.loss === 'number' || typeof results?.accuracy === 'number' || typeof results?.bleu_score === 'number'; }
  formatMetric(value: number | null | undefined) { return typeof value === 'number' ? value.toFixed(4) : '--'; }
  statusLabel() { return this.loading ? 'Running' : this.error ? 'Error' : this.hasResults() ? 'Complete' : 'Idle'; }
  metricLabel(metric: string) {
    const labels: Record<string, string> = { evaluation_report: 'Evaluation report', bleu_score: 'BLEU score' };
    const config = this.report?.metric_configs?.[metric];
    const parts: string[] = [];
    const dataFraction = config?.['data_fraction'];
    const numSamples = config?.['num_samples'];
    if (typeof dataFraction === 'number') parts.push(`${Math.round(dataFraction * 100)}%`);
    if (typeof numSamples === 'number') parts.push(`${numSamples} samples`);
    return `${labels[metric] ?? metric.replace(/_/g, ' ')}${parts.length ? ` · ${parts.join(' · ')}` : ''}`;
  }
}
