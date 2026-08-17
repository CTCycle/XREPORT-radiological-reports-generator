import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideBarChart2, lucideCalendar, lucideFileText, lucideImage, lucideListChecks, lucideLoaderCircle, lucideSliders, lucideX } from '@ng-icons/lucide';
import type { ValidationResponse } from '../types/validationApi';
import { ModalFocusDirective } from './modal-focus.directive';

type ReportStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null | undefined;

@Component({
  standalone: true,
  selector: 'app-validation-report-modal',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideBarChart2, lucideCalendar, lucideFileText, lucideImage, lucideListChecks, lucideLoaderCircle, lucideSliders, lucideX })],
  template: `
    @if (open) {
      <div class="modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="report-modal" role="dialog" aria-modal="true" aria-labelledby="validation-report-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="report-header"><div><h3 id="validation-report-title">Validation Report</h3><p class="report-subtitle">Dataset: <strong>{{ datasetName || 'Unknown' }}</strong></p></div><button type="button" class="report-close" aria-label="Close validation report" (click)="closed.emit()"><ng-icon name="lucideX"/></button></header>
          <div class="report-meta"><span class="report-chip"><ng-icon name="lucideCalendar"/>{{ metadata?.date ? 'Generated: ' + metadata?.date : 'Generated: N/A' }}</span><span class="report-chip"><ng-icon name="lucideSliders"/>{{ sampleLabel() }}</span><span class="report-chip"><ng-icon name="lucideListChecks"/>{{ metricLabels().length }} metrics</span></div>
          <div class="report-metrics">@for (metric of metricLabels(); track metric) { <span class="report-metric-pill">{{ metric }}</span> }</div>
          <div class="report-body"><div class="validation-dashboard">
            <div class="dashboard-header"><div class="dashboard-title"><ng-icon name="lucideBarChart2"/>Dataset Validation Results</div><span class="dashboard-status" [class.success]="!loading && !error && !!result" [class.error]="!!error" [class.running]="loading" role="status">{{ statusLabel() }}</span></div>
            @if (loading) { <div class="loading-container"><ng-icon name="lucideLoaderCircle" class="spin"/><span class="loading-text">{{ status === 'pending' ? 'Queued' : 'Running' }} validation... {{ progress ?? 0 }}%</span><div class="progress-bar"><span class="progress-fill" [style.width.%]="progress ?? 0"></span></div></div> }
            @else if (error) { <div class="idle-message error">{{ error }}</div> }
            @else if (result; as report) {
              <p class="result-message">{{ report.message }}</p>
              <div class="validation-content"><div class="stats-grid">
                @if (report.text_statistics; as text) { <div class="stats-section"><div class="stats-section-title"><ng-icon name="lucideFileText"/>Text Statistics</div><div class="stats-row"><span class="stat-label">Total Reports</span><span class="stat-value highlight">{{ text.count | number }}</span></div><div class="stats-row"><span class="stat-label">Total Words</span><span class="stat-value">{{ text.total_words | number }}</span></div><div class="stats-row"><span class="stat-label">Unique Words (Vocabulary)</span><span class="stat-value">{{ text.unique_words | number }}</span></div><div class="stats-row"><span class="stat-label">Avg Words/Report</span><span class="stat-value">{{ text.avg_words_per_report | number:'1.1-1' }}</span></div><div class="stats-row"><span class="stat-label">Min Words/Report</span><span class="stat-value">{{ text.min_words_per_report | number }}</span></div><div class="stats-row"><span class="stat-label">Max Words/Report</span><span class="stat-value">{{ text.max_words_per_report | number }}</span></div></div> }
                @if (report.image_statistics; as image) { <div class="stats-section"><div class="stats-section-title"><ng-icon name="lucideImage"/>Image Statistics</div><div class="stats-row"><span class="stat-label">Total Images</span><span class="stat-value highlight">{{ image.count | number }}</span></div><div class="stats-row"><span class="stat-label">Avg Height</span><span class="stat-value">{{ image.mean_height | number:'1.0-0' }} px</span></div><div class="stats-row"><span class="stat-label">Avg Width</span><span class="stat-value">{{ image.mean_width | number:'1.0-0' }} px</span></div><div class="stats-row"><span class="stat-label">Avg Pixel Value</span><span class="stat-value">{{ image.mean_pixel_value | number:'1.2-2' }}</span></div><div class="stats-row"><span class="stat-label">Std Pixel Value</span><span class="stat-value">{{ image.std_pixel_value | number:'1.2-2' }}</span></div><div class="stats-row"><span class="stat-label">Avg Noise Std</span><span class="stat-value">{{ image.mean_noise_std | number:'1.2-2' }}</span></div><div class="stats-row"><span class="stat-label">Avg Noise Ratio</span><span class="stat-value">{{ image.mean_noise_ratio | number:'1.4-4' }}</span></div></div> }
              </div>
              @if (report.pixel_distribution; as pixels) { <div class="chart-section"><div class="chart-title"><ng-icon name="lucideBarChart2" class="chart-title-icon"/>Pixel Intensity Distribution</div><div class="histogram" role="img" aria-label="Pixel intensity distribution"><div class="histogram-bars">@for (bar of histogram(pixels.counts); track $index) { <span class="histogram-bar" [style.height.%]="bar" [attr.title]="histogramLabel($index)"></span> }</div><div class="chart-axis"><span>0</span><span>64</span><span>128</span><span>192</span><span>255</span></div></div></div> }
              </div>
            } @else { <div class="idle-message">No validation report is available yet.</div> }
          </div></div>
        </section>
      </div>
    }
  `,
  styleUrls: ['../styles/ValidationReportModal.css', '../styles/ValidationDashboard.css'],
})
export class ValidationReportModalComponent {
  @Input() open = false;
  @Input() datasetName: string | null = null;
  @Input() loading = false;
  @Input() result: ValidationResponse | null = null;
  @Input() error: string | null = null;
  @Input() progress: number | null = null;
  @Input() status: ReportStatus = null;
  @Input() metadata: { date?: string | null; sampleSize?: number | null; metrics?: string[] } | null = null;
  @Output() readonly closed = new EventEmitter<void>();
  private readonly labels: Record<string, string> = { pixels_distribution: 'Pixel intensity histogram', text_statistics: 'Text statistics', image_statistics: 'Image statistics' };
  metricLabels() { return (this.metadata?.metrics ?? []).length ? (this.metadata?.metrics ?? []).map((metric) => this.labels[metric] ?? metric.replace(/_/g, ' ')) : ['No metrics recorded']; }
  sampleLabel() { return this.metadata?.sampleSize == null ? 'Sample size: N/A' : `Sample size: ${(Math.max(0, Math.min(this.metadata.sampleSize, 1)) * 100).toFixed(0)}%`; }
  statusLabel() { return this.loading ? (this.status === 'pending' ? 'Queued' : 'Running') : this.error ? 'Error' : this.result ? 'Success' : 'Idle'; }
  histogramLabel(index: number) { const start = index * 4; return `${start}-${Math.min(start + 3, 255)} intensity`; }
  histogram(counts: number[]) { const bins: number[] = []; for (let index = 0; index < Math.min(256, counts.length); index += 4) bins.push(counts.slice(index, index + 4).reduce((sum, value) => sum + value, 0)); const max = Math.max(...bins, 1); return bins.map((value) => Math.max(2, value / max * 100)); }
}
