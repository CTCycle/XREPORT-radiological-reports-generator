import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import {
  lucideActivity,
  lucideClock,
  lucidePercent,
  lucideSquare,
  lucideTarget,
  lucideTrendingDown,
} from '@ng-icons/lucide';
import type { TrainingDashboardState } from '../types';

interface ChartSeries {
  key: string;
  label: string;
  color: string;
  values: { batch: number; value: number }[];
}

interface ChartGroup {
  title: string;
  maxBatch: number;
  maxValue: number;
  series: ChartSeries[];
}

@Component({
  standalone: true,
  selector: 'app-training-dashboard',
  imports: [CommonModule, NgIcon],
  providers: [provideIcons({ lucideActivity, lucideClock, lucidePercent, lucideSquare, lucideTarget, lucideTrendingDown })],
  template: `
    <section class="training-dashboard" aria-labelledby="training-dashboard-title">
      <header class="dashboard-header">
        <div class="dashboard-title">
          <ng-icon name="lucideActivity" aria-hidden="true"/>
          <div>
            <span class="eyebrow">Live session</span>
            <h2 id="training-dashboard-title">Training Dashboard</h2>
          </div>
        </div>
        <div class="dashboard-status" [class.training]="statusKind === 'training'" [class.complete]="statusKind === 'complete'" [class.error]="statusKind === 'error'" [class.idle]="statusKind === 'idle'" role="status" aria-live="polite">
          <span class="status-indicator" [class.training]="statusKind === 'training'" [class.complete]="statusKind === 'complete'" [class.error]="statusKind === 'error'" [class.idle]="statusKind === 'idle'" aria-hidden="true"></span>
          {{ statusLabel }}
        </div>
      </header>

      <div class="dashboard-metrics-row">
        <div class="dashboard-metrics-grid">
          <div class="dashboard-metric-card">
            <div class="metric-label">Epoch</div>
            <div class="metric-value">{{ dashboardState.currentEpoch }} / {{ dashboardState.totalEpochs || '--' }}</div>
          </div>
          <div class="dashboard-metric-card">
            <div class="metric-label"><ng-icon name="lucideTrendingDown" aria-hidden="true"/>Train Loss</div>
            <div class="metric-value loss">{{ dashboardState.loss | number:'1.3-3' }}</div>
          </div>
          <div class="dashboard-metric-card">
            <div class="metric-label"><ng-icon name="lucideTrendingDown" aria-hidden="true"/>Val Loss</div>
            <div class="metric-value loss">{{ dashboardState.valLoss | number:'1.3-3' }}</div>
          </div>
          <div class="dashboard-metric-card">
            <div class="metric-label"><ng-icon name="lucideTarget" aria-hidden="true"/>Train Acc</div>
            <div class="metric-value accuracy">{{ dashboardState.accuracy * 100 | number:'1.3-3' }}%</div>
          </div>
          <div class="dashboard-metric-card">
            <div class="metric-label"><ng-icon name="lucideTarget" aria-hidden="true"/>Val Acc</div>
            <div class="metric-value accuracy">{{ dashboardState.valAccuracy * 100 | number:'1.3-3' }}%</div>
          </div>
        </div>
        <button type="button" class="btn-stop" (click)="stopRequested.emit()" [disabled]="!dashboardState.isTraining">
          <ng-icon name="lucideSquare" aria-hidden="true"/>Stop training
        </button>
      </div>

      <section class="progress-section" aria-label="Training progress">
        <div class="progress-header">
          <span class="progress-label"><ng-icon name="lucidePercent" aria-hidden="true"/>Progress: {{ clampedProgress | number:'1.0-0' }}%</span>
          <span class="progress-time"><ng-icon name="lucideClock" aria-hidden="true"/>{{ formatTime(dashboardState.elapsedSeconds) }}</span>
        </div>
        <div class="progress-bar-row">
          <div class="progress-bar-container" role="progressbar" [attr.aria-valuenow]="clampedProgress" aria-valuemin="0" aria-valuemax="100" [attr.aria-label]="'Training progress: ' + clampedProgress + ' percent'">
            <div class="progress-bar" [style.width.%]="clampedProgress"></div>
          </div>
        </div>
      </section>

      <div class="training-charts-container">
        @if (chartGroups.length) {
          @for (chart of chartGroups; track chart.title) {
            <div class="chart-section">
              <div class="chart-title">{{ chart.title }}</div>
              <svg class="training-chart" viewBox="0 0 640 240" role="img" [attr.aria-labelledby]="chartId(chart.title)">
                <title [attr.id]="chartId(chart.title)">{{ chart.title }} training chart</title>
                <line x1="48" y1="24" x2="48" y2="210" stroke="currentColor" opacity=".25"/>
                <line x1="48" y1="210" x2="620" y2="210" stroke="currentColor" opacity=".25"/>
                @for (boundary of dashboardState.epochBoundaries; track boundary) {
                  <line [attr.x1]="chartX(boundary, chart.maxBatch)" y1="24" [attr.x2]="chartX(boundary, chart.maxBatch)" y2="210" stroke="currentColor" opacity=".25" stroke-dasharray="4 4"/>
                }
                @for (series of chart.series; track series.key) {
                  <polyline [attr.points]="chartPoints(series.values, chart.maxValue, chart.maxBatch)" fill="none" [attr.stroke]="series.color" stroke-width="2" [attr.aria-label]="series.label"/>
                }
                <text x="48" y="232" font-size="11">0</text>
                <text x="600" y="232" font-size="11">{{ chart.maxBatch }}</text>
              </svg>
              <div class="chart-legend" aria-label="Chart legend">
                @for (series of chart.series; track series.key) {
                  <span><i [style.background]="series.color" aria-hidden="true"></i>{{ series.label }}</span>
                }
              </div>
            </div>
          }
        } @else {
          @for (title of emptyChartTitles; track title) {
            <div class="chart-section">
              <div class="chart-title">{{ title }}</div>
              <div class="chart-placeholder">Waiting for training data...</div>
            </div>
          }
        }
      </div>

      <section class="dashboard-logs" aria-labelledby="training-log-title">
        <div class="log-header" id="training-log-title">Training log</div>
        @if (dashboardState.logEntries.length) {
          <pre class="log-body">{{ dashboardState.logEntries.join('\n') }}</pre>
        } @else {
          <div class="log-empty">No training output yet.</div>
        }
      </section>
    </section>
  `,
  styleUrl: '../styles/TrainingDashboard.css',
})
export class TrainingDashboardComponent {
  @Input({ required: true }) dashboardState!: TrainingDashboardState;
  @Input() error: string | null = null;
  @Output() readonly stopRequested = new EventEmitter<void>();

  readonly emptyChartTitles = ['Loss', 'Accuracy'];

  get statusKind(): 'training' | 'complete' | 'error' | 'idle' {
    if (this.dashboardState.isTraining) return 'training';
    if (this.error) return 'error';
    return this.dashboardState.currentEpoch > 0 ? 'complete' : 'idle';
  }

  get statusLabel(): string {
    if (this.statusKind === 'training') return 'Training in progress';
    if (this.statusKind === 'error') return 'Failed';
    if (this.statusKind === 'complete') return 'Complete';
    return 'Idle';
  }

  get clampedProgress(): number {
    return Math.max(0, Math.min(100, this.dashboardState.progressPercent || 0));
  }

  get chartGroups(): ChartGroup[] {
    const data = this.dashboardState.chartData;
    if (!data.length) return [];
    const groups = [
      { title: 'Loss', keys: this.dashboardState.availableMetrics.filter((key) => key.toLowerCase().includes('loss')), fallback: ['loss', 'val_loss'] },
      { title: 'Accuracy', keys: this.dashboardState.availableMetrics.filter((key) => key.toLowerCase().includes('accuracy')), fallback: ['MaskedAccuracy', 'val_MaskedAccuracy'] },
    ];
    return groups.map((group) => {
      const keys = group.keys.length ? group.keys : group.fallback.filter((key) => data.some((point) => typeof point[key] === 'number'));
      const series = keys.map((key, index) => ({
        key,
        label: key.replaceAll('_', ' '),
        color: index === 0 ? '#2563eb' : '#0d9488',
        values: data.flatMap((point) => typeof point[key] === 'number' ? [{ batch: point.batch, value: point[key] as number }] : []),
      })).filter((item) => item.values.length);
      const maxBatch = Math.max(...data.map((point) => point.batch), 1);
      const maxValue = Math.max(...series.flatMap((item) => item.values.map((point) => point.value)), 1);
      return { title: group.title, maxBatch, maxValue, series };
    }).filter((group) => group.series.length);
  }

  formatTime(seconds: number): string {
    const safeSeconds = Math.max(0, Math.floor(seconds || 0));
    const hrs = Math.floor(safeSeconds / 3600);
    const mins = Math.floor((safeSeconds % 3600) / 60);
    const secs = safeSeconds % 60;
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  }

  chartId(title: string): string {
    return `training-chart-${title.toLowerCase()}`;
  }

  chartX(batch: number, maxBatch: number): number {
    return 48 + Math.max(0, Math.min(1, batch / Math.max(maxBatch, 1))) * 572;
  }

  chartPoints(values: { batch: number; value: number }[], maxValue: number, maxBatch: number): string {
    return values.map((point) => `${this.chartX(point.batch, maxBatch)},${210 - Math.max(0, Math.min(1, point.value / Math.max(maxValue, 1))) * 186}`).join(' ');
  }
}
