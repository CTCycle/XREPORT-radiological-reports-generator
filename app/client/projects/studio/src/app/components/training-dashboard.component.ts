import { CommonModule } from '@angular/common';
import { AfterViewChecked, AfterViewInit, Component, DestroyRef, ElementRef, EventEmitter, inject, Input, Output, signal, ViewChild } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideActivity, lucideChevronDown, lucideChevronRight, lucideClock, lucideSquare, lucideTarget, lucideTrendingDown } from '@ng-icons/lucide';
import type { TrainingDashboardState } from '@xreport/shared/app/types';

export type TrainingSessionState = 'idle' | 'initializing' | 'running' | 'completed' | 'failed' | 'interrupted';
type LogLevel = 'info' | 'warning' | 'error';

interface ChartPoint { batch: number; value: number; }
interface SeriesModel { key: string; label: string; role: 'train' | 'validation'; color: string; dashed: boolean; values: ChartPoint[]; path: string; rawPath: string | null; areaPath: string | null; markers: { x: number; y: number }[]; latest: number; }
interface ChartModel {
  title: string;
  unit: 'loss' | 'accuracy';
  yAxisLabel: string;
  xMax: number;
  yMax: number;
  series: SeriesModel[];
  yTicks: { y: number; label: string }[];
  xTicks: { x: number; label: string }[];
  epochXs: number[];
}
interface MetricCard { label: string; value: string | null; tone: 'neutral' | 'loss' | 'accuracy'; icon?: string; }
interface LogLine { time: string | null; level: LogLevel; message: string; }

const CHART_HEIGHT = 288;
const MARGIN = { left: 56, right: 16, top: 14, bottom: 42 };
const SMOOTHING_MIN_POINTS = 20;
const SMOOTHING_WEIGHT = 0.9;
const LOG_TIMESTAMP = /^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?|\d{2}:\d{2}:\d{2})\]?\s*[-:]?\s*(.*)$/s;
const RUNNING_PHASES = new Set(['fit_entered', 'first_batch_entered', 'first_batch_completed', 'epoch_completed', 'training_fit_completed']);

function niceScale(max: number, target: number): { step: number; max: number } {
  const raw = Math.max(max, Number.EPSILON) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
  return { step, max: Math.ceil(max / step - 1e-9) * step };
}

function tickValues(max: number, step: number): number[] {
  const ticks: number[] = [];
  for (let value = 0; value <= max + step * 1e-6; value += step) ticks.push(Number(value.toPrecision(10)));
  return ticks;
}

/** Bias-corrected exponential moving average, so the first points are not dragged toward zero. */
function smoothSeries(values: ChartPoint[]): ChartPoint[] {
  let average = 0;
  return values.map((point, index) => {
    average = average * SMOOTHING_WEIGHT + point.value * (1 - SMOOTHING_WEIGHT);
    return { batch: point.batch, value: average / (1 - SMOOTHING_WEIGHT ** (index + 1)) };
  });
}

function compact(value: number): string {
  return value >= 10000 ? `${Number((value / 1000).toFixed(1))}k` : String(value);
}

@Component({
  standalone: true,
  selector: 'app-training-dashboard',
  imports: [CommonModule, NgIcon],
  providers: [provideIcons({ lucideActivity, lucideChevronDown, lucideChevronRight, lucideClock, lucideSquare, lucideTarget, lucideTrendingDown })],
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
        <div class="dashboard-controls">
          <div class="dashboard-status" [ngClass]="sessionState" role="status" aria-live="polite">
            <span class="status-indicator" aria-hidden="true"></span>
            {{ statusLabel }}
          </div>
          <button type="button" class="btn-stop" (click)="stopRequested.emit()" [disabled]="!dashboardState.isTraining">
            <ng-icon name="lucideSquare" aria-hidden="true"/>Stop training
          </button>
        </div>
      </header>

      <section class="session-progress" aria-label="Training progress">
        <div class="session-progress-head">
          <div class="training-phase" aria-live="polite" aria-label="Worker status">
            <span class="training-phase-label">Current phase</span>
            <strong class="training-phase-value">{{ phaseLabel }}</strong>
            @if (showPhaseTime) {
              <span class="training-phase-time">{{ formatTime(dashboardState.workerPhaseElapsedSeconds ?? 0) }} in phase</span>
            }
          </div>
          <div class="session-progress-stats">
            <span class="session-percent">{{ progressLabel }}</span>
            <span class="session-elapsed" title="Elapsed time"><ng-icon name="lucideClock" aria-hidden="true"/>{{ elapsedLabel }}</span>
          </div>
        </div>
        <div class="progress-bar-container" role="progressbar" aria-valuemin="0" aria-valuemax="100" [attr.aria-valuenow]="clampedProgress" [attr.aria-label]="'Training progress: ' + clampedProgress + ' percent'">
          <div class="progress-bar" [ngClass]="sessionState" [class.indeterminate]="sessionState === 'initializing'" [style.width.%]="sessionState === 'initializing' ? null : clampedProgress"></div>
        </div>
      </section>

      @if (error) {
        <div class="training-error" role="alert">{{ error }}</div>
      }

      <div class="dashboard-metrics-grid">
        @for (card of metricCards; track card.label) {
          <div class="dashboard-metric-card">
            <div class="metric-label">
              @if (card.icon) { <ng-icon [name]="card.icon" aria-hidden="true"/> }{{ card.label }}
            </div>
            @if (card.value !== null) {
              <div class="metric-value" [ngClass]="card.tone">{{ card.value }}</div>
            } @else {
              <div class="metric-value unavailable" title="Not available yet">—<span class="sr-only">Not available</span></div>
            }
          </div>
        }
      </div>

      <div class="chart-toolbar">
        <label class="chart-smooth-toggle">
          <input type="checkbox" [checked]="smooth()" (change)="smooth.set(!smooth())"/>
          Smooth curves
        </label>
      </div>

      <div class="training-charts-container">
        @for (chart of chartModels; track chart.title) {
          <div class="chart-section">
            <div class="chart-heading">
              <h3 class="chart-title">{{ chart.title }}</h3>
              <div class="chart-legend" aria-label="Chart legend">
                @for (series of chart.series; track series.key) {
                  <span class="legend-item">
                    <svg class="legend-swatch" width="22" height="8" viewBox="0 0 22 8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" [attr.stroke]="series.color" stroke-width="2" [attr.stroke-dasharray]="series.dashed ? '5 3' : null"/></svg>
                    {{ series.label }}<b>{{ formatValue(chart.unit, series.latest) }}</b>
                  </span>
                }
                @if (chart.epochXs.length) {
                  <span class="legend-item muted"><svg class="legend-swatch" width="22" height="8" viewBox="0 0 22 8" aria-hidden="true"><line x1="11" y1="0" x2="11" y2="8" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2"/></svg>Epoch end</span>
                }
              </div>
            </div>
            <div class="chart-frame">
              <svg class="training-chart" tabindex="0" role="img" [attr.width]="chartWidth()" [attr.height]="chartHeight" [attr.viewBox]="'0 0 ' + chartWidth() + ' ' + chartHeight" [attr.aria-label]="chartLabel(chart)"
                (mousemove)="onChartMove($event, chart)" (mouseleave)="hover.set(null)" (keydown)="onChartKey($event, chart)" (blur)="hover.set(null)">
                @for (tick of chart.yTicks; track tick.y) {
                  <line class="chart-y-grid-line" [attr.x1]="plotLeft" [attr.y1]="tick.y" [attr.x2]="plotRight()" [attr.y2]="tick.y"/>
                  <text class="chart-axis-label" [attr.x]="plotLeft - 8" [attr.y]="tick.y + 4" text-anchor="end">{{ tick.label }}</text>
                }
                <line class="chart-axis-line" [attr.x1]="plotLeft" [attr.y1]="plotBottom" [attr.x2]="plotRight()" [attr.y2]="plotBottom"/>
                @for (tick of chart.xTicks; track tick.x) {
                  <line class="chart-axis-line" [attr.x1]="tick.x" [attr.y1]="plotBottom" [attr.x2]="tick.x" [attr.y2]="plotBottom + 4"/>
                  <text class="chart-axis-label" [attr.x]="tick.x" [attr.y]="plotBottom + 17" text-anchor="middle">{{ tick.label }}</text>
                }
                <text class="chart-axis-title" [attr.x]="(plotLeft + plotRight()) / 2" [attr.y]="chartHeight - 4" text-anchor="middle">Training step (batch)</text>
                <text class="chart-axis-title" text-anchor="middle" [attr.transform]="'translate(12 ' + ((plotTop + plotBottom) / 2) + ') rotate(-90)'">{{ chart.yAxisLabel }}</text>
                @for (x of chart.epochXs; track x) {
                  <line class="chart-epoch-line" [attr.x1]="x" [attr.y1]="plotTop" [attr.x2]="x" [attr.y2]="plotBottom"/>
                }
                @for (series of chart.series; track series.key) {
                  @if (series.areaPath) {
                    <path class="chart-series-area" [attr.d]="series.areaPath" [attr.fill]="series.color"/>
                  }
                  @if (series.rawPath) {
                    <path class="chart-series-line raw" [attr.d]="series.rawPath" [attr.stroke]="series.color"/>
                  }
                  @if (series.values.length > 1) {
                    <path class="chart-series-line" [attr.d]="series.path" [attr.stroke]="series.color" [attr.stroke-dasharray]="series.dashed ? '6 4' : null"/>
                  }
                  @for (marker of series.markers; track $index) {
                    <circle class="chart-marker" [attr.cx]="marker.x" [attr.cy]="marker.y" [attr.r]="3" [attr.fill]="series.color"/>
                  }
                }
                @if (hoverBatch(chart); as batch) {
                  <line class="chart-crosshair" [attr.x1]="batchX(batch.value, chart)" [attr.y1]="plotTop" [attr.x2]="batchX(batch.value, chart)" [attr.y2]="plotBottom"/>
                  @for (row of hoverRows(chart, batch.value); track row.label) {
                    <circle class="chart-hover-dot" [attr.cx]="batchX(row.batch, chart)" [attr.cy]="valueY(row.value, chart)" r="4.5" [attr.stroke]="row.color"/>
                  }
                }
              </svg>
              @if (hoverBatch(chart); as batch) {
                <div class="chart-tooltip" [class.flip]="batchX(batch.value, chart) > chartWidth() * 0.6" [style.left.px]="batchX(batch.value, chart)" role="presentation">
                  <div class="chart-tooltip-title">Step {{ batch.value | number }}</div>
                  @for (row of hoverRows(chart, batch.value); track row.label) {
                    <div class="chart-tooltip-row"><i [style.background]="row.color" aria-hidden="true"></i><span>{{ row.label }}</span><b>{{ formatValue(chart.unit, row.value) }}</b>@if (row.batch !== batch.value) { <small>step {{ row.batch | number }}</small> }</div>
                  }
                </div>
              }
              @if (!chart.series.length) {
                <div class="chart-empty">{{ emptyChartMessage }}</div>
              }
            </div>
          </div>
        }
      </div>

      <section class="dashboard-logs" aria-labelledby="training-log-title">
        <div class="log-header">
          <button type="button" class="log-toggle" [attr.aria-expanded]="logOpen()" aria-controls="training-log-body" (click)="toggleLog()">
            <ng-icon [name]="logOpen() ? 'lucideChevronDown' : 'lucideChevronRight'" aria-hidden="true"/>
            <h3 class="section-heading" id="training-log-title">Training log</h3>
          </button>
          @if (logLines.length) { <span class="log-count">{{ logLines.length }} {{ logLines.length === 1 ? 'entry' : 'entries' }}</span> }
        </div>
        @if (!logOpen()) {
          <!-- collapsed -->
        } @else if (logLines.length) {
          <ol #logList id="training-log-body" class="log-list" role="log" aria-labelledby="training-log-title" tabindex="0" (scroll)="onLogScroll()">
            @for (line of logLines; track $index) {
              <li class="log-line" [ngClass]="line.level">
                <time class="log-time">{{ line.time ?? '' }}</time>
                <span class="log-level">{{ line.level === 'info' ? '' : line.level === 'warning' ? 'WARN' : 'ERROR' }}</span>
                <span class="log-message">{{ line.message }}</span>
              </li>
            }
          </ol>
        } @else {
          <div id="training-log-body" class="log-empty">{{ sessionState === 'idle' ? 'Training output appears here once a run starts.' : 'Waiting for training output…' }}</div>
        }
      </section>
    </section>
  `,
  styleUrl: '../styles/TrainingDashboard.css',
})
export class TrainingDashboardComponent implements AfterViewInit, AfterViewChecked {
  @Input({ required: true }) dashboardState!: TrainingDashboardState;
  @Input() error: string | null = null;
  @Output() readonly stopRequested = new EventEmitter<void>();
  @ViewChild('logList') private logList?: ElementRef<HTMLElement>;

  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  readonly chartHeight = CHART_HEIGHT;
  readonly plotLeft = MARGIN.left;
  readonly plotTop = MARGIN.top;
  readonly plotBottom = CHART_HEIGHT - MARGIN.bottom;
  readonly chartWidth = signal(640);
  readonly smooth = signal(true);
  readonly logOpen = signal(false);
  readonly hover = signal<{ title: string; batch: number } | null>(null);
  private modelCache: { data: unknown; metrics: unknown; epochs: unknown; width: number; smooth: boolean; models: ChartModel[] } | null = null;
  private logCache: { entries: string[]; lines: LogLine[] } | null = null;
  private stickToLogEnd = true;
  private renderedLogCount = 0;

  ngAfterViewInit(): void {
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      const frame = this.host.nativeElement.querySelector<HTMLElement>('.chart-frame');
      const width = frame ? Math.floor(frame.clientWidth) : 0;
      if (width >= 240 && width !== this.chartWidth()) this.chartWidth.set(width);
    });
    observer.observe(this.host.nativeElement);
    this.destroyRef.onDestroy(() => observer.disconnect());
  }

  ngAfterViewChecked(): void {
    const list = this.logList?.nativeElement;
    if (!list || this.renderedLogCount === this.logLines.length) return;
    this.renderedLogCount = this.logLines.length;
    if (this.stickToLogEnd) list.scrollTop = list.scrollHeight;
  }

  toggleLog(): void {
    this.logOpen.update((open) => !open);
    this.stickToLogEnd = true;
    this.renderedLogCount = -1;
  }

  onLogScroll(): void {
    const list = this.logList?.nativeElement;
    if (list) this.stickToLogEnd = list.scrollTop + list.clientHeight >= list.scrollHeight - 8;
  }

  /** Session lifecycle: the polled job status wins, with a fallback inferred from the dashboard fields. */
  get sessionState(): TrainingSessionState {
    const state = this.dashboardState;
    switch (state.sessionStatus) {
      case 'cancelled': return 'interrupted';
      case 'failed': return 'failed';
      case 'completed': return 'completed';
      case 'pending': return 'initializing';
      case 'running': return this.hasStartedFitting ? 'running' : 'initializing';
      default: break;
    }
    if (state.isTraining) return this.hasStartedFitting ? 'running' : 'initializing';
    if (this.error) return 'failed';
    return state.currentEpoch > 0 ? 'completed' : 'idle';
  }

  private get hasStartedFitting(): boolean {
    const state = this.dashboardState;
    return state.currentEpoch > 0 || state.chartData.length > 0 || RUNNING_PHASES.has(state.workerPhase ?? '');
  }

  get statusLabel(): string {
    const labels: Record<TrainingSessionState, string> = { idle: 'Idle', initializing: 'Initializing', running: 'Running', completed: 'Completed', failed: 'Failed', interrupted: 'Interrupted' };
    return labels[this.sessionState];
  }

  get phaseLabel(): string {
    const phase = this.dashboardState.workerPhase;
    if (!phase || phase === 'idle') return 'Waiting to start';
    const labels: Record<string, string> = {
      starting: 'Preparing worker',
      child_started: 'Starting worker process',
      target_started: 'Initializing worker',
      dataset_loading_started: 'Preparing dataset',
      dataset_loaded: 'Dataset loaded',
      image_path_validation_completed: 'Validating image paths',
      device_initialization_started: 'Initializing device',
      device_initialization_completed: 'Device ready',
      checkpoint_initialization_started: 'Preparing checkpoint',
      checkpoint_initialized: 'Checkpoint ready',
      dataloader_construction_started: 'Building data loaders',
      dataloader_construction_completed: 'Data loaders ready',
      model_loading_started: 'Loading model',
      model_loading_completed: 'Model loaded',
      model_compilation_completed: 'Model compiled',
      training_initialization_completed: 'Initializing training',
      fit_requested: 'Entering training',
      fit_entered: 'Training started',
      first_batch_entered: 'Training batch 1',
      first_batch_completed: 'First batch completed',
      epoch_completed: 'Epoch completed',
      training_fit_completed: 'Training complete',
      checkpoint_saving_started: 'Saving checkpoint',
      checkpoint_saving_completed: 'Checkpoint saved',
      worker_completed: 'Worker complete',
      worker_cancelled: 'Worker cancelled',
      result_reported: 'Finalizing result',
    };
    return labels[phase] ?? phase.replaceAll('_', ' ');
  }

  get showPhaseTime(): boolean {
    return this.sessionState === 'initializing' || this.sessionState === 'running';
  }

  get clampedProgress(): number {
    return Math.max(0, Math.min(100, this.dashboardState.progressPercent || 0));
  }

  get progressLabel(): string {
    const state = this.sessionState;
    if (state === 'idle') return 'Not started';
    if (state === 'initializing') return 'Initializing…';
    return `${Number(this.clampedProgress.toFixed(1))}%`;
  }

  get elapsedLabel(): string {
    return this.sessionState === 'idle' ? '—' : this.formatTime(this.dashboardState.elapsedSeconds);
  }

  private hasMetric(key: string): boolean {
    const state = this.dashboardState;
    return state.currentEpoch > 0 || state.chartData.some((point) => typeof point[key] === 'number');
  }

  get metricCards(): MetricCard[] {
    const state = this.dashboardState;
    const loss = (value: number, key: string) => this.hasMetric(key) ? value.toFixed(4) : null;
    const accuracy = (value: number, key: string) => this.hasMetric(key) ? `${(value * 100).toFixed(2)}%` : null;
    const total = state.totalEpochs > 0 ? String(state.totalEpochs) : '—';
    return [
      { label: 'Epoch', value: this.sessionState === 'idle' ? null : `${state.currentEpoch} / ${total}`, tone: 'neutral' },
      { label: 'Train loss', value: loss(state.loss, 'loss'), tone: 'loss', icon: 'lucideTrendingDown' },
      { label: 'Val loss', value: loss(state.valLoss, 'val_loss'), tone: 'loss', icon: 'lucideTrendingDown' },
      { label: 'Train acc', value: accuracy(state.accuracy, 'MaskedAccuracy'), tone: 'accuracy', icon: 'lucideTarget' },
      { label: 'Val acc', value: accuracy(state.valAccuracy, 'val_MaskedAccuracy'), tone: 'accuracy', icon: 'lucideTarget' },
    ];
  }

  get emptyChartMessage(): string {
    switch (this.sessionState) {
      case 'idle': return 'Charts appear once a training run starts.';
      case 'initializing': return 'Preparing the run. Metrics appear after the first batch.';
      case 'running': return 'Waiting for the first batch to complete…';
      case 'failed': return 'The run failed before any metrics were recorded.';
      case 'interrupted': return 'The run was stopped before any metrics were recorded.';
      default: return 'No metrics were recorded for this run.';
    }
  }

  plotRight(): number {
    return this.chartWidth() - MARGIN.right;
  }

  get chartModels(): ChartModel[] {
    const { chartData, availableMetrics, epochBoundaries } = this.dashboardState;
    const width = this.chartWidth();
    const cached = this.modelCache;
    if (cached && cached.data === chartData && cached.metrics === availableMetrics && cached.epochs === epochBoundaries && cached.width === width && cached.smooth === this.smooth()) return cached.models;
    const xMax = Math.max(...chartData.map((point) => point.batch), 1);
    const models = (['loss', 'accuracy'] as const).map((unit) => this.buildChart(unit, xMax, width));
    this.modelCache = { data: chartData, metrics: availableMetrics, epochs: epochBoundaries, width, smooth: this.smooth(), models };
    return models;
  }

  private buildChart(unit: 'loss' | 'accuracy', xMax: number, width: number): ChartModel {
    const { chartData, availableMetrics, epochBoundaries } = this.dashboardState;
    const needle = unit === 'loss' ? 'loss' : 'accuracy';
    const fallback = unit === 'loss' ? ['loss', 'val_loss'] : ['MaskedAccuracy', 'val_MaskedAccuracy'];
    const named = availableMetrics.filter((key) => key.toLowerCase().includes(needle));
    const keys = (named.length ? named : fallback).filter((key) => chartData.some((point) => typeof point[key] === 'number'));
    const scale = unit === 'accuracy' ? 100 : 1;
    const raw = keys.map((key) => ({
      key,
      role: key.toLowerCase().startsWith('val') ? 'validation' as const : 'train' as const,
      values: chartData.flatMap((point) => { const value = point[key]; return typeof value === 'number' && Number.isFinite(value) && Number.isFinite(point.batch) ? [{ batch: point.batch, value: value * scale }] : []; }).sort((a, b) => a.batch - b.batch),
    })).filter((item) => item.values.length).sort((a, b) => Number(a.role === 'validation') - Number(b.role === 'validation'));
    const smooth = this.smooth();
    const shown = raw.map((item) => (smooth && item.role === 'train' && item.values.length >= SMOOTHING_MIN_POINTS ? smoothSeries(item.values) : item.values));
    const dataMax = Math.max(...shown.flatMap((values) => values.map((point) => point.value)), 0);
    const scaled = raw.length ? niceScale(Math.max(dataMax * 1.05, unit === 'accuracy' ? 1 : 0.1), 4) : niceScale(unit === 'accuracy' ? 100 : 1, 4);
    const yMax = unit === 'accuracy' && dataMax <= 100 ? Math.min(scaled.max, 100) : scaled.max;
    const decimals = scaled.step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(scaled.step)));
    const px = (batch: number) => MARGIN.left + (batch / xMax) * (width - MARGIN.left - MARGIN.right);
    const py = (value: number) => this.plotBottom - (Math.min(value, yMax) / yMax) * (this.plotBottom - MARGIN.top);
    const xScale = niceScale(xMax, 6);
    const roleCounts = raw.reduce<Record<string, number>>((counts, item) => ({ ...counts, [item.role]: (counts[item.role] ?? 0) + 1 }), {});
    const toPath = (values: ChartPoint[]) => values.map((point, index) => `${index ? 'L' : 'M'}${px(point.batch).toFixed(1)} ${py(point.value).toFixed(1)}`).join(' ');
    const series: SeriesModel[] = raw.map((item, index) => {
      const line = shown[index];
      const smoothed = line !== item.values;
      const path = toPath(line);
      const base = item.role === 'train' ? 'Training' : 'Validation';
      const validation = item.role === 'validation';
      return {
        key: item.key,
        label: roleCounts[item.role] > 1 ? `${base} (${item.key.replaceAll('_', ' ')})` : base,
        role: item.role,
        color: validation ? 'var(--chart-secondary)' : 'var(--chart-primary)',
        dashed: validation,
        values: item.values,
        path,
        rawPath: smoothed ? toPath(item.values) : null,
        areaPath: !validation && line.length > 1 ? `${path} L${px(line[line.length - 1].batch).toFixed(1)} ${this.plotBottom} L${px(line[0].batch).toFixed(1)} ${this.plotBottom} Z` : null,
        markers: (validation || line.length <= 30 ? line : line.slice(-1)).map((point) => ({ x: px(point.batch), y: py(point.value) })),
        latest: item.values[item.values.length - 1].value,
      };
    });
    return {
      title: unit === 'loss' ? 'Loss' : 'Accuracy',
      unit,
      yAxisLabel: unit === 'loss' ? 'Loss' : 'Accuracy (%)',
      xMax,
      yMax,
      series,
      yTicks: tickValues(yMax, scaled.step).map((value) => ({ y: py(value), label: unit === 'accuracy' ? `${value}%` : value.toFixed(decimals) })),
      xTicks: raw.length ? tickValues(xMax, xScale.step).filter((value) => value <= xMax).map((value) => ({ x: px(value), label: compact(value) })) : [],
      epochXs: series.length ? epochBoundaries.filter((boundary) => boundary > 0 && boundary < xMax).map(px) : [],
    };
  }

  formatValue(unit: 'loss' | 'accuracy', value: number): string {
    return unit === 'loss' ? value.toFixed(4) : `${value.toFixed(2)}%`;
  }

  chartLabel(chart: ChartModel): string {
    if (!chart.series.length) return `${chart.title} chart. ${this.emptyChartMessage}`;
    return `${chart.title} chart over ${chart.xMax} training steps. ${chart.series.map((series) => `${series.label} latest ${this.formatValue(chart.unit, series.latest)}`).join('; ')}. Use left and right arrow keys to inspect values.`;
  }

  batchX(batch: number, chart: ChartModel): number {
    return MARGIN.left + (batch / chart.xMax) * (this.chartWidth() - MARGIN.left - MARGIN.right);
  }

  valueY(value: number, chart: ChartModel): number {
    return this.plotBottom - (Math.min(value, chart.yMax) / chart.yMax) * (this.plotBottom - MARGIN.top);
  }

  hoverBatch(chart: ChartModel): { value: number } | null {
    const hover = this.hover();
    return hover && hover.title === chart.title && chart.series.length ? { value: hover.batch } : null;
  }

  hoverRows(chart: ChartModel, batch: number): { label: string; color: string; value: number; batch: number }[] {
    return chart.series.map((series) => {
      const point = series.values.reduce((best, candidate) => Math.abs(candidate.batch - batch) < Math.abs(best.batch - batch) ? candidate : best);
      return { label: series.label, color: series.color, value: point.value, batch: point.batch };
    });
  }

  private chartBatches(chart: ChartModel): number[] {
    return [...new Set(chart.series.flatMap((series) => series.values.map((point) => point.batch)))].sort((a, b) => a - b);
  }

  onChartMove(event: MouseEvent, chart: ChartModel): void {
    if (!chart.series.length) return;
    const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    const x = (event.clientX - rect.left) * (this.chartWidth() / Math.max(rect.width, 1));
    const target = ((x - MARGIN.left) / (this.chartWidth() - MARGIN.left - MARGIN.right)) * chart.xMax;
    const batch = this.chartBatches(chart).reduce((best, candidate) => Math.abs(candidate - target) < Math.abs(best - target) ? candidate : best);
    this.hover.set({ title: chart.title, batch });
  }

  onChartKey(event: KeyboardEvent, chart: ChartModel): void {
    const batches = this.chartBatches(chart);
    if (!batches.length) return;
    if (event.key === 'Escape') { this.hover.set(null); return; }
    const current = this.hover()?.title === chart.title ? batches.indexOf(this.hover()!.batch) : -1;
    const index = event.key === 'ArrowRight' ? Math.min(batches.length - 1, current + 1) : event.key === 'ArrowLeft' ? Math.max(0, current < 0 ? batches.length - 1 : current - 1) : event.key === 'Home' ? 0 : event.key === 'End' ? batches.length - 1 : -2;
    if (index === -2) return;
    event.preventDefault();
    this.hover.set({ title: chart.title, batch: batches[index] });
  }

  get logLines(): LogLine[] {
    const entries = this.dashboardState.logEntries;
    if (this.logCache?.entries === entries) return this.logCache.lines;
    const lines = entries.map((entry): LogLine => {
      const match = LOG_TIMESTAMP.exec(entry);
      const message = match ? match[2] : entry;
      const level: LogLevel = /\b(error|failed|failure|exception|traceback|fatal)\b/i.test(message) ? 'error' : /\b(warn(?:ing)?|deprecated|cancel(?:l)?ed)\b/i.test(message) ? 'warning' : 'info';
      return { time: match ? match[1] : null, level, message };
    });
    this.logCache = { entries, lines };
    return lines;
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
}
