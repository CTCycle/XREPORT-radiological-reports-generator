import { Component, Input, OnChanges, SimpleChanges, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ValidationApiService } from '../services/validation-api.service';
import type {
  AggregateStatistic,
  HistogramModel,
  RecordKind,
  RecordStatisticsRow,
  ValidationReport,
  ValidationRunSummary,
} from '../types/validationApi';

interface RecordColumn { key: string; label: string; digits?: number; }
interface HistogramBar { height: number; title: string; }
interface HistogramView { name: string; label: string; bars: HistogramBar[]; axis: string[]; total: number; }

const METRIC_LABELS: Record<string, string> = {
  pixels_distribution: 'Pixel intensity histogram',
  text_statistics: 'Text statistics',
  image_statistics: 'Image statistics',
};
const HISTOGRAM_LABELS: Record<string, string> = {
  pixel_intensity: 'Pixel intensity (grayscale levels)',
  report_length: 'Report length (words per report)',
};
const TEXT_MEASURES: Record<string, string> = {
  words_per_report: 'Words per report',
  chars_per_report: 'Characters per report',
  unique_words_per_report: 'Unique words per report',
  tokens_per_report: 'Tokens per report',
};
const IMAGE_MEASURES: Record<string, string> = {
  height: 'Height (px)',
  width: 'Width (px)',
  mean_intensity: 'Mean intensity',
  std_intensity: 'Intensity std',
  noise_std: 'Noise std',
  noise_ratio: 'Noise ratio',
};
const RECORD_COLUMNS: Record<RecordKind, RecordColumn[]> = {
  text: [
    { key: 'image_name', label: 'Record' },
    { key: 'word_count', label: 'Words', digits: 0 },
    { key: 'char_count', label: 'Characters', digits: 0 },
    { key: 'unique_words', label: 'Unique words', digits: 0 },
    { key: 'token_count', label: 'Tokens', digits: 0 },
  ],
  image: [
    { key: 'image_name', label: 'Record' },
    { key: 'height', label: 'Height (px)', digits: 0 },
    { key: 'width', label: 'Width (px)', digits: 0 },
    { key: 'mean_value', label: 'Mean', digits: 2 },
    { key: 'median_value', label: 'Median', digits: 1 },
    { key: 'std_value', label: 'Std', digits: 2 },
    { key: 'min_value', label: 'Min', digits: 0 },
    { key: 'max_value', label: 'Max', digits: 0 },
    { key: 'noise_std', label: 'Noise std', digits: 2 },
    { key: 'noise_ratio', label: 'Noise ratio', digits: 4 },
  ],
};
const PAGE_SIZE = 10;
const MAX_BARS = 64;

/** Persisted validation statistics of one dataset: overview, aggregates, distributions, records and history. */
@Component({
  standalone: true,
  selector: 'app-validation-statistics',
  imports: [FormsModule],
  template: `
    <div class="vstats">
      @if (loading()) {
        <p class="vstats-note" role="status">Loading validation statistics…</p>
      } @else if (error()) {
        <p class="vstats-note vstats-error" role="alert">{{ error() }}</p>
      } @else if (!runs().length) {
        <p class="vstats-note">No validation has been run for this dataset yet.</p>
      } @else if (report(); as current) {
        <section class="vstats-section" aria-labelledby="vstats-overview">
          <h3 id="vstats-overview">Dataset overview</h3>
          <dl class="vstats-kv">
            <div><dt>Dataset</dt><dd>{{ current.dataset_name }}</dd></div>
            <div><dt>Run</dt><dd>#{{ current.run.validation_run_id }} · {{ current.run.date ?? '—' }} UTC</dd></div>
            <div><dt>Sample fraction</dt><dd>{{ percent(current.run.sample_size) }}</dd></div>
            <div><dt>Seed</dt><dd>{{ current.run.seed ?? '—' }}</dd></div>
            <div><dt>Records sampled</dt><dd>{{ format(current.run.records_requested, 0) }}</dd></div>
            <div><dt>Records with valid files</dt><dd>{{ format(current.run.records_evaluated, 0) }}</dd></div>
            <div><dt>Tokenizer</dt><dd>{{ current.run.tokenizer ?? 'Not recorded' }}</dd></div>
            <div><dt>Vocabulary (distinct words)</dt><dd>{{ format(current.run.text_vocabulary_size, 0) }}</dd></div>
          </dl>
          <table class="vstats-table">
            <caption class="sr-only">Metrics computed in this run</caption>
            <thead><tr><th scope="col">Metric</th><th scope="col" class="num">Records computed</th></tr></thead>
            <tbody>
              @for (metric of current.run.metrics ?? []; track metric.metric) {
                <tr><th scope="row">{{ metricLabel(metric.metric) }}</th><td class="num">{{ format(metric.records_computed, 0) }}</td></tr>
              } @empty {
                <tr><td colspan="2">No metrics recorded.</td></tr>
              }
            </tbody>
          </table>
        </section>

        <section class="vstats-section" aria-labelledby="vstats-text">
          <h3 id="vstats-text">Text statistics <span class="vstats-scope">dataset aggregates</span></h3>
          @if (aggregates('text').length) {
            <table class="vstats-table">
              <thead><tr><th scope="col">Measure</th>@for (column of aggregateColumns; track column.key) { <th scope="col" class="num">{{ column.label }}</th> }</tr></thead>
              <tbody>
                @for (row of aggregates('text'); track row.measure) {
                  <tr><th scope="row">{{ textMeasure(row.measure) }}</th>@for (column of aggregateColumns; track column.key) { <td class="num">{{ aggregateValue(row, column.key) }}</td> }</tr>
                }
              </tbody>
            </table>
            @if (totalWords(); as total) { <p class="vstats-footnote">Total words across {{ format(total.count, 0) }} reports: {{ format(total.total, 0) }}</p> }
          } @else {
            <p class="vstats-note">Not computed in this run.</p>
          }
        </section>

        <section class="vstats-section" aria-labelledby="vstats-image">
          <h3 id="vstats-image">Image statistics <span class="vstats-scope">dataset aggregates</span></h3>
          @if (aggregates('image').length) {
            <table class="vstats-table">
              <thead><tr><th scope="col">Measure</th>@for (column of aggregateColumns; track column.key) { <th scope="col" class="num">{{ column.label }}</th> }</tr></thead>
              <tbody>
                @for (row of aggregates('image'); track row.measure) {
                  <tr><th scope="row">{{ imageMeasure(row.measure) }}</th>@for (column of aggregateColumns; track column.key) { <td class="num">{{ aggregateValue(row, column.key) }}</td> }</tr>
                }
              </tbody>
            </table>
          } @else {
            <p class="vstats-note">Not computed in this run.</p>
          }
        </section>

        <section class="vstats-section" aria-labelledby="vstats-distribution">
          <h3 id="vstats-distribution">Distribution analysis</h3>
          @for (view of histograms(); track view.name) {
            <figure class="vstats-histogram">
              <figcaption>{{ view.label }} <span class="vstats-scope">{{ format(view.total, 0) }} observations</span></figcaption>
              <div class="vstats-bars" role="img" [attr.aria-label]="view.label">
                @for (bar of view.bars; track $index) { <span class="vstats-bar" [style.height.%]="bar.height" [attr.title]="bar.title"></span> }
              </div>
              <div class="vstats-axis">@for (tick of view.axis; track $index) { <span>{{ tick }}</span> }</div>
            </figure>
          } @empty {
            <p class="vstats-note">No distributions were computed in this run.</p>
          }
        </section>

        <section class="vstats-section" aria-labelledby="vstats-records">
          <h3 id="vstats-records">Per-record statistics</h3>
          <div class="vstats-toolbar">
            <div class="vstats-segments" role="group" aria-label="Statistics kind">
              <button type="button" [class.active]="recordKind() === 'text'" [attr.aria-pressed]="recordKind() === 'text'" (click)="setKind('text')">Text</button>
              <button type="button" [class.active]="recordKind() === 'image'" [attr.aria-pressed]="recordKind() === 'image'" (click)="setKind('image')">Image</button>
            </div>
            <label class="sr-only" for="vstats-search">Filter records by name</label>
            <input id="vstats-search" class="form-input vstats-search" type="search" placeholder="Filter by record name" [ngModel]="search()" (ngModelChange)="setSearch($event)"/>
          </div>
          @if (recordsError()) {
            <p class="vstats-note vstats-error" role="alert">{{ recordsError() }}</p>
          } @else {
            <div class="vstats-scroll">
              <table class="vstats-table vstats-records">
                <thead>
                  <tr>
                    @for (column of recordColumns(); track column.key) {
                      <th scope="col" [class.num]="column.key !== 'image_name'" [attr.aria-sort]="ariaSort(column.key)">
                        <button type="button" class="vstats-sort" (click)="sortBy(column.key)">{{ column.label }}@if (sort() === column.key) { <span aria-hidden="true">{{ descending() ? ' ▼' : ' ▲' }}</span> }</button>
                      </th>
                    }
                  </tr>
                </thead>
                <tbody>
                  @for (row of rows(); track row['record_id']) {
                    <tr>@for (column of recordColumns(); track column.key) { <td [class.num]="column.key !== 'image_name'">{{ recordCell(row, column) }}</td> }</tr>
                  } @empty {
                    <tr><td [attr.colspan]="recordColumns().length">{{ recordsEmptyMessage() }}</td></tr>
                  }
                </tbody>
              </table>
            </div>
            <div class="vstats-pager">
              <span>{{ pageLabel() }}</span>
              <button type="button" class="btn btn-secondary btn-sm" (click)="page(-1)" [disabled]="offset() === 0">Previous</button>
              <button type="button" class="btn btn-secondary btn-sm" (click)="page(1)" [disabled]="offset() + pageSize >= total()">Next</button>
            </div>
          }
        </section>

        <section class="vstats-section" aria-labelledby="vstats-history">
          <h3 id="vstats-history">Validation history</h3>
          <div class="vstats-scroll">
            <table class="vstats-table vstats-history">
              <thead><tr><th scope="col">Run</th><th scope="col">Executed (UTC)</th><th scope="col" class="num">Fraction</th><th scope="col" class="num">Evaluated / sampled</th><th scope="col">Metrics</th><th scope="col">Tokenizer</th></tr></thead>
              <tbody>
                @for (run of runs(); track run.validation_run_id) {
                  <tr [class.selected]="run.validation_run_id === selectedRunId()">
                    <th scope="row"><button type="button" class="vstats-sort" [attr.aria-current]="run.validation_run_id === selectedRunId() ? 'true' : null" (click)="selectRun(run.validation_run_id)">#{{ run.validation_run_id }}</button></th>
                    <td>{{ run.date ?? '—' }}</td>
                    <td class="num">{{ percent(run.sample_size) }}</td>
                    <td class="num">{{ format(run.records_evaluated, 0) }} / {{ format(run.records_requested, 0) }}</td>
                    <td>{{ metricSummary(run) }}</td>
                    <td>{{ run.tokenizer ?? '—' }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </section>
      }
    </div>
  `,
  styleUrl: '../styles/ValidationStatistics.css',
})
export class ValidationStatisticsComponent implements OnChanges {
  private readonly api = inject(ValidationApiService);
  @Input() datasetName: string | null = null;
  /** Changing this value reloads the history and selects the newest run. */
  @Input() refreshToken = 0;

  readonly aggregateColumns = [
    { key: 'count', label: 'Count' }, { key: 'minimum', label: 'Min' }, { key: 'p05', label: 'P5' }, { key: 'p25', label: 'P25' },
    { key: 'median', label: 'Median' }, { key: 'mean', label: 'Mean' }, { key: 'p75', label: 'P75' }, { key: 'p95', label: 'P95' },
    { key: 'maximum', label: 'Max' }, { key: 'std', label: 'Std' },
  ];
  readonly pageSize = PAGE_SIZE;
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly runs = signal<ValidationRunSummary[]>([]);
  readonly selectedRunId = signal<number | null>(null);
  readonly report = signal<ValidationReport | null>(null);
  readonly recordKind = signal<RecordKind>('text');
  readonly rows = signal<RecordStatisticsRow[]>([]);
  readonly total = signal(0);
  readonly offset = signal(0);
  readonly sort = signal<string>('image_name');
  readonly descending = signal(false);
  readonly search = signal('');
  readonly recordsError = signal<string | null>(null);
  private loadSequence = 0;
  private searchTimer: ReturnType<typeof setTimeout> | undefined;

  ngOnChanges(changes: SimpleChanges) {
    if (changes['datasetName'] || changes['refreshToken']) void this.load(true);
  }

  async selectRun(runId: number) {
    if (runId === this.selectedRunId()) return;
    this.selectedRunId.set(runId);
    await this.loadRun(runId);
  }

  setKind(kind: RecordKind) {
    if (kind === this.recordKind()) return;
    this.recordKind.set(kind);
    this.sort.set('image_name');
    this.descending.set(false);
    this.offset.set(0);
    void this.loadRecords();
  }

  setSearch(value: string) {
    this.search.set(value);
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => { this.offset.set(0); void this.loadRecords(); }, 250);
  }

  sortBy(key: string) {
    if (this.sort() === key) this.descending.update((value) => !value);
    else { this.sort.set(key); this.descending.set(false); }
    this.offset.set(0);
    void this.loadRecords();
  }

  page(direction: -1 | 1) {
    this.offset.update((value) => Math.max(0, value + direction * PAGE_SIZE));
    void this.loadRecords();
  }

  ariaSort(key: string) { return this.sort() === key ? (this.descending() ? 'descending' : 'ascending') : 'none'; }
  recordColumns() { return RECORD_COLUMNS[this.recordKind()]; }
  aggregates(domain: AggregateStatistic['domain']) { return (this.report()?.aggregates ?? []).filter((item) => item.domain === domain); }
  totalWords() { return this.aggregates('text').find((item) => item.measure === 'words_per_report' && item.total != null) ?? null; }
  metricLabel(metric: string) { return METRIC_LABELS[metric] ?? metric.replace(/_/g, ' '); }
  textMeasure(measure: string) { return TEXT_MEASURES[measure] ?? measure.replace(/_/g, ' '); }
  imageMeasure(measure: string) { return IMAGE_MEASURES[measure] ?? measure.replace(/_/g, ' '); }
  metricSummary(run: ValidationRunSummary) { return (run.metrics ?? []).map((item) => this.metricLabel(item.metric)).join(', ') || '—'; }
  percent(value: number | null | undefined) { return value == null ? '—' : `${(value * 100).toFixed(0)}%`; }

  /** Missing values are shown as an em dash, never as zero. */
  format(value: number | null | undefined, digits = 2) {
    return value == null || !Number.isFinite(value) ? '—' : new Intl.NumberFormat(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
  }

  aggregateValue(row: AggregateStatistic, key: string) {
    const value = (row as unknown as Record<string, number | null | undefined>)[key];
    const digits = key === 'count' ? 0 : row.measure === 'noise_ratio' ? 4 : row.measure === 'height' || row.measure === 'width' ? 1 : 2;
    return this.format(value, digits);
  }

  recordCell(row: RecordStatisticsRow, column: RecordColumn) {
    const value = row[column.key];
    if (column.key === 'image_name') return String(value ?? '—');
    return this.format(typeof value === 'number' ? value : null, column.digits ?? 2);
  }

  recordsEmptyMessage() {
    if (this.search()) return 'No records match the filter.';
    return this.recordKind() === 'text' ? 'Text statistics were not computed in this run.' : 'Image statistics were not computed in this run.';
  }

  pageLabel() {
    const total = this.total();
    if (!total) return '0 records';
    return `${this.offset() + 1}–${Math.min(this.offset() + PAGE_SIZE, total)} of ${total}`;
  }

  histograms(): HistogramView[] {
    return (this.report()?.histograms ?? []).filter((item) => item.bins.length).map((item) => this.toHistogramView(item));
  }

  private toHistogramView(histogram: HistogramModel): HistogramView {
    const group = Math.max(1, Math.ceil(histogram.bins.length / MAX_BARS));
    const grouped: { lower: number; upper: number; count: number }[] = [];
    for (let index = 0; index < histogram.bins.length; index += group) {
      const slice = histogram.bins.slice(index, index + group);
      grouped.push({ lower: slice[0].lower, upper: slice[slice.length - 1].upper, count: slice.reduce((sum, bin) => sum + bin.count, 0) });
    }
    const peak = Math.max(...grouped.map((bin) => bin.count), 1);
    const first = grouped[0].lower;
    const last = grouped[grouped.length - 1].upper;
    const digits = last - first > 20 ? 0 : 1;
    return {
      name: histogram.name,
      label: HISTOGRAM_LABELS[histogram.name] ?? histogram.name.replace(/_/g, ' '),
      bars: grouped.map((bin) => ({ height: bin.count === 0 ? 0 : Math.max(2, (bin.count / peak) * 100), title: `${this.format(bin.lower, digits)}–${this.format(bin.upper, digits)}: ${this.format(bin.count, 0)}` })),
      axis: [first, first + (last - first) / 2, last].map((value) => this.format(value, digits)),
      total: grouped.reduce((sum, bin) => sum + bin.count, 0),
    };
  }

  private async load(selectLatest: boolean) {
    const dataset = this.datasetName;
    const sequence = ++this.loadSequence;
    this.report.set(null);
    this.rows.set([]);
    this.error.set(null);
    if (!dataset) { this.runs.set([]); return; }
    this.loading.set(true);
    const history = await this.api.listRuns(dataset);
    if (sequence !== this.loadSequence) return;
    if (history.error && !history.result) {
      // The dataset has never been validated: the API answers 404, which is an empty state, not a failure.
      const missing = history.error.startsWith('404');
      this.runs.set([]);
      this.error.set(missing ? null : history.error);
      this.loading.set(false);
      return;
    }
    const runs = history.result?.runs ?? [];
    this.runs.set(runs);
    const keep = !selectLatest && runs.some((run) => run.validation_run_id === this.selectedRunId());
    const runId = keep ? this.selectedRunId() : runs[0]?.validation_run_id ?? null;
    this.selectedRunId.set(runId);
    this.loading.set(false);
    if (runId != null) await this.loadRun(runId);
  }

  private async loadRun(runId: number) {
    const dataset = this.datasetName;
    if (!dataset) return;
    const sequence = ++this.loadSequence;
    const detail = await this.api.getRun(dataset, runId);
    if (sequence !== this.loadSequence) return;
    if (!detail.result) { this.error.set(detail.error ?? 'Failed to load the validation run'); return; }
    this.error.set(null);
    this.report.set(detail.result);
    this.offset.set(0);
    await this.loadRecords();
  }

  private async loadRecords() {
    const dataset = this.datasetName;
    const runId = this.selectedRunId();
    if (!dataset || runId == null) return;
    const kind = this.recordKind();
    const result = await this.api.getRecords(dataset, runId, { kind, sort: this.sort(), descending: this.descending(), search: this.search().trim() || undefined, limit: PAGE_SIZE, offset: this.offset() });
    if (kind !== this.recordKind() || runId !== this.selectedRunId()) return;
    this.recordsError.set(result.result ? null : result.error ?? 'Failed to load record statistics');
    this.rows.set(result.result?.rows ?? []);
    this.total.set(result.result?.total ?? 0);
  }
}
