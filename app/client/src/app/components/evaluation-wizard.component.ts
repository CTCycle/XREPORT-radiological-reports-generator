import { CommonModule } from '@angular/common';
import { Component, computed, EventEmitter, Input, OnChanges, Output, signal, SimpleChanges } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideActivity, lucideArrowLeft, lucideArrowRight, lucideBarChart2, lucideCheck, lucideTarget, lucideX } from '@ng-icons/lucide';
import { ModalFocusDirective } from './modal-focus.directive';

export interface EvaluationMetricConfig {
  dataFraction: number;
}

export interface EvaluationWizardConfirmPayload {
  metrics: string[];
  metricConfigs: Record<string, EvaluationMetricConfig>;
}

interface MetricCatalogItem {
  id: string;
  title: string;
  description: string;
  icon: string;
  defaultFraction: number;
}

const METRICS_CATALOG: MetricCatalogItem[] = [
  {
    id: 'evaluation_report',
    title: 'Evaluation Report',
    description: 'Standard validation including Loss and Accuracy metrics on the validation dataset.',
    icon: 'lucideTarget',
    defaultFraction: 1,
  },
  {
    id: 'bleu_score',
    title: 'BLEU Score',
    description: 'Bilingual Evaluation Understudy score to measure text generation quality.',
    icon: 'lucideBarChart2',
    defaultFraction: 0.1,
  },
];

@Component({
  standalone: true,
  selector: 'app-evaluation-wizard',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideActivity, lucideArrowLeft, lucideArrowRight, lucideBarChart2, lucideCheck, lucideTarget, lucideX })],
  template: `
    @if (open) {
      <div class="wizard-overlay" role="presentation">
        <section class="wizard-container" role="dialog" aria-modal="true" aria-labelledby="evaluation-wizard-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="wizard-header">
            <h2 id="evaluation-wizard-title"><ng-icon name="lucideActivity"/>Evaluation Wizard</h2>
            <button type="button" class="btn-wizard-close" aria-label="Close evaluation wizard" (click)="closed.emit()"><ng-icon name="lucideX"/></button>
          </header>

          <div class="wizard-content">
            <div class="wizard-steps" aria-label="Evaluation configuration progress">
              @for (step of stepIndicators(); track $index) {
                <span class="step-indicator" [class.active]="$index === currentStepIndex()" [class.completed]="$index < currentStepIndex()"></span>
              }
            </div>

            @if (isFirstStep()) {
              <div class="metrics-step">
                <h3 class="wizard-step-title-text">Select Evaluation Metrics</h3>
                <div class="metrics-grid">
                  @for (metric of metricCatalog; track metric.id) {
                    <button type="button" class="metric-card" [class.selected]="isSelected(metric.id)" [attr.aria-pressed]="isSelected(metric.id)" (click)="toggleMetric(metric.id)">
                      <div class="metric-icon"><ng-icon [name]="metric.icon"/></div>
                      <div class="metric-check"><ng-icon name="lucideCheck"/></div>
                      <h3>{{ metric.title }}</h3>
                      <p>{{ metric.description }}</p>
                    </button>
                  }
                </div>
              </div>
            }

            @if (currentMetricId(); as metricId) {
              @if (metricById(metricId); as metric) {
                <div class="config-container">
                  <div class="config-header">
                    <div class="config-header-icon"><ng-icon [name]="metric.icon"/></div>
                    <h3>Configure {{ metric.title }}</h3>
                    <p>{{ metric.description }}</p>
                  </div>
                  <div class="config-form">
                    <div class="wizard-separator"></div>
                    <div class="form-group">
                      <label [for]="'metric-fraction-' + metricId">Data Fraction <span class="fraction-value">{{ fractionPercent(metricId) }}%</span></label>
                      <div class="range-control">
                        <span class="range-end-label">0%</span>
                        <input [id]="'metric-fraction-' + metricId" type="range" min="0.01" max="1" step="0.01" [value]="dataFraction(metricId)" (input)="updateDataFraction(metricId, $event)"/>
                        <span class="range-end-label">100%</span>
                      </div>
                      <p class="param-description">Percentage of the validation dataset to use for this evaluation.{{ metric.id === 'bleu_score' ? ' Lower values recommended for faster BLEU calculation.' : '' }}</p>
                    </div>
                  </div>
                </div>
              }
            }

            @if (isLastStep()) {
              <div class="summary-container">
                <h3 class="wizard-step-title-text">Review &amp; Confirm</h3>
                <div class="summary-card"><div class="summary-row"><span class="summary-label">Checkpoint</span><span class="summary-checkpoint-value">{{ checkpointName }}</span></div></div>
                <h4 class="config-section-title">Selected Metrics Configuration</h4>
                @for (metricId of orderedSelectedMetrics(); track metricId) {
                  @if (metricById(metricId); as metric) {
                    <div class="summary-card compact">
                      <div class="summary-metric-header"><ng-icon [name]="metric.icon"/><span>{{ metric.title }}</span></div>
                      <div class="config-summary-list"><div class="config-item"><span class="item-label">Data Fraction</span><span>{{ fractionPercent(metricId) }}%</span></div></div>
                    </div>
                  }
                }
              </div>
            }
          </div>

          <footer class="wizard-footer">
            <button type="button" class="btn-wizard btn-wizard-secondary" (click)="back()" [disabled]="isFirstStep()"><ng-icon name="lucideArrowLeft"/>Back</button>
            @if (isLastStep()) {
              <button type="button" class="btn-wizard btn-wizard-primary" (click)="confirm()" [disabled]="!orderedSelectedMetrics().length"><ng-icon name="lucideCheck"/>Start Evaluation</button>
            } @else {
              <button type="button" class="btn-wizard btn-wizard-primary" (click)="next()" [disabled]="isFirstStep() && !orderedSelectedMetrics().length">Next<ng-icon name="lucideArrowRight"/></button>
            }
          </footer>
        </section>
      </div>
    }
  `,
  styleUrl: '../styles/EvaluationWizard.css',
})
export class EvaluationWizardComponent implements OnChanges {
  @Input() open = false;
  @Input() checkpointName = '';
  @Output() readonly closed = new EventEmitter<void>();
  @Output() readonly confirmed = new EventEmitter<EvaluationWizardConfirmPayload>();

  readonly metricCatalog = METRICS_CATALOG;
  readonly selectedMetrics = signal<string[]>([]);
  readonly metricConfigs = signal<Record<string, EvaluationMetricConfig>>({});
  readonly currentStepIndex = signal(0);
  readonly orderedSelectedMetrics = computed(() => METRICS_CATALOG.filter((metric) => this.selectedMetrics().includes(metric.id)).map((metric) => metric.id));
  readonly totalSteps = computed(() => this.orderedSelectedMetrics().length + 2);
  readonly stepIndicators = computed(() => Array.from({ length: this.totalSteps() }));
  readonly isFirstStep = computed(() => this.currentStepIndex() === 0);
  readonly isLastStep = computed(() => this.currentStepIndex() === this.totalSteps() - 1);
  readonly currentMetricId = computed(() => {
    if (this.isFirstStep() || this.isLastStep()) return null;
    return this.orderedSelectedMetrics()[this.currentStepIndex() - 1] ?? null;
  });

  ngOnChanges(changes: SimpleChanges) {
    if (changes['open']?.currentValue && !changes['open']?.previousValue) this.reset();
  }

  isSelected(metricId: string) { return this.selectedMetrics().includes(metricId); }

  metricById(metricId: string) { return METRICS_CATALOG.find((metric) => metric.id === metricId); }

  toggleMetric(metricId: string) {
    if (this.isSelected(metricId)) {
      this.selectedMetrics.update((metrics) => metrics.filter((metric) => metric !== metricId));
      return;
    }
    const metric = this.metricById(metricId);
    if (!metric) return;
    this.metricConfigs.update((configs) => ({ ...configs, [metricId]: configs[metricId] ?? { dataFraction: metric.defaultFraction } }));
    this.selectedMetrics.update((metrics) => [...metrics, metricId]);
  }

  dataFraction(metricId: string) { return this.metricConfigs()[metricId]?.dataFraction ?? this.metricById(metricId)?.defaultFraction ?? 1; }

  fractionPercent(metricId: string) { return Math.round(this.dataFraction(metricId) * 100); }

  updateDataFraction(metricId: string, event: Event) {
    const value = Number((event.target as HTMLInputElement).value);
    if (!Number.isFinite(value)) return;
    this.metricConfigs.update((configs) => ({ ...configs, [metricId]: { dataFraction: value } }));
  }

  next() { if (!this.isLastStep()) this.currentStepIndex.update((step) => step + 1); }

  back() { if (!this.isFirstStep()) this.currentStepIndex.update((step) => step - 1); }

  confirm() {
    const metrics = this.orderedSelectedMetrics();
    if (!metrics.length) return;
    const metricConfigs = Object.fromEntries(metrics.map((metricId) => [metricId, { dataFraction: this.dataFraction(metricId) }]));
    this.confirmed.emit({ metrics, metricConfigs });
  }

  private reset() {
    this.currentStepIndex.set(0);
    this.selectedMetrics.set([]);
    this.metricConfigs.set({});
  }
}
