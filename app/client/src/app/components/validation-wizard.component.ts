import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges, inject, signal } from '@angular/core';
import { NonNullableFormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideX } from '@ng-icons/lucide';
import type { DatasetInfo } from '../types/trainingApi';
import type { ValidationMetric, ValidationWizardConfirmPayload } from '../types/validationWizard';
import { ModalFocusDirective } from './modal-focus.directive';

const METRICS: { id: ValidationMetric; name: string; description: string }[] = [
  { id: 'pixels_distribution', name: 'Pixel intensity histogram', description: 'Visualize intensity spread across the dataset to spot exposure issues.' },
  { id: 'text_statistics', name: 'Text statistics', description: 'Summaries of word counts, vocabulary size, and report length distribution.' },
  { id: 'image_statistics', name: 'Image statistics', description: 'Per-image dimensions, mean/std values, and noise indicators.' },
];

@Component({
  standalone: true,
  selector: 'app-validation-wizard',
  imports: [CommonModule, ReactiveFormsModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideX })],
  template: `
    @if (open) {
      <div class="modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="wizard-modal" role="dialog" aria-modal="true" aria-labelledby="validation-wizard-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <div class="wizard-header"><div><h3 id="validation-wizard-title">Validation Wizard</h3><p class="wizard-subtitle">Dataset: <strong>{{ row?.name || 'Select a dataset' }}</strong></p></div><button type="button" class="wizard-close" aria-label="Close validation wizard" (click)="closed.emit()"><ng-icon name="lucideX"/></button></div>
          <form [formGroup]="form" (ngSubmit)="confirm()">
            <div class="wizard-config-bar"><button type="button" class="config-option" [attr.aria-pressed]="form.controls.validateFullDataset.value" (click)="toggleFullDataset()"><span class="toggle-switch" [class.checked]="form.controls.validateFullDataset.value"><span class="toggle-slider"></span></span><span class="toggle-label">Validate full dataset</span></button><div class="config-input-group" [class.disabled]="form.controls.validateFullDataset.value"><label for="validation-wizard-fraction">Fraction:</label><input id="validation-wizard-fraction" type="number" min="0.01" max="1" step="0.01" formControlName="validationFraction"/></div></div>
            <div class="wizard-body"><div class="wizard-page"><div class="wizard-step-title">Metrics selection</div><div class="wizard-metrics-grid">@for (metric of metricOptions; track metric.id) { <button type="button" class="wizard-metric" [class.selected]="isSelected(metric.id)" [attr.aria-pressed]="isSelected(metric.id)" (click)="toggleMetric(metric.id)"><div class="wizard-metric-title">{{ metric.name }}</div><div class="wizard-metric-desc">{{ metric.description }}</div><div class="wizard-metric-state">{{ isSelected(metric.id) ? 'Selected' : 'Select' }}</div></button> }</div></div></div>
            <div class="wizard-footer"><div class="wizard-footer-actions"><button type="button" class="btn btn-secondary" (click)="closed.emit()">Cancel</button><button type="submit" class="btn btn-primary" [disabled]="!selectedMetrics().length || form.invalid">Confirm</button></div></div>
          </form>
        </section>
      </div>
    }
  `,
  styleUrls: ['../styles/ValidationWizard.css'],
})
export class ValidationWizardComponent implements OnChanges {
  @Input() open = false;
  @Input() row: DatasetInfo | null = null;
  @Input() initialSelected: ValidationMetric[] = [];
  @Output() readonly closed = new EventEmitter<void>();
  @Output() readonly confirmed = new EventEmitter<ValidationWizardConfirmPayload>();
  readonly metricOptions = METRICS;
  readonly selectedMetrics = signal<ValidationMetric[]>([]);
  readonly form = inject(NonNullableFormBuilder).group({
    validateFullDataset: true,
    validationFraction: [0.5, [Validators.required, Validators.min(0.01), Validators.max(1)]],
  });

  ngOnChanges(changes: SimpleChanges) {
    if (changes['open']?.currentValue === true) {
      this.selectedMetrics.set([...(this.initialSelected ?? [])]);
      this.form.reset({ validateFullDataset: true, validationFraction: 0.5 });
    }
  }

  toggleFullDataset() { this.form.controls.validateFullDataset.setValue(!this.form.controls.validateFullDataset.value); }
  isSelected(metric: ValidationMetric) { return this.selectedMetrics().includes(metric); }
  toggleMetric(metric: ValidationMetric) { this.selectedMetrics.update((current) => current.includes(metric) ? current.filter((item) => item !== metric) : [...current, metric]); }
  confirm() {
    if (this.form.invalid || !this.selectedMetrics().length) return;
    const values = this.form.getRawValue();
    const sampleFraction = values.validateFullDataset ? 1 : Math.min(Math.max(values.validationFraction, 0.01), 1);
    this.confirmed.emit({ metrics: this.selectedMetrics(), row: this.row, sampleFraction });
  }
}
