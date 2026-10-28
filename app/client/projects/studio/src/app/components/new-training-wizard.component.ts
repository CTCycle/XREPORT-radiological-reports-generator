import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormGroup, ReactiveFormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideActivity, lucideCheck, lucideChevronLeft, lucideChevronRight, lucideCpu, lucideInfo, lucideMonitor, lucidePlay, lucideSettings, lucideX } from '@ng-icons/lucide';
import { formatParameters } from '../common/model-architecture';
import type { ArchitecturePreset } from '../types/modelArchitectureApi';
import { ModalFocusDirective } from '@xreport/shared/app/components/modal-focus.directive';

@Component({
  standalone: true,
  selector: 'app-new-training-wizard',
  imports: [CommonModule, ReactiveFormsModule, RouterLink, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideActivity, lucideCheck, lucideChevronLeft, lucideChevronRight, lucideCpu, lucideInfo, lucideMonitor, lucidePlay, lucideSettings, lucideX })],
  styleUrl: '../styles/TrainingPage.css',
  template: `
    @if (open) {
      <div class="training-modal-backdrop" role="presentation">
        <section class="modal training-wizard-modal" role="dialog" aria-modal="true" aria-labelledby="new-training-title" appModalFocus (modalEscape)="closed.emit()">
          <div class="modal-header"><div><h2 id="new-training-title">New Training Wizard</h2><p class="wizard-subtitle">Dataset: <strong>{{ datasetLabel || 'No dataset selected' }}</strong></p></div><button type="button" class="btn-icon-small" aria-label="Close" (click)="closed.emit()"><ng-icon name="lucideX"/></button></div>
          <nav class="training-wizard-steps" aria-label="Training configuration steps">
            @for (step of steps; track step; let index = $index) {
              <button type="button" class="training-wizard-step" [class.active]="currentStep() === index" [class.done]="currentStep() > index" [attr.aria-current]="currentStep() === index ? 'step' : null" (click)="goToStep(index)"><span class="step-index">@if (currentStep() > index) { <ng-icon name="lucideCheck" size="14"/> } @else { {{ index + 1 }} }</span><span class="step-label">{{ step }}</span></button>
            }
          </nav>
          <div class="training-wizard-body" [formGroup]="form">
            @if (currentStep() === 0) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideCpu"/><span>Model Architecture</span></div>
                <label class="form-group"><span class="form-label">Architecture preset</span><select class="form-select" formControlName="architecturePreset">@for (preset of presets; track preset.key) { <option [value]="preset.key">{{ preset.name }}{{ preset.builtin ? '' : ' (custom)' }}</option> }</select></label>
                @if (selectedPreset(); as preset) {
                  <div class="wizard-preset-summary"><p>{{ preset.description || 'Custom architecture preset.' }}</p><div class="wizard-summary"><div><span>Parameters</span><strong>{{ formatParameters(preset.stats.total_parameters) }}</strong></div><div><span>Trainable</span><strong>{{ formatParameters(preset.stats.trainable_parameters) }}</strong></div><div><span>Encoder / decoder layers</span><strong>{{ preset.architecture.num_encoders }} / {{ preset.architecture.num_decoders }}</strong></div><div><span>Width / heads (enc / dec)</span><strong>{{ preset.architecture.embedding_dims }} / {{ preset.architecture.encoder_heads }} / {{ preset.architecture.decoder_heads }}</strong></div><div><span>Image encoder</span><strong>{{ preset.architecture.freeze_image_encoder ? 'Frozen BEiT' : 'Trainable BEiT' }}</strong></div></div></div>
                } @else { <div class="upload-status error">The selected preset no longer exists. Choose another preset.</div> }
                <a class="wizard-preset-link" routerLink="/model" (click)="closed.emit()">Edit or create presets on the Model page</a>
              </div>
            }
            @if (currentStep() === 1) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideSettings"/><span>Dataset Configuration</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useImgAugment"/><span class="checkbox-visual" aria-hidden="true"></span> Enable image augmentation</label></div>
              </div></div>
            }
            @if (currentStep() === 2) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideActivity"/><span>Training Parameters</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="form-group"><span class="form-label">Epochs</span><input class="form-input" type="number" min="1" formControlName="epochs"/></label><label class="form-group"><span class="form-label">Batch size</span><input class="form-input" type="number" min="1" formControlName="batchSize"/></label><label class="form-group"><span class="form-label">Gradient accumulation steps</span><input class="form-input" type="number" min="1" max="64" formControlName="gradientAccumulationSteps"/></label><label class="checkbox-label"><input type="checkbox" formControlName="saveCheckpoints"/><span class="checkbox-visual" aria-hidden="true"></span> Save checkpoints</label></div>
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useScheduler"/><span class="checkbox-visual" aria-hidden="true"></span> Use learning-rate scheduler</label><label class="form-group"><span class="form-label">Target learning rate</span><input class="form-input" type="number" min="0" step="0.0001" formControlName="targetLR" [disabled]="!form.get('useScheduler')?.value"/></label><label class="form-group"><span class="form-label">Warmup steps</span><input class="form-input" type="number" min="0" formControlName="warmupSteps" [disabled]="!form.get('useScheduler')?.value"/></label></div>
              </div></div>
            }
            @if (currentStep() === 3) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideMonitor"/><span>Device Configuration</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useGpu"/><span class="checkbox-visual" aria-hidden="true"></span> Enable GPU</label><label class="form-group"><span class="form-label">GPU device</span><select class="form-select" formControlName="gpuId" [disabled]="!form.get('useGpu')?.value"><option [ngValue]="0">0</option><option [ngValue]="1">1</option><option [ngValue]="2">2</option><option [ngValue]="3">3</option></select></label><label class="checkbox-label"><input type="checkbox" formControlName="pinMemory"/><span class="checkbox-visual" aria-hidden="true"></span> Pin memory</label><label class="checkbox-label"><input type="checkbox" formControlName="useMixedPrecision"/><span class="checkbox-visual" aria-hidden="true"></span> Mixed precision</label><label class="checkbox-label"><input type="checkbox" formControlName="realTimePlot"/><span class="checkbox-visual" aria-hidden="true"></span> Plot training metrics</label></div>
                <div class="wizard-col"><label class="form-group"><span class="form-label">Dataloader workers</span><input class="form-input" type="number" min="0" formControlName="dataloaderWorkers"/></label><label class="form-group"><span class="form-label">Prefetch factor</span><input class="form-input" type="number" min="1" formControlName="prefetchFactor" [disabled]="!form.get('dataloaderWorkers')?.value"/></label><label class="checkbox-label"><input type="checkbox" formControlName="persistentWorkers" [disabled]="!form.get('dataloaderWorkers')?.value"/><span class="checkbox-visual" aria-hidden="true"></span> Persistent workers</label></div>
              </div></div>
            }
            @if (currentStep() === 4) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideInfo"/><span>Training Summary</span></div><div class="wizard-summary-content"><div class="wizard-summary"><div class="summary-section"><h5>Model</h5><div class="summary-item"><span>Preset</span><span>{{ selectedPreset()?.name ?? 'Missing' }}</span></div><div class="summary-item"><span>Parameters</span><span>{{ selectedPreset() ? formatParameters(selectedPreset()!.stats.total_parameters) : 'N/A' }}</span></div><div class="summary-item"><span>Layers</span><span>{{ selectedPreset() ? selectedPreset()!.architecture.num_encoders + ' + ' + selectedPreset()!.architecture.num_decoders : 'N/A' }}</span></div></div><div class="summary-section"><h5>Training</h5><div class="summary-item"><span>Epochs</span><span>{{ form.get('epochs')?.value }}</span></div><div class="summary-item"><span>Batch size</span><span>{{ form.get('batchSize')?.value }}</span></div><div class="summary-item"><span>Gradient accumulation</span><span>{{ form.get('gradientAccumulationSteps')?.value }}</span></div><div class="summary-item"><span>Scheduler</span><span>{{ form.get('useScheduler')?.value ? 'Yes' : 'No' }}</span></div><div class="summary-item"><span>Checkpoints</span><span>{{ form.get('saveCheckpoints')?.value ? 'Yes' : 'No' }}</span></div></div><div class="summary-section"><h5>Runtime</h5><div class="summary-item"><span>Device</span><span>{{ form.get('useGpu')?.value ? 'GPU ' + form.get('gpuId')?.value : 'CPU' }}</span></div><div class="summary-item"><span>Workers</span><span>{{ form.get('dataloaderWorkers')?.value }}</span></div><div class="summary-item"><span>Mixed precision</span><span>{{ form.get('useMixedPrecision')?.value ? 'Yes' : 'No' }}</span></div></div></div></div></div>
            }
          </div>
          @if (error) { <div class="upload-status error">{{ error }}</div> }
          <div class="modal-footer training-wizard-actions"><button type="button" class="btn btn-secondary wizard-footer-spacer" (click)="closed.emit()">Cancel</button>@if (currentStep() > 0) { <button type="button" class="btn btn-secondary" (click)="previousStep()"><ng-icon name="lucideChevronLeft"/>Back</button> }@if (currentStep() < steps.length - 1) { <button type="button" class="btn btn-primary" (click)="nextStep()">Next<ng-icon name="lucideChevronRight"/></button> }@else { <button type="button" class="btn btn-primary" (click)="submitted.emit()" [disabled]="isLoading || form.invalid || !selectedPreset()"><ng-icon [name]="isLoading ? 'lucideActivity' : 'lucidePlay'"/>{{ isLoading ? 'Starting…' : 'Start training' }}</button> }</div>
        </section>
      </div>
    }
  `,
})
export class NewTrainingWizardComponent {
  private openState = false;
  @Input()
  get open() { return this.openState; }
  set open(value: boolean) { if (value && !this.openState) this.currentStep.set(0); this.openState = value; }
  @Input() datasetLabel = '';
  @Input() form!: FormGroup;
  @Input() isLoading = false;
  @Input() error: string | null = null;
  @Input() presets: ArchitecturePreset[] = [];
  @Output() readonly closed = new EventEmitter<void>();
  @Output() readonly submitted = new EventEmitter<void>();
  readonly steps = ['Model', 'Dataset', 'Training', 'Device', 'Summary'];
  readonly currentStep = signal(0);
  readonly formatParameters = formatParameters;
  selectedPreset(): ArchitecturePreset | null { const key = this.form.get('architecturePreset')?.value; return this.presets.find((preset) => preset.key === key) ?? null; }
  nextStep() { this.currentStep.update((step) => Math.min(step + 1, this.steps.length - 1)); }
  previousStep() { this.currentStep.update((step) => Math.max(step - 1, 0)); }
  goToStep(step: number) { this.currentStep.set(Math.max(0, Math.min(step, this.steps.length - 1))); }
}
