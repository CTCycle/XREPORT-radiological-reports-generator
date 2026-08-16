import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';
import { FormGroup, ReactiveFormsModule } from '@angular/forms';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideActivity, lucideChevronLeft, lucideChevronRight, lucideCpu, lucideInfo, lucideMonitor, lucidePlay, lucideSettings, lucideX } from '@ng-icons/lucide';
import { ModalFocusDirective } from './modal-focus.directive';

@Component({
  standalone: true,
  selector: 'app-new-training-wizard',
  imports: [CommonModule, ReactiveFormsModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideActivity, lucideChevronLeft, lucideChevronRight, lucideCpu, lucideInfo, lucideMonitor, lucidePlay, lucideSettings, lucideX })],
  template: `
    @if (open) {
      <div class="modal-backdrop" role="presentation">
        <section class="modal training-wizard-modal" role="dialog" aria-modal="true" aria-labelledby="new-training-title" appModalFocus (modalEscape)="closed.emit()">
          <div class="modal-header"><h2 id="new-training-title">New Training Wizard</h2><button type="button" class="btn-icon-small" aria-label="Close" (click)="closed.emit()"><ng-icon name="lucideX"/></button></div>
          <p>Dataset: <strong>{{ datasetLabel || 'No dataset selected' }}</strong></p>
          <nav class="training-wizard-steps" aria-label="Training configuration steps">
            @for (step of steps; track step; let index = $index) {
              <button type="button" class="training-wizard-step" [class.active]="currentStep() === index" [attr.aria-current]="currentStep() === index ? 'step' : null" (click)="goToStep(index)"><span>{{ index + 1 }}</span>{{ step }}</button>
            }
          </nav>
          <div class="training-wizard-body" [formGroup]="form">
            @if (currentStep() === 0) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideCpu"/><span>Model Architecture</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="form-group"><span class="form-label">Encoders</span><input class="form-input" type="number" min="1" formControlName="numEncoders"/></label><label class="form-group"><span class="form-label">Decoders</span><input class="form-input" type="number" min="1" formControlName="numDecoders"/></label><label class="form-group"><span class="form-label">Embedding dimensions</span><input class="form-input" type="number" min="1" step="8" formControlName="embeddingDims"/></label><label class="form-group"><span class="form-label">Attention heads</span><input class="form-input" type="number" min="1" formControlName="attnHeads"/></label></div>
                <div class="wizard-col"><label class="form-group"><span class="form-label">Temperature</span><input class="form-input" type="number" min="0" step="0.05" formControlName="trainTemp"/></label><label class="checkbox-label"><input type="checkbox" formControlName="freezeImgEncoder"/> Freeze image encoder</label></div>
              </div></div>
            }
            @if (currentStep() === 1) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideSettings"/><span>Dataset Configuration</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useImgAugment"/> Enable image augmentation</label></div>
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="shuffleWithBuffer"/> Shuffle with buffer</label><label class="form-group"><span class="form-label">Shuffle buffer size</span><input class="form-input" type="number" min="1" formControlName="shuffleBufferSize" [disabled]="!form.get('shuffleWithBuffer')?.value"/></label></div>
              </div></div>
            }
            @if (currentStep() === 2) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideActivity"/><span>Training Parameters</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="form-group"><span class="form-label">Epochs</span><input class="form-input" type="number" min="1" formControlName="epochs"/></label><label class="form-group"><span class="form-label">Batch size</span><input class="form-input" type="number" min="1" formControlName="batchSize"/></label><label class="checkbox-label"><input type="checkbox" formControlName="saveCheckpoints"/> Save checkpoints</label></div>
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useScheduler"/> Use learning-rate scheduler</label><label class="form-group"><span class="form-label">Target learning rate</span><input class="form-input" type="number" min="0" step="0.0001" formControlName="targetLR" [disabled]="!form.get('useScheduler')?.value"/></label><label class="form-group"><span class="form-label">Warmup steps</span><input class="form-input" type="number" min="0" formControlName="warmupSteps" [disabled]="!form.get('useScheduler')?.value"/></label></div>
              </div></div>
            }
            @if (currentStep() === 3) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideMonitor"/><span>Device Configuration</span></div><div class="wizard-2col-panel">
                <div class="wizard-col"><label class="checkbox-label"><input type="checkbox" formControlName="useGpu"/> Enable GPU</label><label class="form-group"><span class="form-label">GPU device</span><select class="form-select" formControlName="gpuId" [disabled]="!form.get('useGpu')?.value"><option [ngValue]="0">0</option><option [ngValue]="1">1</option><option [ngValue]="2">2</option><option [ngValue]="3">3</option></select></label><label class="checkbox-label"><input type="checkbox" formControlName="pinMemory"/> Pin memory</label><label class="checkbox-label"><input type="checkbox" formControlName="useMixedPrecision"/> Mixed precision</label><label class="checkbox-label"><input type="checkbox" formControlName="realTimePlot"/> Plot training metrics</label></div>
                <div class="wizard-col"><label class="form-group"><span class="form-label">Dataloader workers</span><input class="form-input" type="number" min="0" formControlName="dataloaderWorkers"/></label><label class="form-group"><span class="form-label">Prefetch factor</span><input class="form-input" type="number" min="1" formControlName="prefetchFactor" [disabled]="!form.get('dataloaderWorkers')?.value"/></label><label class="checkbox-label"><input type="checkbox" formControlName="persistentWorkers" [disabled]="!form.get('dataloaderWorkers')?.value"/> Persistent workers</label><label class="checkbox-label"><input type="checkbox" formControlName="jitCompile"/> Enable torch compile</label><label class="form-group"><span class="form-label">Compile backend</span><select class="form-select" formControlName="jitBackend" [disabled]="!form.get('jitCompile')?.value"><option value="inductor">inductor</option><option value="eager">eager</option><option value="aot_eager">aot_eager</option><option value="nvprims_nvfuser">nvprims_nvfuser</option></select></label></div>
              </div></div>
            }
            @if (currentStep() === 4) {
              <div class="wizard-page"><div class="wizard-section-title"><ng-icon name="lucideInfo"/><span>Training Summary</span></div><div class="wizard-summary-content"><div class="wizard-summary"><div class="summary-section"><h5>Model</h5><div class="summary-item"><span>Encoders</span><span>{{ form.get('numEncoders')?.value }}</span></div><div class="summary-item"><span>Decoders</span><span>{{ form.get('numDecoders')?.value }}</span></div><div class="summary-item"><span>Embedding</span><span>{{ form.get('embeddingDims')?.value }}</span></div><div class="summary-item"><span>Heads</span><span>{{ form.get('attnHeads')?.value }}</span></div></div><div class="summary-section"><h5>Training</h5><div class="summary-item"><span>Epochs</span><span>{{ form.get('epochs')?.value }}</span></div><div class="summary-item"><span>Batch size</span><span>{{ form.get('batchSize')?.value }}</span></div><div class="summary-item"><span>Scheduler</span><span>{{ form.get('useScheduler')?.value ? 'Yes' : 'No' }}</span></div><div class="summary-item"><span>Checkpoints</span><span>{{ form.get('saveCheckpoints')?.value ? 'Yes' : 'No' }}</span></div></div><div class="summary-section"><h5>Runtime</h5><div class="summary-item"><span>Device</span><span>{{ form.get('useGpu')?.value ? 'GPU ' + form.get('gpuId')?.value : 'CPU' }}</span></div><div class="summary-item"><span>Workers</span><span>{{ form.get('dataloaderWorkers')?.value }}</span></div><div class="summary-item"><span>Mixed precision</span><span>{{ form.get('useMixedPrecision')?.value ? 'Yes' : 'No' }}</span></div><div class="summary-item"><span>Torch compile</span><span>{{ form.get('jitCompile')?.value ? form.get('jitBackend')?.value : 'No' }}</span></div></div></div></div></div>
            }
          </div>
          @if (error) { <div class="upload-status error">{{ error }}</div> }
          <div class="modal-footer training-wizard-actions"><button type="button" class="btn btn-secondary" (click)="closed.emit()">Cancel</button>@if (currentStep() > 0) { <button type="button" class="btn btn-secondary" (click)="previousStep()"><ng-icon name="lucideChevronLeft"/>Back</button> }@if (currentStep() < steps.length - 1) { <button type="button" class="btn btn-primary" (click)="nextStep()">Next<ng-icon name="lucideChevronRight"/></button> }@else { <button type="button" class="btn btn-primary" (click)="submitted.emit()" [disabled]="isLoading || form.invalid"><ng-icon [name]="isLoading ? 'lucideActivity' : 'lucidePlay'"/>{{ isLoading ? 'Starting…' : 'Start training' }}</button> }</div>
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
  @Output() readonly closed = new EventEmitter<void>();
  @Output() readonly submitted = new EventEmitter<void>();
  readonly steps = ['Model', 'Dataset', 'Training', 'Device', 'Summary'];
  readonly currentStep = signal(0);
  nextStep() { this.currentStep.update((step) => Math.min(step + 1, this.steps.length - 1)); }
  previousStep() { this.currentStep.update((step) => Math.max(step - 1, 0)); }
  goToStep(step: number) { this.currentStep.set(Math.max(0, Math.min(step, this.steps.length - 1))); }
}
