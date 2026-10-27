import { CommonModule } from '@angular/common';
import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideCircleCheck, lucideCopyPlus, lucideDownload, lucideLock, lucideNetwork, lucidePlus, lucideSave, lucideTrash2, lucideTriangleAlert, lucideUndo2, lucideUpload, lucideX } from '@ng-icons/lucide';
import { buildArchitectureFile, formatBytes, formatParameters, parseArchitectureFile } from '../common/model-architecture';
import { ArchitectureDiagramComponent } from '../components/architecture-diagram.component';
import { HelpPopoverComponent } from '../components/help-popover.component';
import { ModalFocusDirective } from '../components/modal-focus.directive';
import { DatasetApiService } from '../services/dataset-api.service';
import { ModelArchitectureApiService } from '../services/model-architecture-api.service';
import { NotificationService } from '../services/notification.service';
import { TrainingApiService } from '../services/training-api.service';
import type { ArchitectureGraph, ArchitecturePreset, ArchitecturePreviewResponse, ArchitectureStats, EncoderOption, ModelArchitecture } from '../types/modelArchitectureApi';

type ArchitectureField = keyof ModelArchitecture;

interface FieldSpec {
  field: ArchitectureField;
  label: string;
  help: string;
  kind: 'range' | 'toggle' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  advanced?: boolean;
  /** Changing the value keeps every weight shape (see WEIGHTS_COMPATIBLE_FIELDS). */
  weightsCompatible?: boolean;
}

const FIELDS: Record<ArchitectureField, FieldSpec> = {
  image_encoder: { field: 'image_encoder', label: 'Image encoder', kind: 'select', help: 'Pretrained vision backbone that turns the radiograph into patch features. Only encoders that XREPORT Inference can load are offered.' },
  freeze_image_encoder: { field: 'freeze_image_encoder', label: 'Freeze image encoder', kind: 'toggle', weightsCompatible: true, help: 'Frozen keeps the pretrained BEiT weights fixed and trains only the projection and transformer. Unfreezing fine-tunes ~86M more parameters and needs much more GPU memory.' },
  embedding_dims: { field: 'embedding_dims', label: 'Model width', kind: 'range', min: 64, max: 1024, step: 8, help: 'Width of every token vector: the image projection, the token embeddings and all encoder and decoder blocks share it.' },
  num_encoders: { field: 'num_encoders', label: 'Encoder layers', kind: 'range', min: 1, max: 12, step: 1, help: 'Transformer blocks that refine the projected image features with self-attention.' },
  num_decoders: { field: 'num_decoders', label: 'Decoder layers', kind: 'range', min: 1, max: 12, step: 1, help: 'Transformer blocks that generate the report: masked self-attention over report tokens, then cross-attention to the image features.' },
  attention_heads: { field: 'attention_heads', label: 'Attention heads', kind: 'range', min: 1, max: 16, step: 1, help: 'Parallel attention heads in every encoder and decoder block.' },
  head_dim_mode: { field: 'head_dim_mode', label: 'Head size', kind: 'select', advanced: true, options: [{ value: 'full', label: 'Full width (original XREPORT)' }, { value: 'split', label: 'Split width / heads' }], help: 'Full width gives every head the whole model width (original XREPORT, large). Split divides the width between heads, as in standard transformers; the width must then be divisible by the head count.' },
  ffn_units: { field: 'ffn_units', label: 'Feed-forward units', kind: 'range', min: 64, max: 4096, step: 64, advanced: true, help: 'Hidden size of the two-layer feed-forward network in every block. Original XREPORT uses the model width; standard transformers use about 4x.' },
  dropout: { field: 'dropout', label: 'Dropout', kind: 'range', min: 0, max: 0.5, step: 0.05, advanced: true, weightsCompatible: true, help: 'Fraction of activations dropped during training for regularisation. Applied to the feed-forward networks and the decoder attention.' },
  classifier_units: { field: 'classifier_units', label: 'Classifier units', kind: 'range', min: 64, max: 4096, step: 64, advanced: true, help: 'Hidden layer of the token classifier between the decoder output and the vocabulary.' },
  temperature: { field: 'temperature', label: 'Softmax temperature', kind: 'range', min: 0.1, max: 2, step: 0.05, weightsCompatible: true, help: 'Divides the classifier logits before the softmax. Values below 1 sharpen the token distribution, above 1 flatten it.' },
};

// Inspector fields per diagram node. Heads, head size, feed-forward and dropout
// are shared by encoder and decoder blocks.
const NODE_FIELDS: Record<string, ArchitectureField[]> = {
  image_input: [],
  image_encoder: ['image_encoder', 'freeze_image_encoder'],
  projection: ['embedding_dims'],
  encoder_stack: ['num_encoders', 'attention_heads', 'head_dim_mode', 'ffn_units', 'dropout'],
  token_input: [],
  embedding: ['embedding_dims'],
  decoder_stack: ['num_decoders', 'attention_heads', 'head_dim_mode', 'ffn_units', 'dropout'],
  classifier: ['temperature', 'classifier_units'],
};
const NODE_NOTES: Record<string, string> = {
  image_input: 'Radiographs are resized to 224 x 224 RGB and normalised in code; input preprocessing is not configurable.',
  projection: 'A dense layer maps the 768-wide BEiT features to the model width. It is always trained from scratch.',
  token_input: 'The report length comes from the processed dataset (max report size), not from the architecture.',
  embedding: 'Learned token and position embeddings. The vocabulary and the number of positions come from the dataset tokenizer.',
  encoder_stack: 'Heads, head size, feed-forward and dropout are shared by encoder and decoder blocks.',
  decoder_stack: 'Heads, head size, feed-forward and dropout are shared by encoder and decoder blocks.',
};

interface Source { key: string | null; name: string; description: string; builtin: boolean; }
interface DetailsDialog { mode: 'save-as' | 'details'; name: string; description: string; error: string | null; }

const PREVIEW_DEBOUNCE_MS = 150;

@Component({
  standalone: true,
  selector: 'app-model-page',
  imports: [CommonModule, NgIcon, ArchitectureDiagramComponent, HelpPopoverComponent, ModalFocusDirective],
  providers: [provideIcons({ lucideCircleCheck, lucideCopyPlus, lucideDownload, lucideLock, lucideNetwork, lucidePlus, lucideSave, lucideTrash2, lucideTriangleAlert, lucideUndo2, lucideUpload, lucideX })],
  styleUrls: ['../styles/TrainingPage.css', '../styles/ModelPage.css'],
  template: `
    <main class="model-page page-container">
      <div class="header">
        <h1>Model Architecture</h1>
        <p>Configure, compare and save XREPORT transformer architectures for training.</p>
      </div>
      @if (pageError()) { <div class="model-banner error" role="alert"><ng-icon name="lucideTriangleAlert"/><span>{{ pageError() }}</span><button type="button" class="icon-button" aria-label="Dismiss" (click)="pageError.set(null)"><ng-icon name="lucideX"/></button></div> }
      <div class="model-layout">
        <aside class="model-panel model-library" aria-label="Architecture presets">
          <div class="model-panel-header">
            <h2>Presets</h2>
            <div class="model-panel-actions">
              <button type="button" class="icon-button" title="New architecture" aria-label="New architecture" (click)="newDraft()"><ng-icon name="lucidePlus"/></button>
              <button type="button" class="icon-button" title="Import JSON" aria-label="Import architecture JSON" (click)="importInput.click()"><ng-icon name="lucideUpload"/></button>
              <input #importInput type="file" accept="application/json,.json" hidden (change)="importFile(importInput)" />
            </div>
          </div>
          @for (group of presetGroups(); track group.title) {
            <h3 class="model-library-group">{{ group.title }}</h3>
            @if (!group.presets.length) { <p class="model-library-empty">No saved presets yet. Duplicate a built-in preset or save the current draft.</p> }
            <ul class="model-preset-list">
              @for (preset of group.presets; track preset.key) {
                <li class="model-preset" [class.active]="source().key === preset.key">
                  <button type="button" class="model-preset-main" [attr.aria-current]="source().key === preset.key ? 'true' : null" (click)="loadPreset(preset)">
                    <span class="model-preset-name">@if (preset.builtin) { <ng-icon name="lucideLock" size="12" aria-label="Built-in"/> }{{ preset.name }}</span>
                    <span class="model-preset-meta">{{ formatParameters(preset.stats.total_parameters) }} params · {{ preset.architecture.num_encoders }}+{{ preset.architecture.num_decoders }} layers · width {{ preset.architecture.embedding_dims }}</span>
                    <span class="model-preset-meta">{{ formatParameters(preset.stats.trainable_parameters) }} trainable</span>
                  </button>
                  <div class="model-preset-actions">
                    <button type="button" class="icon-button" title="Duplicate" [attr.aria-label]="'Duplicate ' + preset.name" (click)="duplicatePreset(preset)"><ng-icon name="lucideCopyPlus"/></button>
                    <button type="button" class="icon-button" title="Export JSON" [attr.aria-label]="'Export ' + preset.name" (click)="exportPreset(preset.name, preset.description, preset.architecture)"><ng-icon name="lucideDownload"/></button>
                    @if (!preset.builtin) { <button type="button" class="icon-button danger" title="Delete" [attr.aria-label]="'Delete ' + preset.name" (click)="deletePreset(preset)"><ng-icon name="lucideTrash2"/></button> }
                  </div>
                </li>
              }
            </ul>
          }
        </aside>

        <section class="model-panel model-canvas" aria-label="Architecture">
          <div class="model-canvas-toolbar">
            <div class="model-canvas-title">
              <h2>{{ source().name }}@if (dirty()) { <span class="model-dirty" title="Unsaved changes" aria-label="Unsaved changes"></span> }</h2>
              <span class="model-canvas-subtitle">{{ source().builtin ? 'Built-in preset, read-only' : source().key ? 'Custom preset' : 'Unsaved draft' }}</span>
            </div>
            <div class="model-canvas-actions">
              <button type="button" class="btn btn-secondary" [disabled]="!dirty()" (click)="revert()"><ng-icon name="lucideUndo2"/>Revert</button>
              @if (source().key && !source().builtin) { <button type="button" class="btn btn-secondary" (click)="openDialog('details')">Details</button> }
              <button type="button" class="btn btn-secondary" [disabled]="!valid()" (click)="openDialog('save-as')"><ng-icon name="lucideCopyPlus"/>Save as</button>
              <button type="button" class="btn btn-primary" [disabled]="!canSave()" [title]="source().builtin ? 'Built-in presets are read-only; use Save as' : ''" (click)="save()"><ng-icon name="lucideSave"/>Save</button>
            </div>
          </div>
          @if (stats(); as current) {
            <dl class="model-stats" [class.stale]="!valid()" aria-live="polite">
              <div><dt>Total parameters</dt><dd>{{ formatParameters(current.total_parameters) }}</dd></div>
              <div><dt>Trainable</dt><dd>{{ formatParameters(current.trainable_parameters) }}</dd></div>
              <div><dt>Weights (fp32)</dt><dd>{{ formatBytes(current.weights_bytes) }}</dd></div>
              <div><dt>Training memory <app-help-popover label="Explain training memory" title="Training memory estimate" body="Weights plus gradients and the two AdamW moment buffers of the trainable weights. Activations, which grow with batch size and report length, are not included." /></dt><dd>{{ formatBytes(current.training_memory_bytes) }}</dd></div>
              <div><dt>Transformer depth</dt><dd>{{ current.depth }} layers</dd></div>
            </dl>
          }
          <div class="model-sizing">
            <label class="model-inline-field"><span>Size against</span>
              <select class="form-select" (change)="datasetName.set(selectValue($event))" aria-label="Dataset used to size embeddings and classifier">
                <option value="" [selected]="!datasetName()">Reference ({{ referenceLabel }})</option>
                @for (dataset of datasets(); track dataset) { <option [value]="dataset" [selected]="dataset === datasetName()">{{ dataset }}</option> }
              </select>
            </label>
            <span class="model-sizing-note">Vocabulary {{ stats()?.vocabulary_size ?? '—' }} · report length {{ stats()?.sequence_length ?? '—' }}</span>
          </div>
          <div class="model-diagram" [class.stale]="!valid() || previewPending()">
            @if (graph(); as currentGraph) {
              <app-architecture-diagram [graph]="currentGraph" [selected]="selectedNode()" (nodeSelected)="selectedNode.set($event)" />
            } @else { <p class="model-diagram-empty">Loading architecture…</p> }
          </div>
        </section>

        <aside class="model-panel model-inspector" aria-label="Component settings">
          <div class="model-panel-header"><h2>{{ selectedLabel() }}</h2></div>
          @for (issue of generalIssues(); track issue) { <div class="model-banner error" role="alert"><ng-icon name="lucideTriangleAlert"/><span>{{ issue }}</span></div> }
          @if (nodeNote()) { <p class="model-note">{{ nodeNote() }}</p> }
          @if (selectedNode() === 'image_encoder' && selectedEncoder(); as encoder) {
            <dl class="model-facts">
              <div><dt>Repository</dt><dd class="mono">{{ encoder.repository }}</dd></div>
              <div><dt>Revision</dt><dd class="mono">{{ encoder.revision.slice(0, 12) }}</dd></div>
              <div><dt>Output</dt><dd>{{ encoder.output_tokens }} tokens × {{ encoder.feature_dims }}</dd></div>
              <div><dt>Parameters</dt><dd>{{ formatParameters(encoder.parameters) }}, pretrained</dd></div>
              <div><dt>XREPORT Inference</dt><dd>{{ encoder.inference_compatible ? 'Checkpoints can be exported and imported' : 'Not importable' }}</dd></div>
            </dl>
          }
          @for (spec of basicFields(); track spec.field) { <ng-container *ngTemplateOutlet="control; context: { $implicit: spec }" /> }
          @if (advancedFields().length) {
            <details class="model-advanced" [open]="showAdvanced()" (toggle)="showAdvanced.set($any($event.target).open)">
              <summary>Advanced</summary>
              @for (spec of advancedFields(); track spec.field) { <ng-container *ngTemplateOutlet="control; context: { $implicit: spec }" /> }
            </details>
          }
          @if (!basicFields().length && !advancedFields().length) { <p class="model-note muted">Nothing to configure for this component.</p> }

          <section class="model-compat" aria-label="Weight compatibility">
            <h3>Compatibility</h3>
            <label class="model-inline-field"><span>Compare with</span>
              <select class="form-select" (change)="reference.set(selectValue($event))">
                <option value="" [selected]="!reference()">Nothing</option>
                @for (preset of presets(); track preset.key) { <option [value]="preset.key" [selected]="preset.key === reference()">Preset: {{ preset.name }}</option> }
                @for (checkpoint of checkpoints(); track checkpoint) { <option [value]="'checkpoint:' + checkpoint" [selected]="'checkpoint:' + checkpoint === reference()">Checkpoint: {{ checkpoint }}</option> }
              </select>
            </label>
            @if (preview()?.compatibility; as compat) {
              <div class="model-compat-result" [attr.data-level]="compat.level">
                <ng-icon [name]="compat.level === 'retrain_required' ? 'lucideTriangleAlert' : 'lucideCircleCheck'"/>
                <div><strong>{{ compatibilityTitle(compat.level) }}</strong><p>{{ compat.message }}</p></div>
              </div>
              @if (compat.changes.length) {
                <ul class="model-compat-changes">
                  @for (change of compat.changes; track change.field) {
                    <li><span>{{ fieldLabel(change.field) }}</span><span class="mono">{{ change.before }} → {{ change.after }}</span><span class="model-impact" [attr.data-impact]="change.impact">{{ change.impact === 'retrain_required' ? 'Retrain' : 'Compatible' }}</span></li>
                  }
                </ul>
              }
            }
            @if (reference()?.startsWith('checkpoint:')) {
              <button type="button" class="btn btn-secondary model-compat-load" (click)="loadCheckpointArchitecture()">Load architecture from checkpoint</button>
            }
          </section>
        </aside>
      </div>

      <ng-template #control let-spec>
        <div class="model-control" [class.invalid]="fieldIssue(spec.field)">
          <div class="model-control-head">
            <label [attr.for]="'arch-' + spec.field">{{ spec.label }}</label>
            <app-help-popover [label]="'Explain ' + spec.label" [title]="spec.label" [body]="spec.help" />
            <span class="model-impact" [attr.data-impact]="spec.weightsCompatible ? 'weights_compatible' : 'retrain_required'" [title]="spec.weightsCompatible ? 'Weight shapes stay the same' : 'Changes weight shapes: the transformer trains from scratch'">{{ spec.weightsCompatible ? 'Weights-compatible' : 'Requires retraining' }}</span>
          </div>
          @switch (spec.kind) {
            @case ('range') {
              <div class="model-range">
                <input type="range" [min]="spec.min" [max]="spec.max" [step]="spec.step" [value]="value(spec.field)" (input)="update(spec.field, numberValue($event))" [attr.aria-label]="spec.label + ' slider'" />
                <input class="form-input" type="number" [id]="'arch-' + spec.field" [min]="spec.min" [max]="spec.max" [step]="spec.step" [value]="value(spec.field)" (input)="update(spec.field, numberValue($event))" />
              </div>
            }
            @case ('toggle') {
              <label class="checkbox-label"><input type="checkbox" [id]="'arch-' + spec.field" [checked]="value(spec.field)" (change)="update(spec.field, $any($event.target).checked)" /><span class="checkbox-visual" aria-hidden="true"></span>{{ value(spec.field) ? 'Frozen' : 'Trainable' }}</label>
            }
            @case ('select') {
              <select class="form-select" [id]="'arch-' + spec.field" (change)="update(spec.field, selectValue($event))">
                @for (option of optionsFor(spec); track option.value) { <option [value]="option.value" [selected]="option.value === value(spec.field)">{{ option.label }}</option> }
              </select>
            }
          }
          @if (fieldIssue(spec.field); as issue) { <p class="model-field-error" role="alert">{{ issue }}</p> }
        </div>
      </ng-template>

      @if (dialog(); as current) {
        <div class="model-modal-backdrop" role="presentation">
          <section class="model-modal" role="dialog" aria-modal="true" aria-labelledby="model-dialog-title" appModalFocus (modalEscape)="dialog.set(null)">
            <header><h2 id="model-dialog-title">{{ current.mode === 'save-as' ? 'Save as new preset' : 'Preset details' }}</h2><button type="button" class="icon-button" aria-label="Close" (click)="dialog.set(null)"><ng-icon name="lucideX"/></button></header>
            <label class="form-group"><span class="form-label">Name</span><input class="form-input" maxlength="80" [value]="current.name" (input)="dialog.set({ ...current, name: $any($event.target).value })" /></label>
            <label class="form-group"><span class="form-label">Description</span><textarea class="form-input" rows="3" maxlength="500" [value]="current.description" (input)="dialog.set({ ...current, description: $any($event.target).value })"></textarea></label>
            @if (current.error) { <div class="model-banner error" role="alert">{{ current.error }}</div> }
            <footer><button type="button" class="btn btn-secondary" (click)="dialog.set(null)">Cancel</button><button type="button" class="btn btn-primary" [disabled]="!current.name.trim() || busy()" (click)="submitDialog()">Save</button></footer>
          </section>
        </div>
      }
    </main>
  `,
})
export class ModelPage {
  private readonly api = inject(ModelArchitectureApiService);
  private readonly datasetApi = inject(DatasetApiService);
  private readonly trainingApi = inject(TrainingApiService);
  private readonly notifications = inject(NotificationService);
  readonly formatParameters = formatParameters;
  readonly formatBytes = formatBytes;
  readonly referenceLabel = 'distilbert vocabulary, 200 tokens';

  readonly presets = signal<ArchitecturePreset[]>([]);
  readonly encoders = signal<EncoderOption[]>([]);
  readonly datasets = signal<string[]>([]);
  readonly checkpoints = signal<string[]>([]);
  readonly source = signal<Source>({ key: null, name: 'Loading…', description: '', builtin: false });
  readonly baseline = signal<ModelArchitecture | null>(null);
  readonly draft = signal<ModelArchitecture | null>(null);
  readonly preview = signal<ArchitecturePreviewResponse | null>(null);
  readonly previewPending = signal(false);
  private readonly lastValid = signal<{ graph: ArchitectureGraph; stats: ArchitectureStats } | null>(null);
  readonly selectedNode = signal('encoder_stack');
  readonly datasetName = signal<string | null>(null);
  readonly reference = signal<string | null>(null);
  readonly showAdvanced = signal(false);
  readonly dialog = signal<DetailsDialog | null>(null);
  readonly busy = signal(false);
  readonly pageError = signal<string | null>(null);
  private previewRequest = 0;

  readonly dirty = computed(() => JSON.stringify(this.draft()) !== JSON.stringify(this.baseline()));
  readonly valid = computed(() => this.preview()?.valid ?? false);
  readonly canSave = computed(() => this.valid() && !this.busy() && (!this.source().key || (!this.source().builtin && this.dirty())));
  readonly graph = computed(() => this.preview()?.graph ?? this.lastValid()?.graph ?? null);
  readonly stats = computed(() => this.preview()?.stats ?? this.lastValid()?.stats ?? null);
  readonly presetGroups = computed(() => [
    { title: 'Built-in', presets: this.presets().filter((preset) => preset.builtin) },
    { title: 'Custom', presets: this.presets().filter((preset) => !preset.builtin) },
  ]);
  readonly selectedLabel = computed(() => this.graph()?.nodes.find((node) => node.id === this.selectedNode())?.label ?? 'Component');
  readonly nodeNote = computed(() => NODE_NOTES[this.selectedNode()] ?? '');
  readonly basicFields = computed(() => (NODE_FIELDS[this.selectedNode()] ?? []).map((field) => FIELDS[field]).filter((spec) => !spec.advanced));
  readonly advancedFields = computed(() => (NODE_FIELDS[this.selectedNode()] ?? []).map((field) => FIELDS[field]).filter((spec) => spec.advanced));
  readonly selectedEncoder = computed(() => this.encoders().find((encoder) => encoder.id === this.draft()?.image_encoder) ?? null);
  readonly generalIssues = computed(() => (this.preview()?.issues ?? []).filter((issue) => !issue.field).map((issue) => issue.message));

  constructor() {
    void this.initialize();
    let timer: ReturnType<typeof setTimeout> | undefined;
    effect(() => {
      const architecture = this.draft();
      const datasetName = this.datasetName();
      const reference = this.reference();
      if (!architecture) return;
      clearTimeout(timer);
      this.previewPending.set(true);
      timer = setTimeout(() => void this.refreshPreview(architecture, datasetName, reference), PREVIEW_DEBOUNCE_MS);
    });
    inject(DestroyRef).onDestroy(() => clearTimeout(timer));
  }

  private async initialize() {
    const [presets, encoders, datasets, checkpoints] = await Promise.all([this.api.getPresets(), this.api.getEncoders(), this.datasetApi.getProcessedNames(), this.trainingApi.getCheckpoints()]);
    if (encoders.result) this.encoders.set(encoders.result.encoders);
    if (datasets.result) this.datasets.set(datasets.result.datasets.map((dataset) => dataset.name));
    if (checkpoints.result) this.checkpoints.set(checkpoints.result.checkpoints.filter((checkpoint) => checkpoint.artifact_status === 'ready').map((checkpoint) => checkpoint.name));
    if (!presets.result) { this.pageError.set(presets.error ?? 'Failed to load architecture presets'); return; }
    this.presets.set(presets.result.presets);
    const initial = presets.result.presets[0];
    if (initial) this.applyPreset(initial);
  }

  /** Only the response of the latest request is applied. */
  private async refreshPreview(architecture: ModelArchitecture, datasetName: string | null, reference: string | null) {
    const request = ++this.previewRequest;
    const response = await this.api.preview({ architecture, dataset_name: datasetName, reference });
    if (request !== this.previewRequest) return;
    this.previewPending.set(false);
    if (!response.result) { this.pageError.set(response.error ?? 'Failed to preview the architecture'); return; }
    this.preview.set(response.result);
    if (response.result.graph && response.result.stats) this.lastValid.set({ graph: response.result.graph, stats: response.result.stats });
  }

  private async reloadPresets(selectKey?: string) {
    const result = await this.api.getPresets();
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to load architecture presets'); return; }
    this.presets.set(result.result.presets);
    const selected = selectKey ? result.result.presets.find((preset) => preset.key === selectKey) : null;
    if (selected) this.applyPreset(selected);
  }

  private applyPreset(preset: ArchitecturePreset) {
    const architecture = { ...preset.architecture } as ModelArchitecture;
    this.source.set({ key: preset.key, name: preset.name, description: preset.description, builtin: preset.builtin });
    this.baseline.set(architecture);
    this.draft.set({ ...architecture });
    this.reference.set(preset.key);
  }

  private confirmDiscard(): boolean { return !this.dirty() || confirm('Discard the unsaved changes to the current architecture?'); }

  loadPreset(preset: ArchitecturePreset) { if (preset.key !== this.source().key && this.confirmDiscard()) this.applyPreset(preset); }

  newDraft() {
    const base = this.presets().find((preset) => preset.key === 'xreport-default');
    if (!base || !this.confirmDiscard()) return;
    this.startDraft('Untitled architecture', '', base.architecture as ModelArchitecture, base.key);
  }

  duplicatePreset(preset: ArchitecturePreset) {
    if (!this.confirmDiscard()) return;
    this.startDraft(`${preset.name} copy`, preset.description, preset.architecture as ModelArchitecture, preset.key);
    this.openDialog('save-as');
  }

  private startDraft(name: string, description: string, architecture: ModelArchitecture, reference: string | null) {
    this.source.set({ key: null, name, description, builtin: false });
    this.baseline.set(null);
    this.draft.set({ ...architecture });
    this.reference.set(reference);
  }

  revert() {
    const baseline = this.baseline();
    if (baseline) this.draft.set({ ...baseline });
    else this.newDraft();
  }

  value(field: ArchitectureField): unknown { return this.draft()?.[field]; }

  /** Empty or partial numbers are sent as null so the server reports the field. */
  numberValue(event: Event): number | null { const raw = (event.target as HTMLInputElement).value; return raw.trim() === '' || Number.isNaN(Number(raw)) ? null : Number(raw); }

  selectValue(event: Event): string | null { return (event.target as HTMLSelectElement).value || null; }

  update(field: ArchitectureField, value: unknown) { this.draft.update((draft) => (draft ? { ...draft, [field]: value } as ModelArchitecture : draft)); }

  optionsFor(spec: FieldSpec) { return spec.field === 'image_encoder' ? this.encoders().map((encoder) => ({ value: encoder.id, label: encoder.label })) : spec.options ?? []; }

  fieldIssue(field: string): string | null { return this.preview()?.issues.find((issue) => issue.field === field)?.message ?? null; }

  fieldLabel(field: string): string { return FIELDS[field as ArchitectureField]?.label ?? field; }

  compatibilityTitle(level: string): string {
    if (level === 'identical') return 'Identical architecture';
    if (level === 'weights_compatible') return 'Weights compatible';
    return 'Requires retraining';
  }

  openDialog(mode: DetailsDialog['mode']) {
    const source = this.source();
    const name = mode === 'save-as' && source.key ? `${source.name} copy` : source.name;
    this.dialog.set({ mode, name, description: source.description, error: null });
  }

  async submitDialog() {
    const current = this.dialog();
    const architecture = this.preview()?.architecture;
    if (!current || !architecture) return;
    const request = { name: current.name.trim(), description: current.description, architecture };
    const key = this.source().key;
    this.busy.set(true);
    const result = current.mode === 'details' && key ? await this.api.updatePreset(key, { ...request, architecture: this.baseline() ?? architecture }) : await this.api.createPreset(request);
    this.busy.set(false);
    if (!result.result) { this.dialog.set({ ...current, error: result.error ?? 'Failed to save the preset' }); return; }
    this.dialog.set(null);
    this.notifications.notify({ type: 'success', message: `Saved architecture preset "${result.result.name}".`, history: false });
    if (current.mode === 'details') {
      // Keep unsaved architecture edits while renaming.
      const draft = this.draft();
      await this.reloadPresets(result.result.key);
      if (draft) this.draft.set(draft);
    } else {
      await this.reloadPresets(result.result.key);
    }
  }

  async save() {
    const source = this.source();
    const architecture = this.preview()?.architecture;
    if (!architecture) return;
    if (!source.key) { this.openDialog('save-as'); return; }
    if (source.builtin) return;
    this.busy.set(true);
    const result = await this.api.updatePreset(source.key, { name: source.name, description: source.description, architecture });
    this.busy.set(false);
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to save the preset'); return; }
    this.notifications.notify({ type: 'success', message: `Saved architecture preset "${result.result.name}".`, history: false });
    await this.reloadPresets(result.result.key);
  }

  async deletePreset(preset: ArchitecturePreset) {
    if (!confirm(`Delete the architecture preset "${preset.name}"? Checkpoints trained with it keep their own copy of the architecture.`)) return;
    const result = await this.api.deletePreset(preset.key);
    if (result.error) { this.pageError.set(result.error); return; }
    const wasActive = this.source().key === preset.key;
    await this.reloadPresets();
    if (wasActive) {
      const fallback = this.presets()[0];
      if (fallback) this.applyPreset(fallback);
    }
  }

  exportPreset(name: string, description: string, architecture: ArchitecturePreset['architecture']) {
    const file = buildArchitectureFile(name, description, architecture as ModelArchitecture);
    const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${name.trim().replace(/[^\w.-]+/g, '_') || 'architecture'}.xreport-architecture.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async importFile(input: HTMLInputElement) {
    const file = input.files?.[0];
    input.value = '';
    if (!file || !this.confirmDiscard()) return;
    try {
      const parsed = parseArchitectureFile(await file.text());
      const result = await this.api.createPreset({ name: parsed.name, description: parsed.description, architecture: parsed.architecture as ModelArchitecture });
      if (!result.result) { this.pageError.set(`Import failed: ${result.error ?? 'unknown error'}`); return; }
      this.notifications.notify({ type: 'success', message: `Imported architecture preset "${result.result.name}".`, history: false });
      await this.reloadPresets(result.result.key);
    } catch (error) {
      this.pageError.set(`Import failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async loadCheckpointArchitecture() {
    const reference = this.reference();
    if (!reference?.startsWith('checkpoint:') || !this.confirmDiscard()) return;
    const checkpoint = reference.slice('checkpoint:'.length);
    const result = await this.api.getCheckpointArchitecture(checkpoint);
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to read the checkpoint architecture'); return; }
    const description = result.result.legacy ? `Reconstructed from checkpoint ${checkpoint} (trained before architecture snapshots).` : `Architecture of checkpoint ${checkpoint}.`;
    this.startDraft(`${checkpoint} architecture`, description, result.result.architecture as ModelArchitecture, reference);
  }
}
