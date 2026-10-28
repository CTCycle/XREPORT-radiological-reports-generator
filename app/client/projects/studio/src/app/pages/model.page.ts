import { CommonModule } from '@angular/common';
import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideChevronDown, lucideChevronRight, lucideCircleCheck, lucideCopyPlus, lucideDownload, lucideLock, lucideNetwork, lucidePlus, lucideRotateCcw, lucideSave, lucideTrash2, lucideTriangleAlert, lucideUndo2, lucideUpload, lucideX } from '@ng-icons/lucide';
import { buildArchitectureFile, formatBytes, formatParameters, parseArchitectureFile } from '../common/model-architecture';
import { ArchitectureDiagramComponent } from '../components/architecture-diagram.component';
import { HelpPopoverComponent } from '@xreport/shared/app/components/help-popover.component';
import { ModalFocusDirective } from '@xreport/shared/app/components/modal-focus.directive';
import { ModelArchitectureApiService } from '../services/model-architecture-api.service';
import { NotificationService } from '@xreport/shared/app/services/notification.service';
import { StorageService } from '@xreport/shared/app/services/storage.service';
import { TrainingApiService } from '../services/training-api.service';
import type { ArchitectureGraph, ArchitecturePreset, ArchitecturePreviewResponse, ArchitectureStats, EncoderOption, ModelArchitecture } from '../types/modelArchitectureApi';

type ArchitectureField = keyof ModelArchitecture;
type Stack = 'encoder' | 'decoder';
type PresetSection = 'builtin' | 'custom';

interface FieldSpec {
  field: ArchitectureField;
  label: string;
  help: string;
  kind: 'range' | 'number' | 'toggle' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  advanced?: boolean;
  /** Effective setting that decides tensor shapes when it differs from the field name. */
  shapeKey?: string;
  /** Only shown while this returns true. */
  visible?: (architecture: ModelArchitecture) => boolean;
}

const ACTIVATIONS = [{ value: 'relu', label: 'ReLU (original XREPORT)' }, { value: 'gelu', label: 'GELU' }, { value: 'silu', label: 'SiLU (swish)' }];

function stackFields(stack: Stack): FieldSpec[] {
  const name = stack === 'encoder' ? 'Encoder' : 'Decoder';
  const count = (stack === 'encoder' ? 'num_encoders' : 'num_decoders') as ArchitectureField;
  const key = (suffix: string) => `${stack}_${suffix}` as ArchitectureField;
  const mode = key('head_dim_mode');
  return [
    { field: count, label: `${name} layers`, kind: 'range', min: 1, max: 12, step: 1, help: stack === 'encoder' ? 'Transformer blocks that refine the projected image features with self-attention.' : 'Transformer blocks that generate the report: masked self-attention over report tokens, then cross-attention to the image features.' },
    { field: key('heads'), label: 'Attention heads', kind: 'range', min: 1, max: 16, step: 1, help: `Parallel attention heads in every ${stack} block${stack === 'decoder' ? ' (self- and cross-attention)' : ''}.` },
    { field: mode, label: 'Head size', kind: 'select', shapeKey: `${stack}_key_dim`, options: [{ value: 'full', label: 'Full width (original XREPORT)' }, { value: 'split', label: 'Split: width / heads' }, { value: 'custom', label: 'Custom' }], help: 'Full gives every head the whole model width (original XREPORT, large). Split divides the width between heads, as in standard transformers. Custom sets the size of each head directly.' },
    { field: key('head_dim'), label: 'Custom head size', kind: 'range', min: 8, max: 1024, step: 8, shapeKey: `${stack}_key_dim`, visible: (architecture) => architecture[mode] === 'custom', help: 'Dimensions of each attention head, independent of the model width.' },
    { field: key('ffn_units'), label: 'Feed-forward units', kind: 'range', min: 64, max: 4096, step: 64, help: 'Hidden size of the two-layer feed-forward network in every block. Original XREPORT uses the model width; standard transformers use about 4x.' },
    { field: key('ffn_activation'), label: 'Feed-forward activation', kind: 'select', advanced: true, options: ACTIVATIONS, help: 'Non-linearity applied after both dense layers of the feed-forward network.' },
    { field: key('dropout'), label: 'Dropout', kind: 'range', min: 0, max: 0.5, step: 0.05, advanced: true, help: `Fraction of activations dropped during training for regularisation. Applied to the ${stack} feed-forward networks${stack === 'decoder' ? ' and attention' : ''}.` },
    { field: 'norm_epsilon', label: 'Layer norm epsilon', kind: 'number', min: 1e-8, max: 1e-2, step: 1e-5, advanced: true, help: 'Numerical-stability constant of every residual layer normalisation. Shared by encoder and decoder blocks.' },
  ];
}

const SPECS: FieldSpec[] = [
  { field: 'image_encoder', label: 'Image encoder', kind: 'select', help: 'Pretrained vision backbone that turns the radiograph into patch features. Only encoders that XREPORT Inference can load are offered.' },
  { field: 'freeze_image_encoder', label: 'Freeze image encoder', kind: 'toggle', help: 'Frozen keeps the pretrained BEiT weights fixed and trains only the projection and transformer. Unfreezing fine-tunes ~86M more parameters and needs much more GPU memory.' },
  { field: 'embedding_dims', label: 'Model width', kind: 'range', min: 64, max: 1024, step: 8, help: 'Width of every token vector: the image projection, the token embeddings and all encoder and decoder blocks share it.' },
  ...stackFields('encoder'),
  ...stackFields('decoder'),
  { field: 'classifier_units', label: 'Classifier units', kind: 'range', min: 64, max: 4096, step: 64, help: 'Hidden layer of the token classifier between the decoder output and the vocabulary.' },
  { field: 'classifier_activation', label: 'Classifier activation', kind: 'select', advanced: true, options: ACTIVATIONS, help: 'Non-linearity of the classifier hidden layer.' },
  { field: 'temperature', label: 'Softmax temperature', kind: 'range', min: 0.1, max: 2, step: 0.05, help: 'Divides the classifier logits before the softmax. Values below 1 sharpen the token distribution, above 1 flatten it.' },
];
const FIELDS = new Map<ArchitectureField, FieldSpec>(SPECS.map((spec) => [spec.field, spec]));

// Inspector fields per diagram node. Components that are part of the fixed
// XREPORT design have no fields and say why in NODE_NOTES.
const NODE_FIELDS: Record<string, ArchitectureField[]> = {
  image_input: [],
  image_encoder: ['image_encoder', 'freeze_image_encoder'],
  projection: ['embedding_dims'],
  encoder_stack: ['num_encoders', 'encoder_heads', 'encoder_head_dim_mode', 'encoder_head_dim', 'encoder_ffn_units', 'encoder_ffn_activation', 'encoder_dropout', 'norm_epsilon'],
  token_input: [],
  embedding: ['embedding_dims'],
  decoder_stack: ['num_decoders', 'decoder_heads', 'decoder_head_dim_mode', 'decoder_head_dim', 'decoder_ffn_units', 'decoder_ffn_activation', 'decoder_dropout', 'norm_epsilon'],
  classifier: ['classifier_units', 'classifier_activation', 'temperature'],
};
const NODE_NOTES: Record<string, string> = {
  image_input: 'Fixed component. Radiographs are resized to 224 x 224 RGB and normalised in code; input preprocessing is not configurable.',
  projection: 'A dense layer maps the 768-wide BEiT features to the model width. It is always trained from scratch; its input size is fixed by the BEiT encoder.',
  token_input: 'Fixed component. The report length comes from the processed dataset (max report size), not from the architecture.',
  embedding: 'Learned token and position embeddings. Only the width is configurable: the vocabulary and the number of positions come from the dataset tokenizer.',
  encoder_stack: 'Self-attention plus feed-forward blocks. The encoder is configured independently of the decoder.',
  decoder_stack: 'Masked self-attention, cross-attention to the encoder output and feed-forward blocks. The decoder is configured independently of the encoder.',
  classifier: 'Hidden layer, softmax over the vocabulary. The output size is the dataset vocabulary and is not configurable.',
};
const EXTRA_LABELS: Record<string, string> = {
  encoder_key_dim: 'Encoder head size', decoder_key_dim: 'Decoder head size',
  encoder_heads: 'Encoder heads', decoder_heads: 'Decoder heads',
  encoder_ffn_units: 'Encoder feed-forward units', decoder_ffn_units: 'Decoder feed-forward units',
  encoder_ffn_activation: 'Encoder feed-forward activation', decoder_ffn_activation: 'Decoder feed-forward activation',
  encoder_dropout: 'Encoder dropout', decoder_dropout: 'Decoder dropout',
};
const LEVEL_TITLES: Record<string, string> = { identical: 'Identical architecture', compatible: 'Compatible, configurable differences', mismatch: 'Architecture mismatch' };
const MISMATCH_PREVIEW = 6;
const COLLAPSED_KEY = 'xreport.model.collapsed';
const DEFAULT_PRESET_KEY = 'xreport-default';

interface Source { key: string | null; name: string; description: string; builtin: boolean; }
interface DetailsDialog { mode: 'save-as' | 'details'; name: string; description: string; error: string | null; }
interface Confirmation { title: string; message: string; confirmLabel: string; danger: boolean; resolve: (accepted: boolean) => void; }

const PREVIEW_DEBOUNCE_MS = 150;

@Component({
  standalone: true,
  selector: 'app-model-page',
  imports: [CommonModule, NgIcon, ArchitectureDiagramComponent, HelpPopoverComponent, ModalFocusDirective],
  providers: [provideIcons({ lucideChevronDown, lucideChevronRight, lucideCircleCheck, lucideCopyPlus, lucideDownload, lucideLock, lucideNetwork, lucidePlus, lucideRotateCcw, lucideSave, lucideTrash2, lucideTriangleAlert, lucideUndo2, lucideUpload, lucideX })],
  styleUrls: ['../styles/TrainingPage.css', '../styles/ModelPage.css'],
  template: `
    <main class="model-page page-container">
      <div class="header">
        <h1>Model Architecture</h1>
        <p>Define, compare and save XREPORT transformer architectures. Presets are untrained definitions; trained weights live in checkpoints.</p>
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
          <div class="model-panel-body">
            @for (group of presetGroups(); track group.section) {
              <button type="button" class="model-library-group" [attr.aria-expanded]="!isCollapsed(group.section)" (click)="toggleSection(group.section)">
                <ng-icon [name]="isCollapsed(group.section) ? 'lucideChevronRight' : 'lucideChevronDown'" size="14"/>{{ group.title }}<span class="model-library-count">{{ group.presets.length }}</span>
              </button>
              @if (!isCollapsed(group.section)) {
                @if (!group.presets.length) { <p class="model-library-empty">{{ group.section === 'custom' ? 'No saved presets yet. Duplicate a preset or save the current draft.' : 'All built-in presets are removed.' }}</p> }
                <ul class="model-preset-list">
                  @for (preset of group.presets; track preset.key) {
                    <li class="model-preset" [class.active]="source().key === preset.key">
                      <button type="button" class="model-preset-main" [attr.aria-current]="source().key === preset.key ? 'true' : null" (click)="loadPreset(preset)">
                        <span class="model-preset-name">@if (preset.builtin) { <ng-icon name="lucideLock" size="12" [attr.aria-label]="preset.protected ? 'Protected default' : 'Built-in'"/> }{{ preset.name }}</span>
                        <span class="model-preset-meta">{{ formatParameters(preset.stats.total_parameters) }} params · {{ preset.architecture.num_encoders }}+{{ preset.architecture.num_decoders }} layers · width {{ preset.architecture.embedding_dims }}</span>
                        <span class="model-preset-meta">{{ formatParameters(preset.stats.trainable_parameters) }} trainable</span>
                      </button>
                      <div class="model-preset-actions">
                        <button type="button" class="icon-button" title="Duplicate" [attr.aria-label]="'Duplicate ' + preset.name" (click)="duplicatePreset(preset)"><ng-icon name="lucideCopyPlus"/></button>
                        <button type="button" class="icon-button" title="Export JSON" [attr.aria-label]="'Export ' + preset.name" (click)="exportPreset(preset.name, preset.description, preset.architecture)"><ng-icon name="lucideDownload"/></button>
                        @if (!preset.protected) { <button type="button" class="icon-button danger" [title]="preset.builtin ? 'Remove from my presets' : 'Delete'" [attr.aria-label]="(preset.builtin ? 'Remove ' : 'Delete ') + preset.name" (click)="deletePreset(preset)"><ng-icon name="lucideTrash2"/></button> }
                      </div>
                    </li>
                  }
                </ul>
              }
            }
          </div>
          @if (hiddenBuiltins() > 0) {
            <button type="button" class="btn btn-secondary model-restore" (click)="restoreBuiltins()"><ng-icon name="lucideRotateCcw"/>Restore {{ hiddenBuiltins() }} built-in preset{{ hiddenBuiltins() === 1 ? '' : 's' }}</button>
          }
        </aside>

        <section class="model-panel model-canvas" aria-label="Architecture">
          <div class="model-canvas-toolbar">
            <div class="model-canvas-title">
              <h2>{{ source().name }}@if (dirty()) { <span class="model-dirty" title="Unsaved changes" aria-label="Unsaved changes"></span> }</h2>
              <span class="model-canvas-subtitle">{{ sourceKind() }}</span>
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
              <div><dt>Parameters</dt><dd>{{ formatParameters(current.total_parameters) }}</dd></div>
              <div><dt>Trainable</dt><dd>{{ formatParameters(current.trainable_parameters) }}</dd></div>
              <div><dt>Weights fp32</dt><dd>{{ formatBytes(current.weights_bytes) }}</dd></div>
              <div><dt>Training memory <app-help-popover label="Explain training memory" title="Training memory estimate" body="Weights plus gradients and the two AdamW moment buffers of the trainable weights. Activations, which grow with batch size and report length, are not included." /></dt><dd>{{ formatBytes(current.training_memory_bytes) }}</dd></div>
              <div><dt title="Transformer depth: encoder plus decoder layers">Depth</dt><dd>{{ current.depth }} layers</dd></div>
            </dl>
            <p class="model-assumptions">Estimates for a freshly initialised model, assuming a {{ current.vocabulary_size | number }} token vocabulary, {{ current.sequence_length }} report tokens and fp32 weights.</p>
          }
          <div class="model-diagram" [class.stale]="!valid() || previewPending()">
            @if (graph(); as currentGraph) {
              <app-architecture-diagram [graph]="currentGraph" [selected]="selectedNode()" (nodeSelected)="selectedNode.set($event)" />
            } @else { <p class="model-diagram-empty">Loading architecture…</p> }
          </div>
        </section>

        <aside class="model-panel model-inspector" aria-label="Component settings">
          <div class="model-panel-header">
            <h2>{{ selectedLabel() }}</h2>
            @if (selectedFixed()) { <span class="model-locked"><ng-icon name="lucideLock" size="12"/>Fixed</span> }
          </div>
          <div class="model-panel-body">
            @for (issue of generalIssues(); track issue) { <div class="model-banner error" role="alert"><ng-icon name="lucideTriangleAlert"/><span>{{ issue }}</span></div> }
            @if (nodeNote()) { <p class="model-note">{{ nodeNote() }}</p> }
            @if (selectedEncoder(); as encoder) {
              <dl class="model-facts">
                <div><dt>Repository</dt><dd class="mono">{{ encoder.repository }}</dd></div>
                <div><dt>Revision</dt><dd class="mono">{{ encoder.revision.slice(0, 12) }}</dd></div>
                <div><dt>Output</dt><dd>{{ encoder.output_tokens }} tokens × {{ encoder.feature_dims }}</dd></div>
                <div><dt>Parameters</dt><dd>{{ formatParameters(encoder.parameters) }}, pretrained</dd></div>
                <div><dt>XREPORT Inference</dt><dd>{{ encoder.inference_compatible ? 'Checkpoints can be exported and imported' : 'Not importable' }}</dd></div>
              </dl>
            }
            @if (nodeFacts().length) {
              <dl class="model-facts">
                @for (fact of nodeFacts(); track fact.label) { <div><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div> }
              </dl>
            }
            @for (spec of basicFields(); track spec.field) { <ng-container *ngTemplateOutlet="control; context: { $implicit: spec }" /> }
            @if (advancedFields().length) {
              <details class="model-advanced" [open]="showAdvanced()" (toggle)="showAdvanced.set($any($event.target).open)">
                <summary>Advanced</summary>
                @for (spec of advancedFields(); track spec.field) { <ng-container *ngTemplateOutlet="control; context: { $implicit: spec }" /> }
              </details>
            }

            <section class="model-compat" aria-label="Compatibility">
              <h3>Compatibility</h3>
              <label class="model-inline-field"><span>Compare with</span>
                <select class="form-select" (change)="reference.set(selectValue($event))" aria-label="Compare with a preset or checkpoint">
                  <option value="" [selected]="!reference()">Nothing</option>
                  <optgroup label="Presets (architecture only)">
                    @for (preset of presets(); track preset.key) { <option [value]="preset.key" [selected]="preset.key === reference()">{{ preset.name }}</option> }
                  </optgroup>
                  @if (checkpoints().length) {
                    <optgroup label="Checkpoints (trained weights)">
                      @for (checkpoint of checkpoints(); track checkpoint) { <option [value]="'checkpoint:' + checkpoint" [selected]="'checkpoint:' + checkpoint === reference()">{{ checkpoint }}</option> }
                    </optgroup>
                  }
                </select>
              </label>
              @if (preview()?.compatibility; as compat) {
                <div class="model-compat-result" [attr.data-level]="compat.level">
                  <ng-icon [name]="compat.level === 'mismatch' ? 'lucideTriangleAlert' : 'lucideCircleCheck'"/>
                  <div><strong>{{ levelTitle(compat.level) }}</strong><p>{{ compat.message }}</p></div>
                </div>
                @if (compat.changes.length) {
                  <ul class="model-compat-changes">
                    @for (change of compat.changes; track change.field) {
                      <li><span>{{ fieldLabel(change.field) }}</span><span class="mono">{{ change.before }} → {{ change.after }}</span><span class="model-impact" [attr.data-impact]="change.scope" [title]="change.scope === 'shape' ? 'Changes the shape of weight tensors' : 'Does not change any tensor shape'">{{ change.scope === 'shape' ? 'Shape' : 'Setting' }}</span></li>
                    }
                  </ul>
                }
                @if (compat.reference_kind === 'preset') {
                  <p class="model-note muted">A preset is an architecture definition and holds no trained weights.</p>
                }
                @if (compat.weights; as weights) {
                  <div class="model-compat-result" [attr.data-level]="weights.status === 'loadable' ? 'identical' : 'weights-incompatible'">
                    <ng-icon [name]="weights.status === 'loadable' ? 'lucideCircleCheck' : 'lucideTriangleAlert'"/>
                    <div><strong>{{ weights.status === 'loadable' ? 'Checkpoint weights compatible' : 'Checkpoint weights incompatible' }}</strong><p>{{ weights.message }}</p></div>
                  </div>
                  @if (weights.mismatches.length) {
                    <ul class="model-compat-changes model-weight-list">
                      @for (item of visibleMismatches(weights.mismatches); track item.layer + item.tensor) {
                        <li><span class="mono">{{ item.layer }} · {{ item.tensor }}</span><span class="mono">this {{ shapeText(item.expected) }} · checkpoint {{ shapeText(item.found) }}</span></li>
                      }
                    </ul>
                    @if (weights.mismatches.length > mismatchPreview) {
                      <button type="button" class="btn btn-secondary model-compat-load" (click)="showAllMismatches.set(!showAllMismatches())">{{ showAllMismatches() ? 'Show fewer' : 'Show all ' + weights.mismatches.length }}</button>
                    }
                  }
                  <p class="model-note muted">Assumes this architecture is trained on the checkpoint's vocabulary and report length. Reusing these weights requires every tensor shape to match.</p>
                }
              }
              @if (reference()?.startsWith('checkpoint:')) {
                <button type="button" class="btn btn-secondary model-compat-load" (click)="loadCheckpointArchitecture()">Load architecture from checkpoint</button>
              }
            </section>
          </div>
        </aside>
      </div>

      <ng-template #control let-spec>
        @if (isVisible(spec)) {
          <div class="model-control" [class.invalid]="fieldIssue(spec.field)">
            <div class="model-control-head">
              <label [attr.for]="'arch-' + spec.field">{{ spec.label }}</label>
              <app-help-popover [label]="'Explain ' + spec.label" [title]="spec.label" [body]="spec.help" />
              <span class="model-impact" [attr.data-impact]="affectsShape(spec) ? 'shape' : 'setting'" [title]="affectsShape(spec) ? 'Sets the shape of weight tensors: weights trained with another value cannot be loaded' : 'Does not change any tensor shape: weights trained with another value still load'">{{ affectsShape(spec) ? 'Shape' : 'Setting' }}</span>
            </div>
            @switch (spec.kind) {
              @case ('range') {
                <div class="model-range">
                  <input type="range" [min]="spec.min" [max]="spec.max" [step]="spec.step" [value]="value(spec.field)" (input)="update(spec.field, numberValue($event))" [attr.aria-label]="spec.label + ' slider'" />
                  <input class="form-input" type="number" [id]="'arch-' + spec.field" [min]="spec.min" [max]="spec.max" [step]="spec.step" [value]="value(spec.field)" (input)="update(spec.field, numberValue($event))" />
                </div>
              }
              @case ('number') {
                <input class="form-input" type="number" [id]="'arch-' + spec.field" [min]="spec.min" [max]="spec.max" step="any" [value]="value(spec.field)" (input)="update(spec.field, numberValue($event))" />
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
        }
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

      @if (confirmation(); as current) {
        <div class="model-modal-backdrop" role="presentation">
          <section class="model-modal" role="alertdialog" aria-modal="true" aria-labelledby="model-confirm-title" aria-describedby="model-confirm-message" appModalFocus (modalEscape)="answer(false)">
            <header><h2 id="model-confirm-title">{{ current.title }}</h2><button type="button" class="icon-button" aria-label="Close" (click)="answer(false)"><ng-icon name="lucideX"/></button></header>
            <p id="model-confirm-message" class="model-note">{{ current.message }}</p>
            <footer><button type="button" class="btn btn-secondary" (click)="answer(false)">Cancel</button><button type="button" [class]="current.danger ? 'danger-button' : 'btn btn-primary'" (click)="answer(true)">{{ current.confirmLabel }}</button></footer>
          </section>
        </div>
      }
    </main>
  `,
})
export class ModelPage {
  private readonly api = inject(ModelArchitectureApiService);
  private readonly trainingApi = inject(TrainingApiService);
  private readonly notifications = inject(NotificationService);
  private readonly storage = inject(StorageService);
  readonly formatParameters = formatParameters;
  readonly formatBytes = formatBytes;
  readonly mismatchPreview = MISMATCH_PREVIEW;

  readonly presets = signal<ArchitecturePreset[]>([]);
  readonly hiddenBuiltins = signal(0);
  readonly encoders = signal<EncoderOption[]>([]);
  readonly checkpoints = signal<string[]>([]);
  readonly source = signal<Source>({ key: null, name: 'Loading…', description: '', builtin: false });
  readonly baseline = signal<ModelArchitecture | null>(null);
  readonly draft = signal<ModelArchitecture | null>(null);
  readonly preview = signal<ArchitecturePreviewResponse | null>(null);
  readonly previewPending = signal(false);
  private readonly lastValid = signal<{ graph: ArchitectureGraph; stats: ArchitectureStats } | null>(null);
  readonly shapeFields = signal<ReadonlySet<string>>(new Set());
  readonly selectedNode = signal('encoder_stack');
  readonly reference = signal<string | null>(null);
  readonly showAdvanced = signal(false);
  readonly showAllMismatches = signal(false);
  readonly collapsed = signal<Record<string, boolean>>(this.storage.readRecord<boolean>(COLLAPSED_KEY));
  readonly dialog = signal<DetailsDialog | null>(null);
  readonly confirmation = signal<Confirmation | null>(null);
  readonly busy = signal(false);
  readonly pageError = signal<string | null>(null);
  private previewRequest = 0;

  readonly dirty = computed(() => JSON.stringify(this.draft()) !== JSON.stringify(this.baseline()));
  readonly valid = computed(() => this.preview()?.valid ?? false);
  readonly canSave = computed(() => this.valid() && !this.busy() && (!this.source().key || (!this.source().builtin && this.dirty())));
  readonly graph = computed(() => this.preview()?.graph ?? this.lastValid()?.graph ?? null);
  readonly stats = computed(() => this.preview()?.stats ?? this.lastValid()?.stats ?? null);
  readonly presetGroups = computed(() => [
    { section: 'builtin' as PresetSection, title: 'Built-in', presets: this.presets().filter((preset) => preset.builtin) },
    { section: 'custom' as PresetSection, title: 'Custom', presets: this.presets().filter((preset) => !preset.builtin) },
  ]);
  readonly sourceKind = computed(() => {
    const source = this.source();
    if (source.builtin) return 'Built-in preset · untrained architecture definition, read-only';
    return source.key ? 'Custom preset · untrained architecture definition' : 'Unsaved draft · untrained architecture definition';
  });
  readonly selectedGraphNode = computed(() => this.graph()?.nodes.find((node) => node.id === this.selectedNode()) ?? null);
  readonly selectedLabel = computed(() => this.selectedGraphNode()?.label ?? 'Component');
  readonly selectedFixed = computed(() => this.selectedGraphNode()?.fixed ?? false);
  readonly nodeNote = computed(() => NODE_NOTES[this.selectedNode()] ?? '');
  private readonly nodeSpecs = computed(() => (NODE_FIELDS[this.selectedNode()] ?? []).map((field) => FIELDS.get(field)!));
  readonly basicFields = computed(() => this.nodeSpecs().filter((spec) => !spec.advanced));
  readonly advancedFields = computed(() => this.nodeSpecs().filter((spec) => spec.advanced));
  readonly selectedEncoder = computed(() => this.selectedNode() === 'image_encoder' ? this.encoders().find((encoder) => encoder.id === this.draft()?.image_encoder) ?? null : null);
  readonly generalIssues = computed(() => (this.preview()?.issues ?? []).filter((issue) => !issue.field).map((issue) => issue.message));
  readonly nodeFacts = computed(() => {
    const node = this.selectedGraphNode();
    const properties = node?.properties as Record<string, unknown> | undefined;
    if (!node || !properties || !('block_parameters' in properties)) return [];
    return [
      { label: 'Resolved head size', value: String(properties['key_dim']) },
      { label: 'Parameters per block', value: formatParameters(Number(properties['block_parameters'])) },
      { label: 'Parameters in stack', value: formatParameters(node.parameters) },
    ];
  });

  constructor() {
    void this.initialize();
    let timer: ReturnType<typeof setTimeout> | undefined;
    effect(() => {
      const architecture = this.draft();
      const reference = this.reference();
      if (!architecture) return;
      clearTimeout(timer);
      this.previewPending.set(true);
      timer = setTimeout(() => void this.refreshPreview(architecture, reference), PREVIEW_DEBOUNCE_MS);
    });
    inject(DestroyRef).onDestroy(() => clearTimeout(timer));
  }

  private async initialize() {
    void this.api.getEncoders().then((encoders) => { if (encoders.result) this.encoders.set(encoders.result.encoders); });
    void this.trainingApi.getCheckpoints().then((checkpoints) => { if (checkpoints.result) this.checkpoints.set(checkpoints.result.checkpoints.filter((checkpoint) => checkpoint.artifact_status === 'ready').map((checkpoint) => checkpoint.name)); });
    const presets = await this.api.getPresets();
    if (!presets.result) { this.pageError.set(presets.error ?? 'Failed to load architecture presets'); return; }
    this.setPresets(presets.result.presets, presets.result.hidden_builtins ?? 0);
    const initial = presets.result.presets[0];
    if (initial) this.applyPreset(initial);
  }

  private setPresets(presets: ArchitecturePreset[], hiddenBuiltins: number) {
    this.presets.set(presets);
    this.hiddenBuiltins.set(hiddenBuiltins);
    const reference = this.reference();
    if (reference && !reference.startsWith('checkpoint:') && !presets.some((preset) => preset.key === reference)) this.reference.set(DEFAULT_PRESET_KEY);
  }

  /** Only the response of the latest request is applied. */
  private async refreshPreview(architecture: ModelArchitecture, reference: string | null) {
    const request = ++this.previewRequest;
    const response = await this.api.preview({ architecture, reference });
    if (request !== this.previewRequest) return;
    this.previewPending.set(false);
    if (!response.result) { this.pageError.set(response.error ?? 'Failed to preview the architecture'); return; }
    this.preview.set(response.result);
    this.showAllMismatches.set(false);
    if (response.result.shape_fields?.length) this.shapeFields.set(new Set(response.result.shape_fields));
    if (response.result.graph && response.result.stats) this.lastValid.set({ graph: response.result.graph, stats: response.result.stats });
  }

  private async reloadPresets(selectKey?: string) {
    const result = await this.api.getPresets();
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to load architecture presets'); return; }
    this.setPresets(result.result.presets, result.result.hidden_builtins ?? 0);
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

  private ask(title: string, message: string, confirmLabel: string, danger = false): Promise<boolean> {
    return new Promise((resolve) => this.confirmation.set({ title, message, confirmLabel, danger, resolve }));
  }

  answer(accepted: boolean) {
    const current = this.confirmation();
    this.confirmation.set(null);
    current?.resolve(accepted);
  }

  private async confirmDiscard(): Promise<boolean> {
    return !this.dirty() || this.ask('Discard unsaved changes?', `The changes to "${this.source().name}" have not been saved and will be lost.`, 'Discard changes', true);
  }

  isCollapsed(section: PresetSection): boolean { return this.collapsed()[section] ?? false; }

  toggleSection(section: PresetSection) {
    const next = { ...this.collapsed(), [section]: !this.isCollapsed(section) };
    this.collapsed.set(next);
    this.storage.writeRecord(COLLAPSED_KEY, next);
  }

  async loadPreset(preset: ArchitecturePreset) { if (preset.key !== this.source().key && await this.confirmDiscard()) this.applyPreset(preset); }

  async newDraft() {
    const base = this.presets().find((preset) => preset.key === DEFAULT_PRESET_KEY);
    if (!base || !await this.confirmDiscard()) return;
    this.startDraft('Untitled architecture', '', base.architecture as ModelArchitecture, base.key);
  }

  async duplicatePreset(preset: ArchitecturePreset) {
    if (!await this.confirmDiscard()) return;
    this.startDraft(`${preset.name} copy`, preset.description, preset.architecture as ModelArchitecture, preset.key);
    this.openDialog('save-as');
  }

  private startDraft(name: string, description: string, architecture: ModelArchitecture, reference: string | null) {
    this.source.set({ key: null, name, description, builtin: false });
    this.baseline.set(null);
    this.draft.set({ ...architecture });
    this.reference.set(reference);
  }

  async revert() {
    const baseline = this.baseline();
    if (baseline) this.draft.set({ ...baseline });
    else await this.newDraft();
  }

  value(field: ArchitectureField): unknown { return this.draft()?.[field]; }

  /** Empty or partial numbers are sent as null so the server reports the field. */
  numberValue(event: Event): number | null { const raw = (event.target as HTMLInputElement).value; return raw.trim() === '' || Number.isNaN(Number(raw)) ? null : Number(raw); }

  selectValue(event: Event): string | null { return (event.target as HTMLSelectElement).value || null; }

  update(field: ArchitectureField, value: unknown) { this.draft.update((draft) => (draft ? { ...draft, [field]: value } as ModelArchitecture : draft)); }

  isVisible(spec: FieldSpec): boolean { const draft = this.draft(); return !spec.visible || (draft !== null && spec.visible(draft)); }

  affectsShape(spec: FieldSpec): boolean { return this.shapeFields().has(spec.shapeKey ?? spec.field); }

  optionsFor(spec: FieldSpec) { return spec.field === 'image_encoder' ? this.encoders().map((encoder) => ({ value: encoder.id, label: encoder.label })) : spec.options ?? []; }

  fieldIssue(field: string): string | null { return this.preview()?.issues.find((issue) => issue.field === field)?.message ?? null; }

  fieldLabel(field: string): string { return EXTRA_LABELS[field] ?? FIELDS.get(field as ArchitectureField)?.label ?? field; }

  levelTitle(level: string): string { return LEVEL_TITLES[level] ?? level; }

  shapeText(shape: number[]): string { return shape.length ? `[${shape.join(' × ')}]` : 'absent'; }

  visibleMismatches<T>(items: T[]): T[] { return this.showAllMismatches() ? items : items.slice(0, MISMATCH_PREVIEW); }

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
    const accepted = preset.builtin
      ? await this.ask(`Remove "${preset.name}"?`, 'This built-in preset is removed from your preset list. You can bring it back with "Restore built-in presets". Checkpoints trained with it keep their own copy of the architecture.', 'Remove preset', true)
      : await this.ask(`Delete "${preset.name}"?`, 'The saved preset is deleted permanently. Checkpoints trained with it keep their own copy of the architecture.', 'Delete preset', true);
    if (!accepted) return;
    const result = await this.api.deletePreset(preset.key);
    if (result.error) { this.pageError.set(result.error); return; }
    const wasActive = this.source().key === preset.key;
    await this.reloadPresets();
    if (wasActive) {
      const fallback = this.presets().find((item) => item.key === DEFAULT_PRESET_KEY) ?? this.presets()[0];
      if (fallback) this.applyPreset(fallback);
    }
  }

  async restoreBuiltins() {
    const result = await this.api.restorePresets();
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to restore the built-in presets'); return; }
    this.setPresets(result.result.presets, result.result.hidden_builtins ?? 0);
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
    if (!file || !await this.confirmDiscard()) return;
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
    if (!reference?.startsWith('checkpoint:') || !await this.confirmDiscard()) return;
    const checkpoint = reference.slice('checkpoint:'.length);
    const result = await this.api.getCheckpointArchitecture(checkpoint);
    if (!result.result) { this.pageError.set(result.error ?? 'Failed to read the checkpoint architecture'); return; }
    const description = result.result.legacy ? `Reconstructed from checkpoint ${checkpoint} (trained before architecture snapshots).` : `Architecture of checkpoint ${checkpoint}.`;
    this.startDraft(`${checkpoint} architecture`, description, result.result.architecture as ModelArchitecture, reference);
  }
}
