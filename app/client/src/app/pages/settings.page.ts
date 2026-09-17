import { CommonModule } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { SettingsApiService } from '../services/settings-api.service';
import type {
  ApplicationSettingsPatch,
  ApplicationSettingsResponse,
  EditableSettings,
} from '../types/settingsApi';

interface ValidationErrors {
  seed?: string;
  polling_interval?: string;
  model_timeout?: string;
}

@Component({
  selector: 'app-settings-page',
  imports: [CommonModule, FormsModule],
  template: `
    <main class="settings-page" aria-labelledby="settings-title">
      <header class="settings-header">
        <p class="settings-eyebrow">Application preferences</p>
        <h1 id="settings-title">Settings</h1>
        <p>Application behavior settings for this XREPORT installation.</p>
      </header>

      @if (loading()) {
        <section class="settings-state" role="status">Loading settings…</section>
      } @else if (loadError()) {
        <section class="settings-state settings-state-error" role="alert">
          <strong>Settings could not be loaded</strong>
          <span>{{ loadError() }}</span>
          <button type="button" class="secondary-button" (click)="load()">Try again</button>
        </section>
      } @else if (draft(); as values) {
        <form class="settings-form" (ngSubmit)="save()" novalidate>
          <section class="settings-section" aria-labelledby="settings-general-title">
            <div class="settings-section-heading">
              <h2 id="settings-general-title">General</h2>
              <p>Defaults used when new work starts without an explicit value.</p>
            </div>
            <div class="settings-field">
              <label for="default-seed">Default random seed</label>
              <input id="default-seed" type="number" inputmode="numeric" step="1" min="0" max="4294967295"
                [ngModel]="values.global.seed" (ngModelChange)="updateSeed($event)" name="seed"
                [disabled]="saving() || resetting()" [attr.aria-invalid]="validationErrors().seed ? 'true' : null"
                [attr.aria-describedby]="validationErrors().seed ? 'seed-help seed-error' : 'seed-help'" />
              <span id="seed-help" class="settings-help">Used for new preparation, training, or validation work. Running jobs keep the seed they started with.</span>
              @if (validationErrors().seed; as error) { <span id="seed-error" class="settings-error">{{ error }}</span> }
            </div>
          </section>

          <section class="settings-section" aria-labelledby="settings-data-title">
            <div class="settings-section-heading">
              <h2 id="settings-data-title">Data access</h2>
              <p>Control whether dataset workflows may access local paths.</p>
            </div>
            <label class="settings-toggle" for="local-filesystem-access">
              <input id="local-filesystem-access" type="checkbox"
                [ngModel]="values.features.allow_local_filesystem_access" (ngModelChange)="updateFilesystemAccess($event)"
                name="allow_local_filesystem_access" [disabled]="saving() || resetting()" />
              <span>
                <strong>Allow local filesystem access</strong>
                <span class="settings-help">Turning this off blocks new local filesystem operations but does not remove data already imported into XREPORT.</span>
              </span>
            </label>
          </section>

          <section class="settings-section" aria-labelledby="settings-advanced-title">
            <div class="settings-section-heading">
              <h2 id="settings-advanced-title">Advanced</h2>
              <p>Fine-tune polling and local inference operation limits.</p>
            </div>
            <div class="settings-field">
              <label for="polling-interval">Job status polling interval <span class="settings-unit">seconds</span></label>
              <input id="polling-interval" type="number" inputmode="decimal" step="0.25" min="0.25" max="60"
                [ngModel]="values.jobs.polling_interval" (ngModelChange)="updatePollingInterval($event)" name="polling_interval"
                [disabled]="saving() || resetting()" [attr.aria-invalid]="validationErrors().polling_interval ? 'true' : null"
                [attr.aria-describedby]="validationErrors().polling_interval ? 'polling-help polling-error' : 'polling-help'" />
              <span id="polling-help" class="settings-help">Controls how frequently the UI polls newly started long-running jobs. Existing jobs keep their current interval.</span>
              @if (validationErrors().polling_interval; as error) { <span id="polling-error" class="settings-error">{{ error }}</span> }
            </div>
            <div class="settings-field">
              <label for="inference-timeout">Inference timeout <span class="settings-unit">seconds</span></label>
              <input id="inference-timeout" type="number" inputmode="numeric" step="1" min="1"
                [ngModel]="values.inference.model_timeout" (ngModelChange)="updateModelTimeout($event)" name="model_timeout"
                [disabled]="saving() || resetting()" [attr.aria-invalid]="validationErrors().model_timeout ? 'true' : null"
                [attr.aria-describedby]="validationErrors().model_timeout ? 'timeout-help timeout-error' : 'timeout-help'" />
              <span id="timeout-help" class="settings-help">Maximum runtime for a newly started Hugging Face generation. A running generation keeps its captured timeout.</span>
              @if (validationErrors().model_timeout; as error) { <span id="timeout-error" class="settings-error">{{ error }}</span> }
            </div>
          </section>

          @if (saveError() || resetError()) {
            <div class="settings-feedback settings-feedback-error" role="alert">{{ saveError() || resetError() }}</div>
          }
          @if (statusMessage()) {
            <div class="settings-feedback settings-feedback-success" role="status">{{ statusMessage() }}</div>
          }
          <div class="settings-actions">
            <button type="button" class="secondary-button" (click)="reset()" [disabled]="saving() || resetting()">
              {{ resetting() ? 'Resetting…' : 'Reset to defaults' }}
            </button>
            <button type="submit" class="primary-button" [disabled]="saving() || resetting() || !dirty() || hasValidationErrors()">
              {{ saving() ? 'Saving…' : 'Save changes' }}
            </button>
          </div>
        </form>
      }
    </main>
  `,
  styleUrl: '../styles/SettingsPage.css',
})
export class SettingsPage {
  private readonly api = inject(SettingsApiService);
  readonly loading = signal(true);
  readonly saving = signal(false);
  readonly resetting = signal(false);
  readonly loadError = signal<string | null>(null);
  readonly saveError = signal<string | null>(null);
  readonly resetError = signal<string | null>(null);
  readonly statusMessage = signal<string | null>(null);
  readonly draft = signal<EditableSettings | null>(null);
  private readonly baseline = signal<EditableSettings | null>(null);
  readonly defaults = signal<EditableSettings | null>(null);

  readonly dirty = computed(() => {
    const current = this.draft();
    const original = this.baseline();
    return current !== null && original !== null && JSON.stringify(current) !== JSON.stringify(original);
  });

  readonly validationErrors = computed<ValidationErrors>(() => {
    const values = this.draft();
    if (!values) return {};
    const errors: ValidationErrors = {};
    if (!Number.isInteger(values.global.seed) || values.global.seed < 0 || values.global.seed > 4_294_967_295) {
      errors.seed = 'Enter an integer from 0 to 4,294,967,295.';
    }
    if (!Number.isFinite(values.jobs.polling_interval) || values.jobs.polling_interval < 0.25 || values.jobs.polling_interval > 60) {
      errors.polling_interval = 'Enter a value from 0.25 to 60 seconds.';
    }
    if (!Number.isInteger(values.inference.model_timeout) || values.inference.model_timeout < 1) {
      errors.model_timeout = 'Enter an integer of at least 1 second.';
    }
    return errors;
  });

  constructor() {
    void this.load();
  }

  hasValidationErrors(): boolean {
    return Object.keys(this.validationErrors()).length > 0;
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.loadError.set(null);
    const response = await this.api.getSettings();
    if (response.result) {
      this.applyResponse(response.result);
    } else {
      this.loadError.set(response.error ?? 'Unable to load application settings.');
    }
    this.loading.set(false);
  }

  updateSeed(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, global: { seed: value ?? Number.NaN } }));
  }

  updateFilesystemAccess(value: boolean): void {
    this.updateDraft((draft) => ({ ...draft, features: { allow_local_filesystem_access: value } }));
  }

  updatePollingInterval(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, jobs: { polling_interval: value ?? Number.NaN } }));
  }

  updateModelTimeout(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, inference: { model_timeout: value ?? Number.NaN } }));
  }

  async save(): Promise<void> {
    const current = this.draft();
    const original = this.baseline();
    if (!current || !original || !this.dirty() || this.hasValidationErrors()) return;
    this.saving.set(true);
    this.saveError.set(null);
    this.resetError.set(null);
    this.statusMessage.set(null);
    const patch = this.buildPatch(current, original);
    const response = await this.api.updateSettings(patch);
    if (response.result) {
      this.applyResponse(response.result);
      this.statusMessage.set('Settings saved.');
    } else {
      this.saveError.set(response.error ?? 'Unable to save application settings.');
    }
    this.saving.set(false);
  }

  async reset(): Promise<void> {
    this.resetting.set(true);
    this.resetError.set(null);
    this.saveError.set(null);
    this.statusMessage.set(null);
    const response = await this.api.resetSettings();
    if (response.result) {
      this.applyResponse(response.result);
      this.statusMessage.set('Settings reset to defaults.');
    } else {
      this.resetError.set(response.error ?? 'Unable to reset application settings.');
    }
    this.resetting.set(false);
  }

  private updateDraft(update: (draft: EditableSettings) => EditableSettings): void {
    const current = this.draft();
    if (current) this.draft.set(update(current));
    this.statusMessage.set(null);
    this.saveError.set(null);
    this.resetError.set(null);
  }

  private applyResponse(response: ApplicationSettingsResponse): void {
    const values = this.toEditable(response.values);
    this.draft.set(values);
    this.baseline.set(values);
    this.defaults.set(this.toEditable(response.defaults));
  }

  private toEditable(values: ApplicationSettingsResponse['values']): EditableSettings {
    return {
      global: { seed: values.global.seed },
      features: { allow_local_filesystem_access: values.features.allow_local_filesystem_access },
      jobs: { polling_interval: values.jobs.polling_interval },
      inference: { model_timeout: values.inference.model_timeout },
    };
  }

  private buildPatch(current: EditableSettings, original: EditableSettings): ApplicationSettingsPatch {
    const patch: ApplicationSettingsPatch = {};
    if (current.global.seed !== original.global.seed) patch.global = { seed: current.global.seed };
    if (current.features.allow_local_filesystem_access !== original.features.allow_local_filesystem_access) {
      patch.features = { allow_local_filesystem_access: current.features.allow_local_filesystem_access };
    }
    if (current.jobs.polling_interval !== original.jobs.polling_interval) {
      patch.jobs = { polling_interval: current.jobs.polling_interval };
    }
    if (current.inference.model_timeout !== original.inference.model_timeout) {
      patch.inference = { model_timeout: current.inference.model_timeout };
    }
    return patch;
  }
}
