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

type SettingsTab = 'general' | 'data' | 'advanced';

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
          <nav class="settings-tabs" role="tablist" aria-label="Settings categories">
            @for (tab of tabs; track tab.id) {
              <button type="button" class="settings-tab" [class.active]="activeTab() === tab.id" role="tab"
                [id]="'settings-tab-' + tab.id" [attr.aria-selected]="activeTab() === tab.id"
                [attr.aria-controls]="'settings-panel-' + tab.id" [attr.tabindex]="activeTab() === tab.id ? 0 : -1"
                (click)="selectTab(tab.id)">{{ tab.label }}</button>
            }
          </nav>

          <section class="settings-panel" role="tabpanel" [id]="'settings-panel-' + activeTab()"
            [attr.aria-labelledby]="'settings-tab-' + activeTab()" tabindex="0">
            @if (activeTab() === 'general') {
              <div class="settings-section-heading">
                <h2>General</h2>
                <p>Defaults used when new work starts without an explicit value.</p>
              </div>
              <div class="settings-fields">
                <div class="settings-field">
                  <div class="settings-field-copy"><h3 id="default-seed-title">Default random seed</h3><p>Controls repeatability for new preparation, training, or validation work.</p></div>
                  <div class="settings-field-control">
                    <input id="default-seed" type="number" inputmode="numeric" step="1" min="0" max="4294967295"
                      [ngModel]="values.global.seed" (ngModelChange)="updateSeed($event)" name="seed"
                      [disabled]="saving() || resetting()" [attr.aria-labelledby]="'default-seed-title'"
                      [attr.aria-invalid]="validationErrors().seed ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().seed ? 'seed-help seed-error' : 'seed-help'" />
                    <span id="seed-help" class="settings-help">Running jobs keep the seed they started with.</span>
                    @if (validationErrors().seed; as error) { <span id="seed-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
              </div>
            } @else if (activeTab() === 'data') {
              <div class="settings-section-heading">
                <h2>Data access</h2>
                <p>Control whether dataset workflows may access local paths.</p>
              </div>
              <div class="settings-fields">
                <div class="settings-field">
                  <div class="settings-field-copy"><h3 id="local-filesystem-access-title">Allow local filesystem access</h3><p>Allow dataset workflows to read and write configured local paths.</p></div>
                  <div class="settings-field-control">
                    <label class="settings-checkbox-control" for="local-filesystem-access"><input id="local-filesystem-access" type="checkbox"
                      [ngModel]="values.features.allow_local_filesystem_access" (ngModelChange)="updateFilesystemAccess($event)"
                      name="allow_local_filesystem_access" [disabled]="saving() || resetting()"
                      aria-describedby="filesystem-access-help" /><span>Enabled</span></label>
                    <span id="filesystem-access-help" class="settings-help">Turning this off blocks new local filesystem operations but does not remove data already imported into XREPORT.</span>
                  </div>
                </div>
              </div>
            } @else {
              <div class="settings-section-heading">
                <h2>Advanced</h2>
                <p>Fine-tune polling and local inference operation limits.</p>
              </div>
              <div class="settings-fields">
                <div class="settings-field">
                  <div class="settings-field-copy"><h3 id="polling-interval-title">Job status polling interval <span class="settings-unit">seconds</span></h3><p>How often the UI checks newly started long-running jobs.</p></div>
                  <div class="settings-field-control">
                    <input id="polling-interval" type="number" inputmode="decimal" step="0.25" min="0.25" max="60"
                      [ngModel]="values.jobs.polling_interval" (ngModelChange)="updatePollingInterval($event)" name="polling_interval"
                      [disabled]="saving() || resetting()" [attr.aria-labelledby]="'polling-interval-title'"
                      [attr.aria-invalid]="validationErrors().polling_interval ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().polling_interval ? 'polling-help polling-error' : 'polling-help'" />
                    <span id="polling-help" class="settings-help">Existing jobs keep their current interval.</span>
                    @if (validationErrors().polling_interval; as error) { <span id="polling-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
                <div class="settings-field">
                  <div class="settings-field-copy"><h3 id="inference-timeout-title">Inference timeout <span class="settings-unit">seconds</span></h3><p>Maximum runtime for a newly started local generation.</p></div>
                  <div class="settings-field-control">
                    <input id="inference-timeout" type="number" inputmode="numeric" step="1" min="1"
                      [ngModel]="values.inference.model_timeout" (ngModelChange)="updateModelTimeout($event)" name="model_timeout"
                      [disabled]="saving() || resetting()" [attr.aria-labelledby]="'inference-timeout-title'"
                      [attr.aria-invalid]="validationErrors().model_timeout ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().model_timeout ? 'timeout-help timeout-error' : 'timeout-help'" />
                    <span id="timeout-help" class="settings-help">A running generation keeps its captured timeout.</span>
                    @if (validationErrors().model_timeout; as error) { <span id="timeout-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
              </div>
            }
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
  readonly tabs = [
    { id: 'general' as const, label: 'General' },
    { id: 'data' as const, label: 'Data access' },
    { id: 'advanced' as const, label: 'Advanced' },
  ];
  readonly activeTab = signal<SettingsTab>('general');

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

  selectTab(tab: SettingsTab): void {
    this.activeTab.set(tab);
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
