import { CommonModule } from '@angular/common';
import { Component, computed, EventEmitter, HostListener, inject, Input, OnDestroy, OnInit, Output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import type { Subscription } from 'rxjs';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import { SettingsApiService } from '../services/settings-api.service';
import { ThemeService } from '../services/theme.service';
import type {
  ApplicationSettingsPatch,
  ApplicationSettingsResponse,
  DatabaseConfigurationDraft,
  DatabaseConfigurationSummary,
  DatabaseConfigurationStatus,
  DatabaseConnectionTest,
  DatabaseBackend,
  EditableSettings,
  HuggingFaceAccessStatus,
  StorageStatus,
  StorageValidation,
} from '../types/settingsApi';

interface ValidationErrors {
  seed?: string;
  polling_interval?: string;
  model_timeout?: string;
  model_idle_timeout?: string;
}

type SettingsTab = 'general' | 'advanced' | 'models' | 'storage' | 'database';

@Component({
  selector: 'app-settings-page',
  imports: [CommonModule, FormsModule],
  template: `
    <main class="settings-page page-container" aria-labelledby="settings-title">
      <header class="settings-header">
        <h1 id="settings-title">Settings</h1>
      </header>

      @if (databaseOnly) {
        <ng-container [ngTemplateOutlet]="databasePanel" />
      } @else if (loading()) {
        <section class="settings-state" role="status">Loading settings…</section>
      } @else if (loadError()) {
        <section class="settings-state settings-state-error" role="alert">
          <strong>Settings could not be loaded</strong>
          <span>{{ loadError() }}</span>
          <button type="button" class="secondary-button" (click)="load()">Try again</button>
        </section>
        <ng-container [ngTemplateOutlet]="databasePanel" />
      } @else if (draft(); as values) {
        <div class="settings-form">
          <nav class="settings-tabs" role="tablist" aria-label="Settings categories">
            @for (tab of tabs; track tab.id) {
              <button type="button" class="settings-tab" [class.active]="activeTab() === tab.id" role="tab"
                [id]="'settings-tab-' + tab.id" [attr.aria-selected]="activeTab() === tab.id"
                [attr.aria-controls]="'settings-panel-' + tab.id" [attr.tabindex]="activeTab() === tab.id ? 0 : -1"
                (click)="selectTab(tab.id)" (keydown)="onTabKeydown($event, tab.id)">{{ tab.label }}</button>
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
                      [disabled]="resetting() || databaseSwitching()" [attr.aria-labelledby]="'default-seed-title'"
                      [attr.aria-invalid]="validationErrors().seed ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().seed ? 'seed-help seed-error' : 'seed-help'"
                      (blur)="flushSettingsPersistence()" />
                    <span id="seed-help" class="settings-help">Running jobs keep the seed they started with.</span>
                    @if (validationErrors().seed; as error) { <span id="seed-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
              </div>
            } @else if (activeTab() === 'advanced') {
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
                      [disabled]="resetting() || databaseSwitching()" [attr.aria-labelledby]="'polling-interval-title'"
                      [attr.aria-invalid]="validationErrors().polling_interval ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().polling_interval ? 'polling-help polling-error' : 'polling-help'"
                      (blur)="flushSettingsPersistence()" />
                    <span id="polling-help" class="settings-help">Existing jobs keep their current interval.</span>
                    @if (validationErrors().polling_interval; as error) { <span id="polling-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
                <div class="settings-field">
                  <div class="settings-field-copy"><h3 id="inference-timeout-title">Inference timeout <span class="settings-unit">seconds</span></h3><p>Maximum runtime for a newly started local generation.</p></div>
                  <div class="settings-field-control">
                    <input id="inference-timeout" type="number" inputmode="numeric" step="1" min="1"
                      [ngModel]="values.inference.model_timeout" (ngModelChange)="updateModelTimeout($event)" name="model_timeout"
                      [disabled]="resetting() || databaseSwitching()" [attr.aria-labelledby]="'inference-timeout-title'"
                      [attr.aria-invalid]="validationErrors().model_timeout ? 'true' : null"
                      [attr.aria-describedby]="validationErrors().model_timeout ? 'timeout-help timeout-error' : 'timeout-help'"
                      (blur)="flushSettingsPersistence()" />
                    <span id="timeout-help" class="settings-help">A running generation keeps its captured timeout.</span>
                    @if (validationErrors().model_timeout; as error) { <span id="timeout-error" class="settings-error">{{ error }}</span> }
                  </div>
                </div>
                @if (!studio) {
                  <div class="settings-field">
                    <div class="settings-field-copy"><h3 id="idle-unload-title">Unload idle models after <span class="settings-unit">seconds</span></h3><p>Releases a loaded model's RAM and VRAM once no inference has used it for this long.</p></div>
                    <div class="settings-field-control">
                      <input id="idle-unload" type="number" inputmode="numeric" step="1" min="0"
                        [ngModel]="values.inference.model_idle_timeout" (ngModelChange)="updateModelIdleTimeout($event)" name="model_idle_timeout"
                        [disabled]="resetting() || databaseSwitching()" [attr.aria-labelledby]="'idle-unload-title'"
                        [attr.aria-invalid]="validationErrors().model_idle_timeout ? 'true' : null"
                        [attr.aria-describedby]="validationErrors().model_idle_timeout ? 'idle-unload-help idle-unload-error' : 'idle-unload-help'"
                        (blur)="flushSettingsPersistence()" />
                      <span id="idle-unload-help" class="settings-help">0 keeps the model loaded. Inference studies are never cleared.</span>
                      @if (validationErrors().model_idle_timeout; as error) { <span id="idle-unload-error" class="settings-error">{{ error }}</span> }
                    </div>
                  </div>
                }
              </div>
            } @else if (activeTab() === 'models') {
              <div class="settings-section-heading">
                <h2>Model access</h2>
                <p>Gated public models, such as MedGemma, need a Hugging Face access token before they can be downloaded.</p>
              </div>
              <div class="settings-fields">
                <div class="settings-field">
                  <div class="settings-field-copy">
                    <h3 id="hf-token-title">Hugging Face access token</h3>
                    <p>Accept the model terms on its Hugging Face page first, then create a read token in your Hugging Face account settings. The token is stored on this computer by the backend and is never shown again.</p>
                  </div>
                  <div class="settings-field-control">
                    @if (huggingFaceLoadError(); as error) {
                      <span class="settings-error" role="alert">{{ error }}</span>
                    } @else if (huggingFaceStatus(); as access) {
                      <span id="hf-token-status" class="settings-help" role="status">{{ huggingFaceStatusLabel(access) }}</span>
                    }
                    <input id="hf-token" type="password" autocomplete="off" spellcheck="false" name="hf_token" placeholder="hf_…"
                      [(ngModel)]="huggingFaceTokenInput" [disabled]="huggingFaceBusy() || huggingFaceStatus()?.source === 'environment'"
                      aria-labelledby="hf-token-title" aria-describedby="hf-token-status" />
                    <div class="settings-inline-actions">
                      <button type="button" class="secondary-button" (click)="saveHuggingFaceToken()"
                        [disabled]="huggingFaceBusy() || !huggingFaceTokenInput.trim() || huggingFaceStatus()?.source === 'environment'">
                        {{ huggingFaceBusy() ? 'Working…' : 'Save token' }}
                      </button>
                      @if (huggingFaceStatus()?.source === 'stored') {
                        <button type="button" class="secondary-button" (click)="clearHuggingFaceToken()" [disabled]="huggingFaceBusy()">Remove token</button>
                      }
                    </div>
                    @if (huggingFaceActionError(); as error) { <span class="settings-error" role="alert">{{ error }}</span> }
                    @if (huggingFaceActionMessage(); as message) { <span class="settings-help" role="status">{{ message }}</span> }
                  </div>
                </div>
              </div>
            } @else if (activeTab() === 'storage') {
              <div class="settings-section-heading">
                <h2>Storage location</h2>
                <p>Downloaded models, checkpoints, the application database, templates and logs live under one folder. Changes take effect after the application restarts.</p>
              </div>
              <div class="storage-layout">
                @if (storageLoadError(); as error) {
                  <span class="settings-error" role="alert">{{ error }}</span>
                } @else if (storageStatus(); as storage) {
                  <div class="storage-overview">
                    <section class="storage-block" aria-labelledby="storage-root-title">
                      <h3 id="storage-root-title">Current location</h3>
                      <code id="storage-active-root" class="storage-path">{{ storage.active_root }}</code>
                      @if (storage.restart_required) {
                        <span class="settings-help" role="status">Restart required: the application will use {{ storage.pending_root }} after it restarts.</span>
                      }
                      @if (storage.locked) {
                        <span class="settings-help" role="status">The location is set by XREPORT_RESOURCES_DIR in the environment or settings/.env and can only be changed there.</span>
                      }
                    </section>
                    <section class="storage-block" aria-labelledby="storage-usage-title">
                      <h3 id="storage-usage-title">Usage <span class="settings-unit">{{ formatBytes(totalBytes(storage)) }} total</span></h3>
                      <ul class="storage-usage" aria-label="Storage folders">
                        @for (folder of storage.folders; track folder.name) {
                          <li><span>{{ folder.name }}</span><span class="storage-size">{{ formatBytes(folder.size_bytes) }}</span></li>
                        }
                      </ul>
                    </section>
                  </div>
                }
                <section class="storage-block storage-destination" aria-labelledby="storage-destination-title">
                  <h3 id="storage-destination-title">New destination</h3>
                  <p class="settings-help">Existing data is copied to the new folder and verified first. The old folder is left untouched; delete it yourself once you no longer need it.</p>
                  <input id="storage-path" type="text" spellcheck="false" name="storage_path" placeholder="D:\\XREPORT-data"
                    [ngModel]="storagePathInput()" (ngModelChange)="onStoragePathChange($event)"
                    [disabled]="storageBusy() || storageStatus()?.locked !== false" aria-labelledby="storage-destination-title" />
                  @if (storageValidation(); as check) {
                    <div class="storage-feedback">
                      @for (error of check.errors; track error) { <span class="settings-error" role="alert">{{ error }}</span> }
                      @for (warning of check.warnings; track warning) { <span class="settings-help" role="status">{{ warning }}</span> }
                      @if (check.valid && check.target_state !== 'existing') {
                        <span class="settings-help" role="status">{{ formatBytes(check.required_bytes) }} will be copied; {{ formatBytes(check.free_bytes ?? 0) }} free.</span>
                      }
                    </div>
                  }
                  @if (storageBusy()) {
                    <span class="settings-help" role="status">{{ storageProgressMessage() }} ({{ storageProgress() | number:'1.0-0' }}%)</span>
                  }
                  @if (storageActionError(); as error) { <span class="settings-error" role="alert">{{ error }}</span> }
                  <div class="storage-actions">
                    <button type="button" class="secondary-button" (click)="validateStorage()"
                      [disabled]="storageBusy() || !storagePathInput().trim() || storageStatus()?.locked !== false">Check folder</button>
                    <button type="button" class="primary-button" (click)="applyStorage()"
                      [disabled]="storageBusy() || !storageValidation()?.valid">
                      {{ storageBusy() ? 'Moving…' : storageValidation()?.target_state === 'existing' ? 'Use this folder' : 'Move data here' }}
                    </button>
                  </div>
                </section>
              </div>
            } @else {
              <ng-container [ngTemplateOutlet]="databasePanel" />
            }
          </section>

          @if (saveError() || resetError()) {
            <div class="settings-feedback settings-feedback-error" role="alert">{{ saveError() || resetError() }}</div>
          }
          @if (hasValidationErrors()) {
            <div class="settings-feedback settings-feedback-error" role="status">Correct the highlighted values before they can be saved.</div>
          }
          @if (saving()) { <div class="settings-save-status" role="status">Saving changes…</div> }
          @if (statusMessage()) {
            <div class="settings-save-status settings-save-status-success" role="status">{{ statusMessage() }}</div>
          }
          @if (activeTab() !== 'database' && activeTab() !== 'models' && activeTab() !== 'storage') {
            <div class="settings-actions">
              <button type="button" class="secondary-button" (click)="reset()" [disabled]="resetting() || databaseSwitching()">
                {{ resetting() ? 'Resetting…' : 'Reset to defaults' }}
              </button>
            </div>
          }
        </div>
      }
    </main>
    <ng-template #databasePanel>
      @if (databaseLoading()) {
        <section class="settings-state" role="status">Loading database settings…</section>
      } @else {
        <section class="database-panel" aria-labelledby="database-settings-title">
          <div class="settings-section-heading">
            <h2 id="database-settings-title">Database</h2>
            <p>Choose where the application stores application data. Each database keeps its own settings and records.</p>
          </div>
          <div class="database-status" [class.database-status-error]="databaseStatus()?.state !== 'ready'" role="status">
            <span class="database-status-indicator" aria-hidden="true"></span>
            <div><strong>{{ databaseStatusLabel() }}</strong><span>{{ databaseStatusDescription() }}</span></div>
          </div>
          @if (databaseLoadError()) {
            <div class="settings-feedback settings-feedback-error" role="alert">
              {{ databaseLoadError() }}
              <button type="button" class="secondary-button" (click)="loadDatabaseStatus()">Try again</button>
            </div>
          }
          @if (databaseStatus()?.error; as error) { <div class="settings-feedback settings-feedback-error" role="alert">{{ error }}</div> }

          @if (databaseDraft(); as db) {
            <div class="database-fields">
              <div class="database-field">
                <label for="database-backend">Database backend</label>
                <select id="database-backend" name="database_backend" [ngModel]="db.backend"
                  (ngModelChange)="updateDatabaseBackend($event)" [ngModelOptions]="{standalone: true}"
                  [disabled]="databaseSwitching()">
                  <option value="sqlite">SQLite</option><option value="postgresql">PostgreSQL</option>
                </select>
                <span class="settings-help">SQLite stores a local database file. PostgreSQL connects through the installed psycopg driver.</span>
              </div>

              @if (db.backend === 'sqlite') {
                <div class="database-field">
                  <label for="database-sqlite-path">Database file path</label>
                  <input id="database-sqlite-path" name="database_sqlite_path" type="text" autocomplete="off"
                    [ngModel]="db.sqlite_path" (ngModelChange)="updateDatabaseDraft({sqlite_path: $event})"
                    [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()"
                    (keydown.enter)="$event.preventDefault()" />
                  <span class="settings-help">Use an absolute path. A new file is initialized when you activate it.</span>
                </div>
              } @else {
                <div class="database-grid">
                  <div class="database-field">
                    <label for="database-host">Host</label>
                    <input id="database-host" name="database_host" type="text" autocomplete="off" required
                      [ngModel]="db.host" (ngModelChange)="updateDatabaseDraft({host: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field">
                    <label for="database-port">Port</label>
                    <input id="database-port" name="database_port" type="number" min="1" max="65535" step="1"
                      [ngModel]="db.port" (ngModelChange)="updateDatabaseDraft({port: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field">
                    <label for="database-name">Database name</label>
                    <input id="database-name" name="database_name" type="text" autocomplete="off" required
                      [ngModel]="db.database_name" (ngModelChange)="updateDatabaseDraft({database_name: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field">
                    <label for="database-username">Username</label>
                    <input id="database-username" name="database_username" type="text" autocomplete="username" required
                      [ngModel]="db.username" (ngModelChange)="updateDatabaseDraft({username: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field">
                    <label for="database-password">Password</label>
                    <input id="database-password" name="database_password" type="password" autocomplete="new-password"
                      [placeholder]="databaseStatus()?.active?.password_configured ? 'Saved securely; leave blank to keep it' : 'Optional for this server'"
                      [ngModel]="db.password" (ngModelChange)="updateDatabaseDraft({password: $event || null})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                    <span class="settings-help">Saved passwords are protected with Windows DPAPI and are never returned to this page.</span>
                  </div>
                  <div class="database-field">
                    <label for="database-timeout">Connection timeout <span class="settings-unit">seconds</span></label>
                    <input id="database-timeout" name="database_timeout" type="number" min="1" max="300" step="1"
                      [ngModel]="db.connect_timeout" (ngModelChange)="updateDatabaseDraft({connect_timeout: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field">
                    <label for="database-batch-size">Insert batch size</label>
                    <input id="database-batch-size" name="database_batch_size" type="number" min="1" step="1"
                      [ngModel]="db.insert_batch_size" (ngModelChange)="updateDatabaseDraft({insert_batch_size: $event})"
                      [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                  </div>
                  <div class="database-field database-field-toggle">
                    <label class="settings-checkbox-control" for="database-ssl">
                      <input id="database-ssl" name="database_ssl" type="checkbox" [ngModel]="db.ssl"
                        (ngModelChange)="updateDatabaseDraft({ssl: $event})" [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" />
                      <span>Require SSL</span>
                    </label>
                  </div>
                  @if (db.ssl) {
                    <div class="database-field database-field-wide">
                      <label for="database-ssl-ca">CA certificate path <span class="settings-unit">optional</span></label>
                      <input id="database-ssl-ca" name="database_ssl_ca" type="text" autocomplete="off"
                        [ngModel]="db.ssl_ca" (ngModelChange)="updateDatabaseDraft({ssl_ca: $event})"
                        [ngModelOptions]="{standalone: true}" [disabled]="databaseSwitching()" (keydown.enter)="$event.preventDefault()" />
                    </div>
                  }
                </div>
              }

              @if (databaseValidationError()) { <div class="settings-feedback settings-feedback-error" role="alert">{{ databaseValidationError() }}</div> }
              @if (databaseTestResult(); as result) {
                <div class="settings-feedback" [class.settings-feedback-success]="result.success && result.schema_ready"
                  [class.settings-feedback-error]="!result.success" role="status">{{ result.detail }}</div>
              }
              @if (databaseActionError()) { <div class="settings-feedback settings-feedback-error" role="alert">{{ databaseActionError() }}</div> }
              @if (databaseActionMessage()) { <div class="settings-feedback settings-feedback-success" role="status">{{ databaseActionMessage() }}</div> }
              <div class="database-actions">
                <button type="button" class="secondary-button" (click)="testDatabaseConnection()"
                  [disabled]="databaseTesting() || databaseSwitching() || !!databaseValidationError()">
                  {{ databaseTesting() ? 'Testing connection…' : 'Test connection' }}
                </button>
                <button type="button" class="primary-button" (click)="switchDatabase()"
                  [disabled]="databaseSwitching() || databaseTesting() || saving() || dirty() || hasValidationErrors() || databaseStatus()?.jobs_running || !!databaseValidationError() || !databaseCanSwitch() || !databaseTestResult()?.success">
                  {{ databaseSwitching() ? 'Switching database…' : (databaseOnly ? 'Activate database' : 'Apply database settings') }}
                </button>
              </div>
              @if (databaseCanSwitch() && !databaseTestResult()?.success && !databaseValidationError()) {
                <p class="settings-help database-job-note">Test the connection before applying this database configuration.</p>
              }
              @if (saving() || dirty() || hasValidationErrors()) {
                <p class="settings-help database-job-note">Wait for pending application settings to save or correct invalid values before switching databases.</p>
              }
              @if (databaseStatus()?.jobs_running) {
                <p class="settings-help database-job-note">Database switching is unavailable while a background job is running.</p>
              }
            </div>
          }
        </section>
      }
    </ng-template>
  `,
  styleUrl: '../styles/SettingsPage.css',
})
export class SettingsPage implements OnInit, OnDestroy {
  private readonly api = inject(SettingsApiService);
  private readonly theme = inject(ThemeService);
  // Idle model release is an XREPORT inference control; Studio shares this page without it
  readonly studio = inject(ActivatedRoute, { optional: true })?.snapshot.pathFromRoot.some((route) => route.data['application'] === 'studio') ?? false;
  @Input() databaseOnly = false;
  @Output() readonly databaseActivated = new EventEmitter<void>();
  readonly loading = signal(true);
  readonly saving = signal(false);
  readonly resetting = signal(false);
  readonly loadError = signal<string | null>(null);
  readonly saveError = signal<string | null>(null);
  readonly resetError = signal<string | null>(null);
  readonly statusMessage = signal<string | null>(null);
  readonly databaseLoading = signal(true);
  readonly databaseTesting = signal(false);
  readonly databaseSwitching = signal(false);
  readonly databaseLoadError = signal<string | null>(null);
  readonly databaseActionError = signal<string | null>(null);
  readonly databaseActionMessage = signal<string | null>(null);
  readonly databaseStatus = signal<DatabaseConfigurationStatus | null>(null);
  readonly databaseDraft = signal<DatabaseConfigurationDraft | null>(null);
  readonly databaseTestResult = signal<DatabaseConnectionTest | null>(null);
  readonly huggingFaceStatus = signal<HuggingFaceAccessStatus | null>(null);
  readonly huggingFaceBusy = signal(false);
  readonly huggingFaceLoadError = signal<string | null>(null);
  readonly huggingFaceActionError = signal<string | null>(null);
  readonly huggingFaceActionMessage = signal<string | null>(null);
  huggingFaceTokenInput = '';
  private readonly jobsApi = inject(JobsApiService);
  private readonly polling = inject(JobPollingService);
  private storagePoll: Subscription | null = null;
  readonly storageStatus = signal<StorageStatus | null>(null);
  readonly storageLoadError = signal<string | null>(null);
  readonly storagePathInput = signal('');
  readonly storageValidation = signal<StorageValidation | null>(null);
  readonly storageBusy = signal(false);
  readonly storageProgress = signal(0);
  readonly storageProgressMessage = signal('Starting…');
  readonly storageActionError = signal<string | null>(null);
  readonly draft = signal<EditableSettings | null>(null);
  private readonly baseline = signal<EditableSettings | null>(null);
  readonly defaults = signal<EditableSettings | null>(null);
  readonly tabs = [
    { id: 'general' as const, label: 'General' },
    { id: 'advanced' as const, label: 'Advanced' },
    { id: 'models' as const, label: 'Model access' },
    { id: 'storage' as const, label: 'Storage' },
    { id: 'database' as const, label: 'Database' },
  ];
  readonly activeTab = signal<SettingsTab>('general');
  private readonly autosaveDelayMs = 350;
  private autosaveTimer: ReturnType<typeof setTimeout> | null = null;
  private settingsRevision = 0;
  private persistenceQueue: Promise<void> = Promise.resolve();

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
    if (!Number.isInteger(values.inference.model_idle_timeout) || values.inference.model_idle_timeout < 0) {
      errors.model_idle_timeout = 'Enter an integer of at least 0 seconds.';
    }
    return errors;
  });

  ngOnInit(): void {
    if (!this.databaseOnly) void this.load();
    void this.loadDatabaseStatus();
    if (!this.databaseOnly) void this.loadHuggingFaceStatus();
    if (!this.databaseOnly) void this.loadStorageStatus();
  }

  async loadStorageStatus(): Promise<void> {
    this.storageLoadError.set(null);
    const response = await this.api.getStorage();
    if (response.result) this.storageStatus.set(response.result);
    else this.storageLoadError.set(response.error ?? 'Unable to load the storage location.');
  }

  onStoragePathChange(value: string): void {
    this.storagePathInput.set(value);
    this.storageValidation.set(null);
    this.storageActionError.set(null);
  }

  async validateStorage(): Promise<void> {
    this.storageActionError.set(null);
    const response = await this.api.validateStorage(this.storagePathInput().trim());
    if (response.result) this.storageValidation.set(response.result);
    else this.storageActionError.set(response.error ?? 'Unable to check this folder.');
  }

  async applyStorage(): Promise<void> {
    const check = this.storageValidation();
    if (!check?.valid || this.storageBusy()) return;
    this.storageBusy.set(true);
    this.storageProgress.set(0);
    this.storageProgressMessage.set('Starting…');
    this.storageActionError.set(null);
    const started = await this.api.applyStorage(check.resolved_path, check.target_state === 'existing' ? 'use_existing' : 'copy');
    if (!started.result) {
      this.storageBusy.set(false);
      this.storageActionError.set(started.error ?? 'Unable to start moving the data.');
      return;
    }
    this.storagePoll = this.polling.poll((id) => this.jobsApi.get(id), started.result.job_id, started.result.poll_interval ?? 1).subscribe((status) => {
      this.storageProgress.set(status.progress ?? 0);
      const lifecycle = status.result?.['lifecycle'] as { message?: string } | undefined;
      if (lifecycle?.message) this.storageProgressMessage.set(lifecycle.message);
      if (!['completed', 'failed', 'cancelled'].includes(status.status)) return;
      this.storageBusy.set(false);
      if (status.status !== 'completed') this.storageActionError.set(status.error ?? 'Moving the data did not complete. The current location is unchanged.');
      else this.storageValidation.set(null);
      void this.loadStorageStatus();
    });
  }

  totalBytes(storage: StorageStatus): number {
    return storage.folders.reduce((total, folder) => total + folder.size_bytes, 0);
  }

  formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
    return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
  }

  async loadHuggingFaceStatus(): Promise<void> {
    this.huggingFaceLoadError.set(null);
    const response = await this.api.getHuggingFaceAccess();
    if (response.result) this.huggingFaceStatus.set(response.result);
    else this.huggingFaceLoadError.set(response.error ?? 'Unable to load Hugging Face access status.');
  }

  huggingFaceStatusLabel(access: HuggingFaceAccessStatus): string {
    if (access.source === 'environment') return 'A token is set through the HF_TOKEN environment variable and takes precedence. Remove it there to manage the token here.';
    return access.configured ? 'A token is saved on this computer.' : 'No token is configured.';
  }

  async saveHuggingFaceToken(): Promise<void> {
    await this.runHuggingFaceAction(
      () => this.api.saveHuggingFaceToken(this.huggingFaceTokenInput.trim()),
      'Token verified and saved.',
    );
    if (!this.huggingFaceActionError()) this.huggingFaceTokenInput = '';
  }

  async clearHuggingFaceToken(): Promise<void> {
    await this.runHuggingFaceAction(() => this.api.clearHuggingFaceToken(), 'Token removed.');
  }

  private async runHuggingFaceAction(
    action: () => Promise<{ result?: HuggingFaceAccessStatus | null; error?: string | null }>,
    successMessage: string,
  ): Promise<void> {
    this.huggingFaceBusy.set(true);
    this.huggingFaceActionError.set(null);
    this.huggingFaceActionMessage.set(null);
    const response = await action();
    if (response.result) {
      this.huggingFaceStatus.set(response.result);
      this.huggingFaceActionMessage.set(successMessage);
    } else {
      this.huggingFaceActionError.set(response.error ?? 'Hugging Face token update failed.');
    }
    this.huggingFaceBusy.set(false);
  }

  ngOnDestroy(): void {
    this.storagePoll?.unsubscribe();
    void this.flushSettingsPersistence();
  }

  @HostListener('window:pagehide')
  onPageHide(): void {
    void this.flushSettingsPersistence();
  }

  async loadDatabaseStatus(): Promise<void> {
    this.databaseLoading.set(true);
    this.databaseLoadError.set(null);
    const response = await this.api.getDatabaseSettings();
    if (response.result) {
      this.databaseStatus.set(response.result);
      const initial = response.result.active ?? response.result.suggested;
      this.databaseDraft.set(this.configurationDraft(initial));
    } else {
      this.databaseLoadError.set(response.error ?? 'Unable to load database settings.');
    }
    this.databaseLoading.set(false);
  }

  updateDatabaseBackend(backend: DatabaseBackend): void {
    const current = this.databaseDraft();
    const status = this.databaseStatus();
    if (!current || !status) return;
    if (backend === current.backend) return;
    const matching = status.active?.backend === backend ? status.active : null;
    const base = this.configurationDraft(matching ?? status.suggested);
    this.databaseDraft.set({
      ...base,
      backend,
      sqlite_path: backend === 'sqlite' ? (matching?.sqlite_path ?? status.suggested.sqlite_path) : null,
      engine: backend === 'postgresql' ? 'postgresql+psycopg' : null,
      host: backend === 'postgresql' ? (matching?.host ?? '') : null,
      port: backend === 'postgresql' ? (matching?.port ?? 5432) : 5432,
      database_name: backend === 'postgresql' ? (matching?.database_name ?? '') : null,
      username: backend === 'postgresql' ? (matching?.username ?? '') : null,
      password: null,
    });
    this.clearDatabaseFeedback();
  }

  updateDatabaseDraft(update: Partial<DatabaseConfigurationDraft>): void {
    const current = this.databaseDraft();
    if (!current) return;
    this.databaseDraft.set({ ...current, ...update });
    this.clearDatabaseFeedback();
  }

  databaseValidationError(): string | null {
    const draft = this.databaseDraft();
    if (!draft) return 'Database settings are not available.';
    if (draft.backend === 'sqlite') {
      const path = draft.sqlite_path?.trim() ?? '';
      if (!path) return 'Enter a database file path.';
      if (!(/^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(path))) {
        return 'Enter an absolute database file path.';
      }
      return null;
    }
    if (!draft.host?.trim()) return 'Enter a PostgreSQL host.';
    if (!Number.isInteger(draft.port) || draft.port < 1 || draft.port > 65535) return 'Enter a port from 1 to 65,535.';
    if (!draft.database_name?.trim()) return 'Enter a PostgreSQL database name.';
    if (!draft.username?.trim()) return 'Enter a PostgreSQL username.';
    if (!Number.isInteger(draft.connect_timeout) || draft.connect_timeout < 1 || draft.connect_timeout > 300) {
      return 'Enter a connection timeout from 1 to 300 seconds.';
    }
    if (!Number.isInteger(draft.insert_batch_size) || draft.insert_batch_size < 1) return 'Enter an insert batch size of at least 1.';
    return null;
  }

  databaseStatusLabel(): string {
    const status = this.databaseStatus();
    if (!status) return 'Database status unavailable';
    if (status.state === 'recovery') return 'Database setup required';
    if (status.state === 'unavailable') return 'Database unavailable';
    return `Connected to ${status.active?.backend === 'postgresql' ? 'PostgreSQL' : 'SQLite'}`;
  }

  databaseStatusDescription(): string {
    const status = this.databaseStatus();
    if (!status) return 'Retry loading the saved database connection settings.';
    if (status.state === 'recovery') return 'Choose or repair a database connection to continue startup.';
    if (status.state === 'unavailable') return 'The saved connection could not be reached. Test a connection or select another database.';
    return status.active?.backend === 'postgresql'
      ? `${status.active.host}:${status.active.port}/${status.active.database_name}`
      : (status.active?.sqlite_path ?? 'Local database');
  }

  databaseCanSwitch(): boolean {
    const draft = this.databaseDraft();
    const status = this.databaseStatus();
    if (!draft || !status) return false;
    const active = status.active;
    if (!active || status.state !== 'ready' || !status.connected || !status.schema_ready) return true;
    const candidate = this.configurationDraft(active);
    const withoutSecret = (value: DatabaseConfigurationDraft) => ({ ...value, password: null });
    return JSON.stringify(withoutSecret(draft)) !== JSON.stringify(withoutSecret(candidate));
  }

  async testDatabaseConnection(): Promise<void> {
    const draft = this.databaseDraft();
    if (!draft || this.databaseValidationError()) return;
    this.databaseTesting.set(true);
    this.databaseActionError.set(null);
    this.databaseActionMessage.set(null);
    const response = await this.api.testDatabaseConnection(draft);
    if (response.result) {
      this.databaseTestResult.set(response.result);
    } else {
      this.databaseTestResult.set(null);
      this.databaseActionError.set(response.error ?? 'Unable to test this database connection.');
    }
    this.databaseTesting.set(false);
  }

  async switchDatabase(): Promise<void> {
    const draft = this.databaseDraft();
    if (!draft || this.databaseValidationError() || !this.databaseCanSwitch() || this.databaseTesting()
      || !this.databaseTestResult()?.success || this.saving() || this.dirty() || this.hasValidationErrors()) return;
    this.databaseSwitching.set(true);
    this.theme.setDatabaseSwitching(true);
    this.databaseActionError.set(null);
    this.databaseActionMessage.set(null);
    try {
      await this.flushSettingsPersistence();
      await this.theme.flushPersistence();
      if (this.saving() || this.dirty() || this.hasValidationErrors()) {
        this.databaseActionError.set('Save or correct pending application settings before switching databases.');
        return;
      }
      const response = await this.api.switchDatabase(draft);
      if (response.result) {
        const current = this.databaseStatus();
        this.databaseStatus.set(current ? {
          ...current,
          state: 'ready',
          active: response.result.active,
          connected: true,
          jobs_running: false,
          error: null,
        } : null);
        this.databaseDraft.set(this.configurationDraft(response.result.active));
        this.databaseTestResult.set({
          success: true,
          backend: response.result.active.backend,
          connected: true,
          schema_ready: true,
          detail: 'Database switched successfully. Application settings were loaded from the selected database.',
        });
        this.applyResponse(response.result.settings);
        this.theme.applyDatabasePreference(response.result.settings.values.theme_preference);
        this.databaseActionMessage.set('Database switched successfully.');
        this.databaseActivated.emit();
      } else {
        this.databaseActionError.set(response.error ?? 'Unable to switch databases. The previous connection remains active.');
      }
    } catch (error) {
      this.databaseActionError.set(error instanceof Error ? error.message : 'Unable to switch databases. The previous connection remains active.');
    } finally {
      this.databaseSwitching.set(false);
      this.theme.setDatabaseSwitching(false);
    }
  }

  private configurationDraft(summary: DatabaseConfigurationSummary): DatabaseConfigurationDraft {
    return {
      backend: summary.backend,
      sqlite_path: summary.sqlite_path,
      engine: summary.engine,
      host: summary.host,
      port: summary.port,
      database_name: summary.database_name,
      username: summary.username,
      password: null,
      ssl: summary.ssl,
      ssl_ca: summary.ssl_ca,
      connect_timeout: summary.connect_timeout,
      insert_batch_size: summary.insert_batch_size,
    };
  }

  private clearDatabaseFeedback(): void {
    this.databaseTestResult.set(null);
    this.databaseActionError.set(null);
    this.databaseActionMessage.set(null);
  }

  hasValidationErrors(): boolean {
    return Object.keys(this.validationErrors()).length > 0;
  }

  selectTab(tab: SettingsTab): void {
    void this.flushSettingsPersistence();
    this.activeTab.set(tab);
  }

  onTabKeydown(event: KeyboardEvent, tab: SettingsTab): void {
    const currentIndex = this.tabs.findIndex((candidate) => candidate.id === tab);
    if (currentIndex < 0) return;

    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (currentIndex + 1) % this.tabs.length;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (currentIndex - 1 + this.tabs.length) % this.tabs.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = this.tabs.length - 1;
    }

    if (nextIndex === null) return;
    event.preventDefault();
    const nextTab = this.tabs[nextIndex];
    document.getElementById(`settings-tab-${nextTab.id}`)?.focus();
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

  updatePollingInterval(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, jobs: { polling_interval: value ?? Number.NaN } }));
  }

  updateModelTimeout(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, inference: { ...draft.inference, model_timeout: value ?? Number.NaN } }));
  }

  updateModelIdleTimeout(value: number | null): void {
    this.updateDraft((draft) => ({ ...draft, inference: { ...draft.inference, model_idle_timeout: value ?? Number.NaN } }));
  }

  async reset(): Promise<void> {
    await this.flushSettingsPersistence();
    this.resetting.set(true);
    this.resetError.set(null);
    this.saveError.set(null);
    this.statusMessage.set(null);
    const response = await this.api.resetSettings();
    if (response.result) {
      this.theme.applyDatabasePreference(response.result.values.theme_preference);
      this.applyResponse(response.result);
      this.statusMessage.set('Settings reset to defaults.');
    } else {
      this.resetError.set(response.error ?? 'Unable to reset application settings.');
    }
    this.resetting.set(false);
  }

  async flushSettingsPersistence(): Promise<void> {
    this.clearAutosaveTimer();
    const run = this.persistenceQueue.then(() => this.persistLatestSettings());
    this.persistenceQueue = run.catch(() => undefined);
    await run;
  }

  private updateDraft(update: (draft: EditableSettings) => EditableSettings, delayMs = this.autosaveDelayMs): void {
    const current = this.draft();
    if (current) this.draft.set(update(current));
    this.settingsRevision += 1;
    this.statusMessage.set(null);
    this.saveError.set(null);
    this.resetError.set(null);
    this.scheduleAutosave(delayMs);
  }

  private scheduleAutosave(delayMs: number): void {
    this.clearAutosaveTimer();
    if (delayMs <= 0) {
      void this.flushSettingsPersistence();
      return;
    }
    this.autosaveTimer = setTimeout(() => {
      this.autosaveTimer = null;
      void this.flushSettingsPersistence();
    }, delayMs);
  }

  private clearAutosaveTimer(): void {
    if (this.autosaveTimer === null) return;
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = null;
  }

  private async persistLatestSettings(): Promise<void> {
    const current = this.draft();
    const original = this.baseline();
    if (!current || !original) return;
    const patch = this.buildPatch(current, original);
    if (!this.hasPatch(patch)) return;

    const revision = this.settingsRevision;
    this.saving.set(true);
    this.saveError.set(null);
    this.resetError.set(null);
    this.statusMessage.set(null);
    try {
      const response = await this.api.updateSettings(patch);
      if (response.result) {
        const persisted = this.toEditable(response.result.values);
        this.baseline.set(persisted);
        if (revision === this.settingsRevision) {
          this.draft.set(this.mergePersistedPatch(current, persisted, patch));
        }
        this.statusMessage.set(this.dirty() && this.hasValidationErrors()
          ? 'Valid changes saved. Correct the remaining value to save it.'
          : 'Settings saved.');
      } else {
        if (revision === this.settingsRevision) {
          this.draft.set(this.mergePersistedPatch(current, original, patch));
        }
        this.saveError.set(`${response.error ?? 'Unable to save application settings.'} ${revision === this.settingsRevision ? 'The failed changes were reverted.' : 'Newer edits remain visible and will be saved when possible.'}`);
      }
    } catch (error) {
      if (revision === this.settingsRevision) {
        this.draft.set(this.mergePersistedPatch(current, original, patch));
      }
      this.saveError.set(`${error instanceof Error ? error.message : 'Unable to save application settings.'} ${revision === this.settingsRevision ? 'The failed changes were reverted.' : 'Newer edits remain visible and will be saved when possible.'}`);
    } finally {
      this.saving.set(false);
    }
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
      jobs: { polling_interval: values.jobs.polling_interval },
      inference: { model_timeout: values.inference.model_timeout, model_idle_timeout: values.inference.model_idle_timeout },
    };
  }

  private buildPatch(current: EditableSettings, original: EditableSettings): ApplicationSettingsPatch {
    const patch: ApplicationSettingsPatch = {};
    if (current.global.seed !== original.global.seed && Number.isInteger(current.global.seed)
      && current.global.seed >= 0 && current.global.seed <= 4_294_967_295) {
      patch.global = { seed: current.global.seed };
    }
    if (current.jobs.polling_interval !== original.jobs.polling_interval && Number.isFinite(current.jobs.polling_interval)
      && current.jobs.polling_interval >= 0.25 && current.jobs.polling_interval <= 60) {
      patch.jobs = { polling_interval: current.jobs.polling_interval };
    }
    const inference: NonNullable<ApplicationSettingsPatch['inference']> = {};
    if (current.inference.model_timeout !== original.inference.model_timeout && Number.isInteger(current.inference.model_timeout)
      && current.inference.model_timeout >= 1) {
      inference.model_timeout = current.inference.model_timeout;
    }
    if (current.inference.model_idle_timeout !== original.inference.model_idle_timeout && Number.isInteger(current.inference.model_idle_timeout)
      && current.inference.model_idle_timeout >= 0) {
      inference.model_idle_timeout = current.inference.model_idle_timeout;
    }
    if (Object.keys(inference).length) patch.inference = inference;
    return patch;
  }

  private hasPatch(patch: ApplicationSettingsPatch): boolean {
    return patch.global !== undefined || patch.jobs !== undefined || patch.inference !== undefined;
  }

  private mergePersistedPatch(
    current: EditableSettings,
    persisted: EditableSettings,
    patch: ApplicationSettingsPatch,
  ): EditableSettings {
    return {
      global: patch.global ? { seed: persisted.global.seed } : current.global,
      jobs: patch.jobs ? { polling_interval: persisted.jobs.polling_interval } : current.jobs,
      inference: {
        model_timeout: patch.inference?.model_timeout !== undefined ? persisted.inference.model_timeout : current.inference.model_timeout,
        model_idle_timeout: patch.inference?.model_idle_timeout !== undefined ? persisted.inference.model_idle_timeout : current.inference.model_idle_timeout,
      },
    };
  }
}
