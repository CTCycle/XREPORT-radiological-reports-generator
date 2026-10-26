import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { SettingsApiService } from '../services/settings-api.service';
import type { ApplicationSettingsPatch } from '../types/settingsApi';
import { SettingsPage } from './settings.page';

const settingsResponse = () => ({
  values: {
    global: { seed: 42 },
    jobs: { polling_interval: 1 },
    inference: { model_timeout: 600 },
  },
  defaults: {
    global: { seed: 42 },
    jobs: { polling_interval: 1 },
    inference: { model_timeout: 600 },
  },
});

const databaseSummary = (sqlitePath = 'C:\\data\\database.db') => ({
  backend: 'sqlite' as const,
  sqlite_path: sqlitePath,
  engine: null,
  host: null,
  port: 5432,
  database_name: null,
  username: null,
  password_configured: false,
  ssl: false,
  ssl_ca: null,
  connect_timeout: 30,
  insert_batch_size: 1000,
});

const databaseStatus = () => ({
  state: 'ready' as const,
  active: databaseSummary(),
  suggested: databaseSummary(),
  connected: true,
  schema_ready: true,
  jobs_running: false,
  error: null,
});

describe('SettingsPage', () => {
  const api = {
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
    resetSettings: vi.fn(),
    getDatabaseSettings: vi.fn(),
    testDatabaseConnection: vi.fn(),
    switchDatabase: vi.fn(),
    getHuggingFaceAccess: vi.fn(),
    saveHuggingFaceToken: vi.fn(),
    clearHuggingFaceToken: vi.fn(),
    getStorage: vi.fn(),
    validateStorage: vi.fn(),
    applyStorage: vi.fn(),
  };

  beforeEach(() => {
    api.getSettings.mockReset().mockResolvedValue({ result: settingsResponse(), error: null });
    api.updateSettings.mockReset().mockImplementation(async (patch: ApplicationSettingsPatch) => {
      const result = settingsResponse();
      if (patch.global?.seed != null) result.values.global.seed = patch.global.seed;
      if (patch.jobs?.polling_interval != null) result.values.jobs.polling_interval = patch.jobs.polling_interval;
      if (patch.inference?.model_timeout != null) result.values.inference.model_timeout = patch.inference.model_timeout;
      return { result, error: null };
    });
    api.resetSettings.mockReset().mockResolvedValue({ result: settingsResponse(), error: null });
    api.getDatabaseSettings.mockReset().mockResolvedValue({ result: databaseStatus(), error: null });
    api.testDatabaseConnection.mockReset().mockResolvedValue({
      result: { success: true, backend: 'sqlite', connected: true, schema_ready: true, detail: 'Database is ready.' },
      error: null,
    });
    api.switchDatabase.mockReset().mockResolvedValue({
      result: { active: databaseSummary('C:\\data\\target.db'), settings: settingsResponse() },
      error: null,
    });
    api.getHuggingFaceAccess.mockReset().mockResolvedValue({ result: { configured: false, source: null }, error: null });
    api.saveHuggingFaceToken.mockReset().mockResolvedValue({ result: { configured: true, source: 'stored' }, error: null });
    api.clearHuggingFaceToken.mockReset().mockResolvedValue({ result: { configured: false, source: null }, error: null });
    api.getStorage.mockReset().mockResolvedValue({ result: { active_root: 'C:/data', default_root: 'C:/data', pending_root: null, source: 'default', locked: false, restart_required: false, folders: [{ name: 'models', path: 'C:/data/models', size_bytes: 2048 }] }, error: null });
    api.validateStorage.mockReset();
    api.applyStorage.mockReset();
    TestBed.configureTestingModule({ imports: [SettingsPage], providers: [{ provide: SettingsApiService, useValue: api }] });
  });

  it('loads the server baseline and autosaves only changed fields', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const page = fixture.componentInstance;

    expect(page.draft()?.global.seed).toBe(42);
    expect(page.dirty()).toBe(false);
    page.updateSeed(123);
    page.updatePollingInterval(2.5);
    expect(page.dirty()).toBe(true);

    await page.flushSettingsPersistence();

    expect(api.updateSettings).toHaveBeenCalledWith({ global: { seed: 123 }, jobs: { polling_interval: 2.5 } });
    expect(page.dirty()).toBe(false);
    expect(page.statusMessage()).toBe('Settings saved.');
  });

  it('blocks invalid values before making a persistence request', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;
    page.updatePollingInterval(0);

    await page.flushSettingsPersistence();

    expect(page.validationErrors().polling_interval).toBeTruthy();
    expect(api.updateSettings).not.toHaveBeenCalled();
  });

  it('debounces continuous edits and persists every numeric setting', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;

    page.updateSeed(123);
    page.updateSeed(124);
    page.updatePollingInterval(2.5);
    page.updateModelTimeout(900);
    expect(api.updateSettings).not.toHaveBeenCalled();
    await new Promise((resolve) => setTimeout(resolve, 400));

    expect(api.updateSettings).toHaveBeenCalledOnce();
    expect(api.updateSettings).toHaveBeenCalledWith({
      global: { seed: 124 },
      jobs: { polling_interval: 2.5 },
      inference: { model_timeout: 900 },
    });
    expect(page.draft()?.global.seed).toBe(124);
    expect(page.draft()?.jobs.polling_interval).toBe(2.5);
    expect(page.draft()?.inference.model_timeout).toBe(900);
    expect(page.dirty()).toBe(false);
  });

  it('serializes rapid updates so an older response cannot overwrite the latest value', async () => {
    let completeFirst!: (value: { result: ReturnType<typeof settingsResponse> | null; error: string | null }) => void;
    api.updateSettings
      .mockImplementationOnce(() => new Promise((resolve) => { completeFirst = resolve; }))
      .mockImplementationOnce(async () => {
        const result = settingsResponse();
        result.values.global.seed = 200;
        return { result, error: null };
      });
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;

    page.updateSeed(100);
    const first = page.flushSettingsPersistence();
    await Promise.resolve();
    await Promise.resolve();
    page.updateSeed(200);
    const second = page.flushSettingsPersistence();
    const firstResult = settingsResponse();
    firstResult.values.global.seed = 100;
    completeFirst({ result: firstResult, error: null });
    await Promise.all([first, second]);

    expect(api.updateSettings).toHaveBeenNthCalledWith(1, { global: { seed: 100 } });
    expect(api.updateSettings).toHaveBeenNthCalledWith(2, { global: { seed: 200 } });
    expect(page.draft()?.global.seed).toBe(200);
    expect(page.dirty()).toBe(false);
  });

  it('reverts a change and reports an error when automatic persistence fails', async () => {
    api.updateSettings.mockResolvedValue({ result: null, error: '503 Service Unavailable: backend offline' });
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;

    page.updateSeed(7);
    await page.flushSettingsPersistence();

    expect(page.draft()?.global.seed).toBe(42);
    expect(page.saveError()).toContain('backend offline');
    expect(page.saveError()).toContain('reverted');
  });

  it('switches logical categories while keeping the shared settings form', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const page = fixture.componentInstance;
    const element = fixture.nativeElement as HTMLElement;

    expect(page.activeTab()).toBe('general');
    expect(element.querySelector('#default-seed')).not.toBeNull();

    (element.querySelector('#settings-tab-advanced') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(page.activeTab()).toBe('advanced');
    expect(element.querySelector('#polling-interval')).not.toBeNull();
    expect(element.querySelector('#inference-timeout')).not.toBeNull();

    (element.querySelector('#settings-tab-database') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(page.activeTab()).toBe('database');
    expect(element.querySelector('#database-backend')).not.toBeNull();
    expect(element.querySelector('#database-sqlite-path')).not.toBeNull();
  });

  it('saves the Hugging Face token once, clears the field, and offers removal', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const page = fixture.componentInstance;
    const element = fixture.nativeElement as HTMLElement;

    (element.querySelector('#settings-tab-models') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(element.textContent).toContain('No token is configured.');

    page.huggingFaceTokenInput = ' hf_example ';
    await page.saveHuggingFaceToken();
    fixture.detectChanges();

    expect(api.saveHuggingFaceToken).toHaveBeenCalledWith('hf_example');
    expect(page.huggingFaceTokenInput).toBe('');
    expect(element.textContent).toContain('A token is saved on this computer.');
    expect(element.textContent).toContain('Remove token');
  });

  it('validates a storage folder and starts a move only for a valid check', async () => {
    api.validateStorage.mockResolvedValue({ result: { valid: true, resolved_path: 'D:/xreport', target_state: 'empty', required_bytes: 2048, free_bytes: 4096, errors: [], warnings: [] }, error: null });
    api.applyStorage.mockResolvedValue({ result: { job_id: 'move-1', job_type: 'storage_migration', status: 'running', message: 'started', poll_interval: 1 }, error: null });
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const page = fixture.componentInstance;
    const element = fixture.nativeElement as HTMLElement;

    page.selectTab('storage');
    fixture.detectChanges();
    expect(element.textContent).toContain('C:/data');
    expect(element.textContent).toContain('models · 2.0 KB');

    await page.applyStorage();
    expect(api.applyStorage).not.toHaveBeenCalled();

    page.onStoragePathChange('D:/xreport');
    await page.validateStorage();
    await page.applyStorage();

    expect(api.validateStorage).toHaveBeenCalledWith('D:/xreport');
    expect(api.applyStorage).toHaveBeenCalledWith('D:/xreport', 'copy');
    page.ngOnDestroy();
  });

  it('tests and switches the selected database, then applies its settings immediately', async () => {
    const switchedSettings = settingsResponse();
    switchedSettings.values.global.seed = 77;
    api.switchDatabase.mockResolvedValue({
      result: { active: databaseSummary('C:\\data\\target.db'), settings: switchedSettings },
      error: null,
    });
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;
    page.selectTab('database');
    fixture.detectChanges();
    const applyButton = (fixture.nativeElement as HTMLElement).querySelector('.database-actions .primary-button') as HTMLButtonElement;

    await page.testDatabaseConnection();
    expect(page.databaseTestResult()?.success).toBe(true);

    page.updateSeed(98);
    page.updateDatabaseDraft({ sqlite_path: 'C:\\data\\target.db' });
    await page.switchDatabase();
    expect(api.switchDatabase).not.toHaveBeenCalled();
    await page.flushSettingsPersistence();
    await page.switchDatabase();
    expect(api.switchDatabase).not.toHaveBeenCalled();
    expect(page.databaseTestResult()).toBeNull();
    expect(applyButton.disabled).toBe(true);
    await page.testDatabaseConnection();
    fixture.detectChanges();
    expect(applyButton.disabled).toBe(false);
    await page.switchDatabase();

    expect(api.switchDatabase).toHaveBeenCalledOnce();
    expect(page.databaseStatus()?.active?.sqlite_path).toBe('C:\\data\\target.db');
    expect(page.draft()?.global.seed).toBe(77);
    expect(page.dirty()).toBe(false);
  });

  it('has no ordinary Save or Apply control', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('button[type="submit"]')).toBeNull();
    expect(element.textContent).not.toContain('Save changes');
  });

  it('moves focus across categories with the tablist arrow keys', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const page = fixture.componentInstance;
    const element = fixture.nativeElement as HTMLElement;
    const generalTab = element.querySelector('#settings-tab-general') as HTMLButtonElement;
    const advancedTab = element.querySelector('#settings-tab-advanced') as HTMLButtonElement;

    generalTab.focus();
    page.onTabKeydown(new KeyboardEvent('keydown', { key: 'ArrowRight' }), 'general');

    expect(document.activeElement).toBe(advancedTab);
    expect(page.activeTab()).toBe('general');
  });

  it('delegates reset and replaces the local baseline with the server response', async () => {
    const resetResponse = settingsResponse();
    resetResponse.values.global.seed = 99;
    api.resetSettings.mockResolvedValue({ result: resetResponse, error: null });
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();

    await fixture.componentInstance.reset();

    expect(api.resetSettings).toHaveBeenCalledOnce();
    expect(fixture.componentInstance.draft()?.global.seed).toBe(99);
    expect(fixture.componentInstance.dirty()).toBe(false);
  });
});
