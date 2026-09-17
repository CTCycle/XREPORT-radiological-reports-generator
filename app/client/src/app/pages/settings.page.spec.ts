import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { SettingsApiService } from '../services/settings-api.service';
import { SettingsPage } from './settings.page';

const settingsResponse = () => ({
  values: {
    global: { seed: 42 },
    features: { allow_local_filesystem_access: true },
    jobs: { polling_interval: 1 },
    inference: { model_timeout: 600 },
  },
  defaults: {
    global: { seed: 42 },
    features: { allow_local_filesystem_access: true },
    jobs: { polling_interval: 1 },
    inference: { model_timeout: 600 },
  },
});

describe('SettingsPage', () => {
  const api = {
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
    resetSettings: vi.fn(),
  };

  beforeEach(() => {
    api.getSettings.mockReset().mockResolvedValue({ result: settingsResponse(), error: null });
    api.updateSettings.mockReset().mockResolvedValue({ result: settingsResponse(), error: null });
    api.resetSettings.mockReset().mockResolvedValue({ result: settingsResponse(), error: null });
    TestBed.configureTestingModule({ imports: [SettingsPage], providers: [{ provide: SettingsApiService, useValue: api }] });
  });

  it('loads the server baseline and sends only changed fields', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;

    expect(page.draft()?.global.seed).toBe(42);
    expect(page.dirty()).toBe(false);
    page.updateSeed(123);
    page.updatePollingInterval(2.5);
    expect(page.dirty()).toBe(true);

    await page.save();

    expect(api.updateSettings).toHaveBeenCalledWith({ global: { seed: 123 }, jobs: { polling_interval: 2.5 } });
    expect(page.dirty()).toBe(false);
    expect(page.statusMessage()).toBe('Settings saved.');
  });

  it('blocks invalid values before making a save request', async () => {
    const fixture = TestBed.createComponent(SettingsPage);
    fixture.detectChanges();
    await fixture.whenStable();
    const page = fixture.componentInstance;
    page.updatePollingInterval(0);

    await page.save();

    expect(page.validationErrors().polling_interval).toBeTruthy();
    expect(api.updateSettings).not.toHaveBeenCalled();
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
