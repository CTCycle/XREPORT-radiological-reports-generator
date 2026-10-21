import { PLATFORM_ID } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import type { ApiResult } from './api-request.service';
import { SettingsApiService } from './settings-api.service';
import { ThemeService } from './theme.service';
import type { ThemePreference } from '../types/theme';
import type { ApplicationSettingsResponse } from '../types/settingsApi';
import { vi } from 'vitest';

describe('ThemeService', () => {
  const settingsApi = {
    getSettings: vi.fn(),
    updateSettings: vi.fn(),
  };
  let service: ThemeService;

  const response = (themePreference: ThemePreference | null) => ({
    result: {
      values: { theme_preference: themePreference },
      defaults: { theme_preference: null },
    },
    error: null,
  }) as unknown as ApiResult<ApplicationSettingsResponse>;

  beforeEach(() => {
    localStorage.clear();
    settingsApi.getSettings.mockReset();
    settingsApi.updateSettings.mockReset();
    TestBed.configureTestingModule({
      providers: [
        ThemeService,
        { provide: SettingsApiService, useValue: settingsApi },
        { provide: PLATFORM_ID, useValue: 'browser' },
      ],
    });
    service = TestBed.inject(ThemeService);
  });

  it('uses the selected database theme and ignores a stale browser copy', async () => {
    localStorage.setItem('theme-preference', 'light');
    settingsApi.getSettings.mockResolvedValue(response('dark'));

    await service.loadFromDatabase();

    expect(settingsApi.updateSettings).not.toHaveBeenCalled();
    expect(service.preference()).toBe('dark');
  });

  it('persists only the latest rapid theme selection', async () => {
    settingsApi.updateSettings.mockResolvedValue(response('light'));

    const first = service.setPreference('dark');
    const latest = service.setPreference('light');
    await Promise.all([first, latest]);

    expect(settingsApi.updateSettings).toHaveBeenCalledOnce();
    expect(settingsApi.updateSettings).toHaveBeenCalledWith({ theme_preference: 'light' });
    expect(service.preference()).toBe('light');
    expect(service.persistenceSaved()).toBe(true);
  });

  it('reverts to the last persisted theme when saving a new preference fails', async () => {
    settingsApi.getSettings.mockResolvedValue(response('dark'));
    await service.loadFromDatabase();
    settingsApi.updateSettings.mockResolvedValue({ result: null, error: '503 Service Unavailable: backend offline' });

    await service.setPreference('light');

    expect(service.preference()).toBe('dark');
    expect(service.persistenceError()).toContain('backend offline');
    expect(service.persistenceSaved()).toBe(false);
  });

  it('does not accept theme changes while a database switch is in progress', async () => {
    service.setDatabaseSwitching(true);

    await service.setPreference('dark');

    expect(service.preference()).toBe('system');
    expect(settingsApi.updateSettings).not.toHaveBeenCalled();
  });
});
