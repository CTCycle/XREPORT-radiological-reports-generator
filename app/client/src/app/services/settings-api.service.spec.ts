import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { ApiRequestService } from './api-request.service';
import { SettingsApiService } from './settings-api.service';

describe('SettingsApiService', () => {
  const request = { request: vi.fn() };
  let service: SettingsApiService;

  beforeEach(() => {
    request.request.mockReset();
    TestBed.configureTestingModule({ providers: [SettingsApiService, { provide: ApiRequestService, useValue: request }] });
    service = TestBed.inject(SettingsApiService);
  });

  it('uses the settings endpoints and preserves ApiRequestService results', async () => {
    const response = { result: { values: {}, defaults: {} }, error: null };
    request.request.mockResolvedValue(response);

    await expect(service.getSettings()).resolves.toBe(response);
    await expect(service.updateSettings({ global: { seed: 123 } })).resolves.toBe(response);
    await expect(service.resetSettings()).resolves.toBe(response);

    expect(request.request).toHaveBeenNthCalledWith(1, 'GET', '/api/settings');
    expect(request.request).toHaveBeenNthCalledWith(2, 'PATCH', '/api/settings', { global: { seed: 123 } });
    expect(request.request).toHaveBeenNthCalledWith(3, 'POST', '/api/settings/reset');
  });

  it('does not replace transport errors', async () => {
    const response = { result: null, error: '503 Service Unavailable: backend offline' };
    request.request.mockResolvedValue(response);

    await expect(service.getSettings()).resolves.toBe(response);
  });
});
