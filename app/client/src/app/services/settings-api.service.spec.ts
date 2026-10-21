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
    expect(request.request).toHaveBeenNthCalledWith(2, 'PATCH', '/api/settings', { global: { seed: 123 } }, { keepalive: true });
    expect(request.request).toHaveBeenNthCalledWith(3, 'POST', '/api/settings/reset', undefined, { keepalive: true });
  });

  it('does not replace transport errors', async () => {
    const response = { result: null, error: '503 Service Unavailable: backend offline' };
    request.request.mockResolvedValue(response);

    await expect(service.getSettings()).resolves.toBe(response);
  });

  it('serializes settings writes so rapid changes reach the database in order', async () => {
    let completeFirst!: (value: { result: { values: Record<string, unknown>; defaults: Record<string, unknown> }; error: null }) => void;
    request.request
      .mockImplementationOnce(() => new Promise((resolve) => { completeFirst = resolve; }))
      .mockResolvedValueOnce({ result: { values: {}, defaults: {} }, error: null });

    const first = service.updateSettings({ global: { seed: 100 } });
    const second = service.updateSettings({ global: { seed: 101 } });
    await Promise.resolve();

    expect(request.request).toHaveBeenCalledOnce();
    completeFirst({ result: { values: {}, defaults: {} }, error: null });
    await Promise.all([first, second]);

    expect(request.request).toHaveBeenNthCalledWith(1, 'PATCH', '/api/settings', { global: { seed: 100 } }, { keepalive: true });
    expect(request.request).toHaveBeenNthCalledWith(2, 'PATCH', '/api/settings', { global: { seed: 101 } }, { keepalive: true });
  });

  it('uses the database status, connection-test, and switch endpoints', async () => {
    const response = { result: { state: 'ready' }, error: null };
    request.request.mockResolvedValue(response);
    const candidate = {
      backend: 'sqlite' as const,
      sqlite_path: 'C:\\data\\database.db',
      engine: null,
      host: null,
      port: 5432,
      database_name: null,
      username: null,
      password: null,
      ssl: false,
      ssl_ca: null,
      connect_timeout: 30,
      insert_batch_size: 1000,
    };

    await expect(service.getDatabaseSettings()).resolves.toBe(response);
    await expect(service.testDatabaseConnection(candidate)).resolves.toBe(response);
    await expect(service.switchDatabase(candidate)).resolves.toBe(response);

    expect(request.request).toHaveBeenNthCalledWith(1, 'GET', '/api/settings/database');
    expect(request.request).toHaveBeenNthCalledWith(2, 'POST', '/api/settings/database/test', candidate);
    expect(request.request).toHaveBeenNthCalledWith(3, 'POST', '/api/settings/database/switch', candidate);
  });
});
