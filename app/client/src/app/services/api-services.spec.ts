import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { ApiRequestService } from './api-request.service';
import { InferenceApiService } from './inference-api.service';
import { SettingsApiService } from './settings-api.service';

describe('InferenceApiService history methods', () => {
  const inferenceRequest = { request: vi.fn() };
  let service: InferenceApiService;

  beforeEach(() => {
    inferenceRequest.request.mockReset().mockResolvedValue({ result: {}, error: null });
    TestBed.configureTestingModule({
      providers: [InferenceApiService, { provide: ApiRequestService, useValue: inferenceRequest }],
    });
    service = TestBed.inject(InferenceApiService);
  });

  it('serializes list filters and keeps request identifiers encoded', async () => {
    await service.listHistory({
      modelRef: 'huggingface:e2e/reports',
      status: 'succeeded',
      sort: 'oldest',
      limit: 10,
      offset: 20,
    });
    await service.getHistory('request/with spaces');

    expect(inferenceRequest.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/inference/history?model_ref=huggingface%3Ae2e%2Freports&status=succeeded&sort=oldest&limit=10&offset=20',
    );
    expect(inferenceRequest.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/inference/history/request%2Fwith%20spaces',
    );
  });

  it('delegates atomic update and deletion without replacing transport results', async () => {
    const update = { reports: [{ image_index: 0, edited_report: 'Edited draft.' }] };
    const response = { result: { request_id: 'request-1' }, error: null };
    inferenceRequest.request.mockResolvedValue(response);

    await expect(service.updateHistory('request-1', update)).resolves.toBe(response);
    await expect(service.deleteHistory('request-1')).resolves.toBe(response);

    expect(inferenceRequest.request).toHaveBeenNthCalledWith(1, 'PATCH', '/api/inference/history/request-1', update);
    expect(inferenceRequest.request).toHaveBeenNthCalledWith(2, 'DELETE', '/api/inference/history/request-1');
  });
});

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
