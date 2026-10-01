import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { ApiRequestService } from './api-request.service';
import { InferenceApiService } from './inference-api.service';

describe('InferenceApiService history methods', () => {
  const request = { request: vi.fn() };
  let service: InferenceApiService;

  beforeEach(() => {
    request.request.mockReset().mockResolvedValue({ result: {}, error: null });
    TestBed.configureTestingModule({
      providers: [InferenceApiService, { provide: ApiRequestService, useValue: request }],
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

    expect(request.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/inference/history?model_ref=huggingface%3Ae2e%2Freports&status=succeeded&sort=oldest&limit=10&offset=20',
    );
    expect(request.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/inference/history/request%2Fwith%20spaces',
    );
  });

  it('delegates atomic update and deletion without replacing transport results', async () => {
    const update = { reports: [{ image_index: 0, edited_report: 'Edited draft.' }] };
    const response = { result: { request_id: 'request-1' }, error: null };
    request.request.mockResolvedValue(response);

    await expect(service.updateHistory('request-1', update)).resolves.toBe(response);
    await expect(service.deleteHistory('request-1')).resolves.toBe(response);

    expect(request.request).toHaveBeenNthCalledWith(1, 'PATCH', '/api/inference/history/request-1', update);
    expect(request.request).toHaveBeenNthCalledWith(2, 'DELETE', '/api/inference/history/request-1');
  });
});
