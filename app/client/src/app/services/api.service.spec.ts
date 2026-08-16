import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { ApiService } from './api.service';

describe('ApiService', () => {
  let service: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [ApiService, provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('preserves the inference models HTTP contract', async () => {
    const response = service.getInferenceModels();
    const request = http.expectOne('/api/inference/models');
    expect(request.request.method).toBe('GET');
    request.flush({ models: [], providers: {} });
    await expect(response).resolves.toEqual({ result: { models: [], providers: {} }, error: null });
  });

  it('returns a useful error envelope for failed requests', async () => {
    const response = service.getDatasetNames();
    http.expectOne('/api/preparation/dataset/names').flush({ detail: 'database unavailable' }, { status: 503, statusText: 'Service Unavailable' });
    await expect(response).resolves.toEqual({ result: null, error: '503 Service Unavailable: database unavailable' });
  });
});
