import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { ApiRequestService } from './api-request.service';
import { DatasetApiService } from './dataset-api.service';
import { InferenceApiService } from './inference-api.service';

describe('feature API services', () => {
  let http: HttpTestingController;
  let inferenceApi: InferenceApiService;
  let datasetApi: DatasetApiService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ApiRequestService, InferenceApiService, DatasetApiService, provideHttpClient(), provideHttpClientTesting()],
    });
    http = TestBed.inject(HttpTestingController);
    inferenceApi = TestBed.inject(InferenceApiService);
    datasetApi = TestBed.inject(DatasetApiService);
  });

  afterEach(() => http.verify());

  it('keeps the inference models contract in the inference client', async () => {
    const response = inferenceApi.getModels();
    const request = http.expectOne('/api/inference/models');
    expect(request.request.method).toBe('GET');
    request.flush({ models: [], providers: {} });
    await expect(response).resolves.toEqual({ result: { models: [], providers: {} }, error: null });
  });

  it('keeps shared error formatting in the dataset client', async () => {
    const response = datasetApi.getNames();
    http.expectOne('/api/preparation/dataset/names').flush({ detail: 'database unavailable' }, { status: 503, statusText: 'Service Unavailable' });
    await expect(response).resolves.toEqual({ result: null, error: '503 Service Unavailable: database unavailable' });
  });
});
