import { Injectable, inject } from '@angular/core';
import type {
  ApplicationSettingsPatch,
  ApplicationSettingsResponse,
  DatabaseConfigurationDraft,
  DatabaseConfigurationStatus,
  DatabaseConnectionTest,
  DatabaseSwitchResult,
  HuggingFaceAccessStatus,
} from '../types/settingsApi';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  private readonly request = inject(ApiRequestService);
  private settingsQueue: Promise<void> = Promise.resolve();

  getSettings() {
    return this.enqueueSettingsRequest(() => this.request.request<ApplicationSettingsResponse>('GET', '/api/settings'));
  }

  updateSettings(patch: ApplicationSettingsPatch) {
    return this.enqueueSettingsRequest(() => this.request.request<ApplicationSettingsResponse>(
      'PATCH', '/api/settings', patch, { keepalive: true },
    ));
  }

  resetSettings() {
    return this.enqueueSettingsRequest(() => this.request.request<ApplicationSettingsResponse>(
      'POST', '/api/settings/reset', undefined, { keepalive: true },
    ));
  }

  getDatabaseSettings() {
    return this.request.request<DatabaseConfigurationStatus>('GET', '/api/settings/database');
  }

  testDatabaseConnection(candidate: DatabaseConfigurationDraft) {
    return this.request.request<DatabaseConnectionTest>('POST', '/api/settings/database/test', candidate);
  }

  switchDatabase(candidate: DatabaseConfigurationDraft) {
    return this.request.request<DatabaseSwitchResult>('POST', '/api/settings/database/switch', candidate);
  }

  getHuggingFaceAccess() {
    return this.request.request<HuggingFaceAccessStatus>('GET', '/api/settings/huggingface');
  }

  saveHuggingFaceToken(token: string) {
    return this.request.request<HuggingFaceAccessStatus>('PUT', '/api/settings/huggingface', { token });
  }

  clearHuggingFaceToken() {
    return this.request.request<HuggingFaceAccessStatus>('DELETE', '/api/settings/huggingface');
  }

  private enqueueSettingsRequest<T>(operation: () => Promise<T>): Promise<T> {
    const pending = this.settingsQueue.then(operation, operation);
    this.settingsQueue = pending.then(() => undefined, () => undefined);
    return pending;
  }
}
