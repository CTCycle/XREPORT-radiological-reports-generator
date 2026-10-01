import { Injectable, inject } from '@angular/core';
import type { ApplicationSettingsPatch, ApplicationSettingsResponse } from '../types/settingsApi';
import { ApiRequestService } from './api-request.service';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  private readonly request = inject(ApiRequestService);

  getSettings() {
    return this.request.request<ApplicationSettingsResponse>('GET', '/api/settings');
  }

  updateSettings(patch: ApplicationSettingsPatch) {
    return this.request.request<ApplicationSettingsResponse>('PATCH', '/api/settings', patch);
  }

  resetSettings() {
    return this.request.request<ApplicationSettingsResponse>('POST', '/api/settings/reset');
  }
}
