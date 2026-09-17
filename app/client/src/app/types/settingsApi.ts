import type { components } from './api.generated';

type Schemas = components['schemas'];

export type RuntimeApplicationSettings = Schemas['RuntimeApplicationSettings'];
export type ApplicationSettingsResponse = Schemas['ApplicationSettingsResponse'];
export type ApplicationSettingsPatch = Schemas['ApplicationSettingsPatch'];

export interface EditableSettings {
  global: { seed: number };
  features: { allow_local_filesystem_access: boolean };
  jobs: { polling_interval: number };
  inference: { model_timeout: number };
}
