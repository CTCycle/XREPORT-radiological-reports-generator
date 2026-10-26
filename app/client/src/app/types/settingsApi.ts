import type {
  ApplicationSettingsPatch as ApiApplicationSettingsPatch,
  ApplicationSettingsResponse as ApiApplicationSettingsResponse,
  RuntimeApplicationSettings as ApiRuntimeApplicationSettings,
} from './generated/xreport/types.gen';

export type RuntimeApplicationSettings = ApiRuntimeApplicationSettings;
export type ApplicationSettingsResponse = ApiApplicationSettingsResponse;
export type ApplicationSettingsPatch = ApiApplicationSettingsPatch;

export type DatabaseBackend = 'sqlite' | 'postgresql';

export interface DatabaseConfigurationDraft {
  backend: DatabaseBackend;
  sqlite_path: string | null;
  engine: string | null;
  host: string | null;
  port: number;
  database_name: string | null;
  username: string | null;
  password: string | null;
  ssl: boolean;
  ssl_ca: string | null;
  connect_timeout: number;
  insert_batch_size: number;
}

export type DatabaseConfigurationSummary = Omit<DatabaseConfigurationDraft, 'password'> & {
  password_configured: boolean;
};

export interface DatabaseConfigurationStatus {
  state: 'ready' | 'recovery' | 'unavailable';
  active: DatabaseConfigurationSummary | null;
  suggested: DatabaseConfigurationSummary;
  connected: boolean;
  schema_ready: boolean;
  jobs_running: boolean;
  error: string | null;
}

export interface DatabaseConnectionTest {
  success: boolean;
  backend: DatabaseBackend;
  connected: boolean;
  schema_ready: boolean;
  detail: string;
}

export interface DatabaseSwitchResult {
  active: DatabaseConfigurationSummary;
  settings: ApplicationSettingsResponse;
}

export interface EditableSettings {
  global: { seed: number };
  features: { allow_local_filesystem_access: boolean };
  jobs: { polling_interval: number };
  inference: { model_timeout: number };
}

export interface HuggingFaceAccessStatus {
  configured: boolean;
  source: 'environment' | 'stored' | null;
}
