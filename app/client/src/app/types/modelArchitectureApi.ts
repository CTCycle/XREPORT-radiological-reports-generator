import type {
  ArchitectureCompatibility as ApiArchitectureCompatibility,
  ArchitectureGraph as ApiArchitectureGraph,
  ArchitectureIssue as ApiArchitectureIssue,
  ArchitectureNode as ApiArchitectureNode,
  ArchitecturePreset as ApiArchitecturePreset,
  ArchitecturePresetRequest as ApiArchitecturePresetRequest,
  ArchitecturePresetsResponse as ApiArchitecturePresetsResponse,
  ArchitecturePreviewRequest as ApiArchitecturePreviewRequest,
  ArchitecturePreviewResponse as ApiArchitecturePreviewResponse,
  ArchitectureStats as ApiArchitectureStats,
  CheckpointArchitectureResponse as ApiCheckpointArchitectureResponse,
  EncoderOption as ApiEncoderOption,
  EncodersResponse as ApiEncodersResponse,
  ModelArchitecture as ApiModelArchitecture,
} from './generated/studio/types.gen';

export type ModelArchitecture = Required<ApiModelArchitecture>;
export type ArchitectureCompatibility = ApiArchitectureCompatibility;
export type ArchitectureGraph = ApiArchitectureGraph;
export type ArchitectureIssue = ApiArchitectureIssue;
export type ArchitectureNode = ApiArchitectureNode;
export type ArchitecturePreset = ApiArchitecturePreset;
export type ArchitecturePresetRequest = ApiArchitecturePresetRequest;
export type ArchitecturePresetsResponse = ApiArchitecturePresetsResponse;
export type ArchitecturePreviewRequest = ApiArchitecturePreviewRequest;
export type ArchitecturePreviewResponse = ApiArchitecturePreviewResponse;
export type ArchitectureStats = ApiArchitectureStats;
export type CheckpointArchitectureResponse = ApiCheckpointArchitectureResponse;
export type EncoderOption = ApiEncoderOption;
export type EncodersResponse = ApiEncodersResponse;

/** Portable preset file written by export and accepted by import. */
export interface ArchitectureFile {
  format: 'xreport-model-architecture';
  version: 1;
  name: string;
  description: string;
  architecture: ModelArchitecture;
}
