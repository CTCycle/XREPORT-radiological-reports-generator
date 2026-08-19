export type GuidanceStatus = 'seen' | 'dismissed' | 'skipped' | 'completed';

export type TourPlacement = 'top' | 'right' | 'bottom' | 'left';

export interface GuidanceEntry {
  version: number;
  status: GuidanceStatus;
}

export interface TourStep {
  id: string;
  target: string;
  title: string;
  body: string;
  route?: string;
  placement?: TourPlacement;
}

export interface GuidanceDefinition {
  id: string;
  version: number;
  route: string;
  steps: readonly TourStep[];
}
