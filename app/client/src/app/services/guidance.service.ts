import { Injectable, inject } from '@angular/core';
import { Subject } from 'rxjs';
import { StorageService } from './storage.service';
import type { GuidanceDefinition, GuidanceEntry, GuidanceStatus } from '../types/guidance';

export const GUIDANCE_STORAGE_KEY = 'xreport.guidance.v1';
export const INFERENCE_TOUR_ID = 'inference-tour';

export const INFERENCE_TOUR: GuidanceDefinition = {
  id: INFERENCE_TOUR_ID,
  version: 1,
  route: '/inference',
  steps: [
    {
      id: 'model',
      target: '[data-guidance-target="inference-model-catalog"]',
      title: 'Choose a model',
      body: 'Select a model to see its input, output, readiness, and hardware contract before you begin.',
      placement: 'right',
    },
    {
      id: 'study',
      target: '[data-guidance-target="inference-study-upload"]',
      title: 'Add the study images',
      body: 'Browse local image files or drop them here. The selected model determines how many current views you can add.',
      placement: 'right',
    },
    {
      id: 'generate',
      target: '[data-guidance-target="inference-generation-settings"]',
      title: 'Configure and generate',
      body: 'Add optional context when the model supports it, choose a profile, and start a draft generation job.',
      placement: 'left',
    },
    {
      id: 'review',
      target: '[data-guidance-target="inference-draft-review"]',
      title: 'Review the draft',
      body: 'Edit the declared sections, inspect the provenance, then copy or export the research-use draft for qualified review.',
      placement: 'top',
    },
  ],
};

@Injectable({ providedIn: 'root' })
export class GuidanceService {
  private readonly storage = inject(StorageService);
  private readonly entries = this.storage.readRecord<GuidanceEntry>(GUIDANCE_STORAGE_KEY);
  private readonly tourRequestSubject = new Subject<string>();

  readonly tourRequests$ = this.tourRequestSubject.asObservable();

  getEntry(id: string): GuidanceEntry | null {
    return this.entries[id] ?? null;
  }

  shouldShow(id: string, version: number): boolean {
    const entry = this.entries[id];
    return !entry || entry.version !== version;
  }

  markSeen(id: string, version: number): void {
    this.setStatus(id, version, 'seen');
  }

  dismiss(id: string, version: number): void {
    this.setStatus(id, version, 'dismissed');
  }

  skip(id: string, version: number): void {
    this.setStatus(id, version, 'skipped');
  }

  complete(id: string, version: number): void {
    this.setStatus(id, version, 'completed');
  }

  requestTour(id: string): void {
    this.tourRequestSubject.next(id);
  }

  private setStatus(id: string, version: number, status: GuidanceStatus): void {
    const current = this.entries[id];
    if (status === 'seen' && current?.version === version && ['dismissed', 'skipped', 'completed'].includes(current.status)) return;
    this.entries[id] = { version, status };
    this.storage.writeRecord(GUIDANCE_STORAGE_KEY, this.entries);
  }
}
