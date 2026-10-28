import { Injectable, inject } from '@angular/core';
import { Subject } from 'rxjs';
import { StorageService } from './storage.service';
import type { GuidanceDefinition, GuidanceEntry, GuidanceStatus } from '../types/guidance';

export const GUIDANCE_STORAGE_KEY = 'xreport.guidance.v1';
export const INFERENCE_TOUR_ID = 'inference-tour';
export const DATA_TRAINING_TOUR_ID = 'data-training-tour';

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

export const DATA_TRAINING_TOUR: GuidanceDefinition = {
  id: DATA_TRAINING_TOUR_ID,
  version: 1,
  route: '/dataset',
  steps: [
    {
      id: 'source',
      target: '[data-guidance-target="dataset-source"]',
      route: '/dataset',
      title: '1. Upload and load the source',
      body: 'Choose an image folder (the desktop uses the native system picker; browser mode uses the local server fallback) and upload the CSV/XLSX reports or metadata file. Load Dataset matches the two inputs and stores the source rows for processing.',
      placement: 'bottom',
    },
    {
      id: 'process',
      target: '[data-guidance-target="dataset-processing"]',
      route: '/dataset',
      title: '2. Process the uploaded source',
      body: 'Select exactly one source dataset, configure the sample, validation split, tokenizer, and report limit, then Build Dataset. The work runs in the background.',
      placement: 'top',
    },
    {
      id: 'training-dataset',
      target: '[data-guidance-target="training-dataset-list"]',
      route: '/training',
      title: '3. Select the training dataset',
      body: 'Processing creates a separate training-ready dataset. Select it in New Training Session; this is the prepared input used by the model, not the raw source rows.',
      placement: 'right',
    },
    {
      id: 'new-training',
      target: '[data-guidance-target="training-new-action"]',
      route: '/training',
      title: '4. Start a new training run',
      body: 'Configure Training opens the five-step wizard. Review the model, dataset, epochs, batch size, device, and checkpoint-saving choices before starting the background run.',
      placement: 'left',
    },
    {
      id: 'resume',
      target: '[data-guidance-target="training-resume-action"]',
      route: '/training',
      title: '5. Resume from a checkpoint',
      body: 'When a checkpoint is available, select it under Resume Training, enter additional epochs, and continue from the saved state instead of starting from zero.',
      placement: 'left',
    },
  ],
};

const GUIDANCE_TOURS: Record<string, GuidanceDefinition> = {
  [INFERENCE_TOUR_ID]: INFERENCE_TOUR,
  [DATA_TRAINING_TOUR_ID]: DATA_TRAINING_TOUR,
};

@Injectable({ providedIn: 'root' })
export class GuidanceService {
  private readonly storage = inject(StorageService);
  private readonly entries = this.storage.readRecord<GuidanceEntry>(GUIDANCE_STORAGE_KEY);
  private readonly tourRequestSubject = new Subject<string>();

  readonly tourRequests$ = this.tourRequestSubject.asObservable();

  getTour(id: string): GuidanceDefinition | null {
    return GUIDANCE_TOURS[id] ?? null;
  }

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
