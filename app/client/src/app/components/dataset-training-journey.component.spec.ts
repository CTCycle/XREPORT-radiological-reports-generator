import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ApiService } from '../services/api.service';
import { DatasetTrainingJourneyComponent } from './dataset-training-journey.component';

@Component({
  standalone: true,
  imports: [DatasetTrainingJourneyComponent],
  template: '<app-dataset-training-journey (routeRequested)="route = $event" (walkthroughRequested)="walkthroughStarted = true" />',
})
class JourneyHostComponent {
  route = '';
  walkthroughStarted = false;
}

function apiResult<T>(result: T) {
  return Promise.resolve({ result, error: null });
}

function emptyApi() {
  return {
    getDatasetStatus: () => apiResult({ has_data: false, row_count: 0, allow_server_browse: true, message: 'No data' }),
    getDatasetNames: () => apiResult({ datasets: [], count: 0 }),
    getProcessedDatasetNames: () => apiResult({ datasets: [], count: 0 }),
    getCheckpoints: () => apiResult({ checkpoints: [] }),
    getTrainingStatus: () => apiResult({ is_training: false, current_epoch: 0, total_epochs: 0, loss: 0, val_loss: 0, accuracy: 0, val_accuracy: 0, progress_percent: 0, elapsed_seconds: 0 }),
  };
}

describe('DatasetTrainingJourneyComponent', () => {
  it('starts at source upload and keeps later actions locked when no data exists', async () => {
    await TestBed.configureTestingModule({
      imports: [JourneyHostComponent],
      providers: [{ provide: ApiService, useValue: emptyApi() }],
    }).compileComponents();

    const fixture = TestBed.createComponent(JourneyHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Next: upload and load your source data');
    expect(fixture.nativeElement.querySelectorAll('.guidance-journey-step')).toHaveLength(5);
    expect(fixture.nativeElement.querySelector('.guidance-journey-step.is-current')).toBeTruthy();

    const steps = fixture.nativeElement.querySelectorAll('.guidance-journey-step-toggle');
    (steps[2] as HTMLButtonElement).click();
    fixture.detectChanges();
    const lockedAction = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Complete previous step')) as HTMLButtonElement;
    expect(lockedAction.disabled).toBe(true);

    const walkthrough = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Walk me through it')) as HTMLButtonElement;
    walkthrough.click();
    expect(fixture.componentInstance.walkthroughStarted).toBe(true);
  });

  it('recognizes a processed dataset and exposes the checkpoint branch', async () => {
    const api = {
      getDatasetStatus: () => apiResult({ has_data: true, row_count: 12, allow_server_browse: true, message: 'Found 12 records' }),
      getDatasetNames: () => apiResult({ datasets: [{ name: 'source', folder_path: 'images', row_count: 12, has_validation_report: false }], count: 1 }),
      getProcessedDatasetNames: () => apiResult({ datasets: [{ name: 'source', folder_path: 'processed', row_count: 10, has_validation_report: false }], count: 1 }),
      getCheckpoints: () => apiResult({ checkpoints: [{ name: 'source-epoch-4', epochs: 4, loss: 0.2, val_loss: 0.3 }] }),
      getTrainingStatus: () => apiResult({ is_training: false, current_epoch: 4, total_epochs: 10, loss: 0.2, val_loss: 0.3, accuracy: 0.8, val_accuracy: 0.7, progress_percent: 40, elapsed_seconds: 120 }),
    };

    await TestBed.configureTestingModule({
      imports: [JourneyHostComponent],
      providers: [{ provide: ApiService, useValue: api }],
    }).compileComponents();

    const fixture = TestBed.createComponent(JourneyHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Your checkpoint branch is ready to resume');
    expect(fixture.nativeElement.querySelectorAll('.guidance-journey-step.is-completed')).toHaveLength(4);
    expect(fixture.nativeElement.querySelector('.guidance-journey-step.is-available')).toBeTruthy();

    const openCheckpoint = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Open Checkpoints')) as HTMLButtonElement;
    openCheckpoint.click();
    expect(fixture.componentInstance.route).toBe('/training');
  });
});
