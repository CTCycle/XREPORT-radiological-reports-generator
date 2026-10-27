import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';
import { CheckpointExchangeApiService } from '../services/checkpoint-exchange-api.service';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import { routes } from '../app.routes';
import { studioRoutes } from '../../studio/studio.routes';
import { CheckpointImportComponent } from './checkpoint-import.component';

function setup(startImport: ReturnType<typeof vi.fn>, finalStatus?: object) {
  TestBed.configureTestingModule({
    providers: [
      { provide: CheckpointExchangeApiService, useValue: { startImport } },
      { provide: JobsApiService, useValue: { get: vi.fn(), cancel: vi.fn() } },
      { provide: JobPollingService, useValue: { poll: () => of(finalStatus) } },
    ],
  });
  const fixture = TestBed.createComponent(CheckpointImportComponent);
  fixture.componentRef.setInput('open', true);
  fixture.componentInstance.file = new File(['x'], 'run.xreportckpt');
  return fixture;
}

describe('CheckpointImportComponent', () => {
  it('asks for another name when the archive conflicts with an installed checkpoint', async () => {
    const fixture = setup(vi.fn().mockResolvedValue({ result: null, error: "409 Conflict: Import it under another name, for example 'run_2'." }));
    await fixture.componentInstance.start();
    fixture.detectChanges();

    expect(fixture.componentInstance.conflict()).toBe(true);
    expect(fixture.componentInstance.busy()).toBe(false);
    expect(fixture.nativeElement.textContent).toContain('run_2');
    expect(fixture.nativeElement.querySelector('input[type="text"]')).not.toBeNull();
  });

  it('reports the installed checkpoint when the import job completes', async () => {
    const fixture = setup(
      vi.fn().mockResolvedValue({ result: { job_id: 'j1', poll_interval: 1 }, error: null }),
      { job_id: 'j1', job_type: 'checkpoint_import', status: 'completed', poll_interval: 1, progress: 100, result: { import: { checkpoint: 'run1', status: 'imported' } }, error: null },
    );
    const imported = vi.fn();
    fixture.componentInstance.imported.subscribe(imported);
    await fixture.componentInstance.start();

    expect(imported).toHaveBeenCalledWith('run1');
    expect(fixture.componentInstance.message()).toContain('Imported run1');
  });

  it('shows the failure reason and installs nothing when the job fails', async () => {
    const fixture = setup(
      vi.fn().mockResolvedValue({ result: { job_id: 'j2', poll_interval: 1 }, error: null }),
      { job_id: 'j2', job_type: 'checkpoint_import', status: 'failed', poll_interval: 1, progress: 40, result: null, error: 'Checksum mismatch for saved_model.keras' },
    );
    const imported = vi.fn();
    fixture.componentInstance.imported.subscribe(imported);
    await fixture.componentInstance.start();

    expect(imported).not.toHaveBeenCalled();
    expect(fixture.componentInstance.failed()).toBe(true);
    expect(fixture.componentInstance.message()).toContain('Checksum mismatch');
  });
});

describe('application route sets', () => {
  const paths = (set: typeof routes) => (set[0].children ?? []).map((route) => route.path);

  it('keeps development pages out of the XREPORT routes and inference out of Studio', () => {
    expect(paths(routes)).toEqual(['', 'inference', 'reports', 'reports/:requestId', 'settings', '**']);
    expect(paths(studioRoutes)).toEqual(['', 'dataset', 'dataset/validate/:datasetName', 'model', 'training', 'settings', '**']);
    expect(studioRoutes[0].data?.['application']).toBe('studio');
  });
});
