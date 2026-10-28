import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ValidationApiService } from '../services/validation-api.service';
import type { ValidationReport, ValidationRunSummary } from '../types/validationApi';
import { ValidationStatisticsComponent } from './validation-statistics.component';

@Component({
  standalone: true,
  imports: [ValidationStatisticsComponent],
  template: '<app-validation-statistics datasetName="chest" [refreshToken]="token" />',
})
class HostComponent { token = 0; }

function run(id: number, overrides: Partial<ValidationRunSummary> = {}): ValidationRunSummary {
  return { validation_run_id: id, date: `2026-10-0${id} 10:00:00`, sample_size: 0.5, seed: 7, records_requested: 6, records_evaluated: 6, text_vocabulary_size: 25, tokenizer: null, max_report_size: null, metrics: [{ metric: 'text_statistics', records_computed: 6 }], ...overrides };
}

function report(summary: ValidationRunSummary): ValidationReport {
  return {
    dataset_name: 'chest',
    run: summary,
    aggregates: [{ domain: 'text', measure: 'words_per_report', count: 6, total: 60, minimum: 4, maximum: 20, mean: 10, median: 9, std: null, p05: null, p25: 6, p75: 12, p95: null }],
    histograms: [{ name: 'report_length', bins: [{ lower: 0, upper: 10, count: 4 }, { lower: 10, upper: 20, count: 2 }] }],
  };
}

async function render(api: Partial<Record<keyof ValidationApiService, unknown>>) {
  await TestBed.configureTestingModule({ imports: [HostComponent], providers: [{ provide: ValidationApiService, useValue: api }] }).compileComponents();
  const fixture = TestBed.createComponent(HostComponent);
  fixture.detectChanges();
  await settle(fixture);
  return fixture;
}

async function settle(fixture: { detectChanges(): void }) {
  for (let pass = 0; pass < 3; pass += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    fixture.detectChanges();
  }
}

describe('ValidationStatisticsComponent', () => {
  it('treats a dataset that was never validated as an empty state', async () => {
    const fixture = await render({ listRuns: () => Promise.resolve({ result: null, error: '404 Not Found: Dataset not found' }) });

    expect(fixture.nativeElement.textContent).toContain('No validation has been run for this dataset yet.');
    expect(fixture.nativeElement.querySelector('[role="alert"]')).toBeNull();
  });

  it('shows unavailable values as dashes, computed ones with units, and explains skipped metrics', async () => {
    const summary = run(2);
    const api = {
      listRuns: () => Promise.resolve({ result: { dataset_name: 'chest', runs: [summary] }, error: null }),
      getRun: () => Promise.resolve({ result: report(summary), error: null }),
      getRecords: () => Promise.resolve({ result: { total: 0, rows: [] }, error: null }),
    };
    const fixture = await render(api);
    const text = fixture.nativeElement.textContent as string;

    const row = [...fixture.nativeElement.querySelectorAll('tr')].find((item: HTMLElement) => item.textContent?.includes('Words per report')) as HTMLElement;
    const cells = [...row.querySelectorAll('td')].map((cell) => cell.textContent?.trim());
    expect(cells).toContain('—'); // std, P5 and P95 were not computed and must not read as 0
    expect(cells.some((cell) => /^9[.,]00$/.test(cell ?? ''))).toBe(true); // median
    expect(text).toContain('Total words across 6 reports: 60');
    expect(text).toContain('Image statistics');
    expect(text).toContain('Not computed in this run.');
    expect(fixture.nativeElement.querySelectorAll('.vstats-bar').length).toBe(2);
    expect(fixture.nativeElement.querySelector('.report-chip, .report-metric-pill')).toBeNull();
  });

  it('loads the selected historical run and its record page', async () => {
    const [latest, older] = [run(2), run(1, { sample_size: 1 })];
    const getRun = vi.fn((_: string, id: number) => Promise.resolve({ result: report(id === 2 ? latest : older), error: null }));
    const getRecords = vi.fn().mockResolvedValue({ result: { total: 1, rows: [{ record_id: 3, image_name: 'a.png', word_count: 4, char_count: 20, unique_words: 4, token_count: null }] }, error: null });
    const fixture = await render({ listRuns: () => Promise.resolve({ result: { dataset_name: 'chest', runs: [latest, older] }, error: null }), getRun, getRecords });

    expect(getRun).toHaveBeenLastCalledWith('chest', 2);
    const button = [...fixture.nativeElement.querySelectorAll('.vstats-history button')].find((item: HTMLElement) => item.textContent?.includes('#1')) as HTMLButtonElement;
    button.click();
    await settle(fixture);

    expect(getRun).toHaveBeenLastCalledWith('chest', 1);
    expect(getRecords).toHaveBeenLastCalledWith('chest', 1, expect.objectContaining({ kind: 'text', offset: 0 }));
    expect(fixture.nativeElement.textContent).toContain('a.png');
    expect(fixture.nativeElement.textContent).toContain('—'); // token_count is null for this run
  });
});
