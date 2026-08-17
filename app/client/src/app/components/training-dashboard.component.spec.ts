import { TestBed } from '@angular/core/testing';
import type { TrainingDashboardState } from '../types';
import { TrainingDashboardComponent } from './training-dashboard.component';

const createState = (overrides: Partial<TrainingDashboardState> = {}): TrainingDashboardState => ({
  isTraining: false,
  currentEpoch: 0,
  totalEpochs: 10,
  loss: 0.1234,
  valLoss: 0.2345,
  accuracy: 0.8,
  valAccuracy: 0.7,
  progressPercent: 0,
  elapsedSeconds: 0,
  chartData: [],
  availableMetrics: [],
  epochBoundaries: [],
  logEntries: [],
  ...overrides,
});

describe('TrainingDashboardComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingDashboardComponent],
    }).compileComponents();
  });

  it('renders the idle dashboard with metric cards and two chart placeholders', () => {
    const fixture = TestBed.createComponent(TrainingDashboardComponent);
    fixture.componentRef.setInput('dashboardState', createState());
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelectorAll('.dashboard-metric-card')).toHaveLength(5);
    expect(element.querySelectorAll('.chart-placeholder')).toHaveLength(2);
    expect(element.querySelector('[role="status"]')?.textContent).toContain('Idle');
    expect(element.querySelector<HTMLButtonElement>('.btn-stop')?.disabled).toBe(true);
    expect(element.querySelector('.log-empty')?.textContent).toContain('No training output yet.');
  });

  it('renders running metrics, progress, charts, elapsed time, and log output', () => {
    const fixture = TestBed.createComponent(TrainingDashboardComponent);
    fixture.componentRef.setInput('dashboardState', createState({
      isTraining: true,
      currentEpoch: 3,
      progressPercent: 42,
      elapsedSeconds: 3661,
      chartData: [
        { batch: 1, loss: 0.9, val_loss: 1.1, MaskedAccuracy: 0.4, val_MaskedAccuracy: 0.3 },
        { batch: 2, loss: 0.6, val_loss: 0.8, MaskedAccuracy: 0.6, val_MaskedAccuracy: 0.5 },
      ],
      availableMetrics: ['loss', 'val_loss', 'MaskedAccuracy', 'val_MaskedAccuracy'],
      epochBoundaries: [1],
      logEntries: ['Training job started (job-1).'],
    }));
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('[role="status"]')?.textContent).toContain('Training in progress');
    expect(element.querySelector('.progress-bar')?.getAttribute('style')).toContain('width: 42%');
    expect(element.querySelector('.progress-time')?.textContent).toContain('1h 1m 1s');
    expect(element.querySelectorAll('.training-chart')).toHaveLength(2);
    expect(element.querySelectorAll('.chart-legend span')).toHaveLength(4);
    expect(element.querySelector('.log-body')?.textContent).toContain('job-1');
    expect(element.querySelector<HTMLButtonElement>('.btn-stop')?.disabled).toBe(false);
    expect(element.querySelectorAll('.metric-value.accuracy')[0]?.textContent).toContain('80.000%');
  });

  it('shows complete status and emits a stop request only when enabled', () => {
    const fixture = TestBed.createComponent(TrainingDashboardComponent);
    fixture.componentRef.setInput('dashboardState', createState({ currentEpoch: 10, progressPercent: 100 }));
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('[role="status"]')?.textContent).toContain('Complete');
    expect(element.querySelector<HTMLButtonElement>('.btn-stop')?.disabled).toBe(true);

    fixture.componentRef.setInput('dashboardState', createState({ isTraining: true, currentEpoch: 1 }));
    fixture.detectChanges();
    let emitted = 0;
    fixture.componentInstance.stopRequested.subscribe(() => emitted += 1);
    element.querySelector<HTMLButtonElement>('.btn-stop')?.click();
    expect(emitted).toBe(1);
  });

  it('shows failed status when the training workflow reports an error', () => {
    const fixture = TestBed.createComponent(TrainingDashboardComponent);
    fixture.componentRef.setInput('dashboardState', createState({ currentEpoch: 1 }));
    fixture.componentRef.setInput('error', 'Current dataset metadata does not match checkpoint.');
    fixture.detectChanges();

    const element = fixture.nativeElement as HTMLElement;
    expect(element.querySelector('[role="status"]')?.textContent).toContain('Failed');
    expect(element.querySelector('.dashboard-status.error')).not.toBeNull();
  });
});
