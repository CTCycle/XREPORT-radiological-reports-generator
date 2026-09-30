import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import type { TrainingDashboardState } from '../types';
import { TrainingDashboardComponent } from './training-dashboard.component';

const dashboardState = (): TrainingDashboardState => ({
  isTraining: false,
  currentEpoch: 0,
  totalEpochs: 1,
  loss: 0,
  valLoss: 0,
  accuracy: 0,
  valAccuracy: 0,
  progressPercent: 0,
  elapsedSeconds: 0,
  workerPhase: 'idle',
  workerPhaseStatus: 'idle',
  workerPhaseElapsedSeconds: 0,
  workerElapsedSeconds: 0,
  chartData: [],
  availableMetrics: [],
  epochBoundaries: [],
  logEntries: [],
});

@Component({
  standalone: true,
  imports: [TrainingDashboardComponent],
  template: '<app-training-dashboard [dashboardState]="state" [error]="error" />',
})
class DashboardHostComponent {
  state = dashboardState();
  error: string | null = null;
}

describe('TrainingDashboardComponent', () => {
  it('shows a distinct zero-progress initialization phase and keeps stop disabled', async () => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Waiting to start');
    expect(fixture.nativeElement.textContent).toContain('0s in phase');
    expect(fixture.nativeElement.querySelector('[role="progressbar"]').getAttribute('aria-valuenow')).toBe('0');
    expect((fixture.nativeElement.querySelector('.btn-stop') as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders the active worker phase and a terminal error accessibly', async () => {
    await TestBed.configureTestingModule({ imports: [DashboardHostComponent] }).compileComponents();

    const fixture = TestBed.createComponent(DashboardHostComponent);
    fixture.componentInstance.state = {
      ...dashboardState(),
      isTraining: true,
      workerPhase: 'model_loading_started',
      workerPhaseStatus: 'started',
      workerPhaseElapsedSeconds: 4,
    };
    fixture.componentInstance.error = 'Training worker stalled during model loading';
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Loading model');
    expect(fixture.nativeElement.textContent).toContain('4s in phase');
    expect(fixture.nativeElement.querySelector('[role="alert"]')?.textContent).toContain('Training worker stalled');
    expect((fixture.nativeElement.querySelector('.btn-stop') as HTMLButtonElement).disabled).toBe(false);
  });
});
