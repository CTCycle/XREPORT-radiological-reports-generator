import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { StartupScreenComponent } from './startup-screen.component';

describe('StartupScreenComponent', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function createComponent() {
    await TestBed.configureTestingModule({ imports: [StartupScreenComponent] }).compileComponents();
    const fixture = TestBed.createComponent(StartupScreenComponent);
    fixture.detectChanges();
    return fixture;
  }

  it('renders the XREPORT logo and an accessible progress bar without extra copy', async () => {
    const fixture = await createComponent();
    const element = fixture.nativeElement as HTMLElement;

    expect(element.textContent).toContain('XREPORT');
    expect(element.textContent).not.toMatch(/Radiological Reports Generator|pneumonia|fracture|mass|effusion|FINDINGS|IMPRESSION/i);
    const logo = element.querySelector<HTMLImageElement>('.startup-logo');
    expect(logo?.getAttribute('src')).toBe('favicon.png');
    expect(logo?.getAttribute('alt')).toBe('');
    expect(logo?.getAttribute('aria-hidden')).toBe('true');
    const progress = element.querySelector<HTMLElement>('.startup-progress');
    expect(progress?.getAttribute('role')).toBe('progressbar');
    expect(progress?.getAttribute('aria-label')).toBe('Initializing XREPORT');
    expect(progress?.getAttribute('aria-valuemin')).toBe('0');
    expect(progress?.getAttribute('aria-valuemax')).toBe('100');
    expect(element.querySelector('.startup-progress-fill')).not.toBeNull();
    expect(element.querySelector('.startup-status-line')).not.toBeNull();
    expect(element.querySelector('svg')).toBeNull();
  });

  it('shows a single minimal status line and exposes retry only when unavailable', async () => {
    const fixture = await createComponent();
    const line = fixture.nativeElement.querySelector('.startup-status-line');

    fixture.componentRef.setInput('phase', 'slow');
    fixture.detectChanges();
    expect(line.textContent.trim()).toMatch(/Validating|Preparing|Starting|Connecting/);
    expect(fixture.nativeElement.querySelector('.startup-retry')).toBeNull();

    fixture.componentRef.setInput('phase', 'unavailable');
    fixture.detectChanges();
    expect(line.textContent.trim()).toBe('Waiting for the local service');
    expect(fixture.nativeElement.querySelector('.startup-retry')).not.toBeNull();
  });

  it('advances progress monotonically from the started timestamp and completes only when ready', async () => {
    const fixture = await createComponent();
    const progressbar = fixture.nativeElement.querySelector('.startup-progress');
    const fill = fixture.nativeElement.querySelector('.startup-progress-fill');

    fixture.componentRef.setInput('startedAt', Date.now());
    fixture.detectChanges();
    expect(Number(progressbar.getAttribute('aria-valuenow'))).toBe(0);

    vi.advanceTimersByTime(10_000);
    fixture.detectChanges();
    const mid = Number(progressbar.getAttribute('aria-valuenow'));
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(100);
    expect(fill.style.width).toBe(`${mid}%`);

    vi.advanceTimersByTime(20_000);
    fixture.detectChanges();
    const later = Number(progressbar.getAttribute('aria-valuenow'));
    expect(later).toBeGreaterThanOrEqual(mid);
    expect(later).toBeLessThan(100);

    fixture.componentRef.setInput('phase', 'ready');
    fixture.detectChanges();
    expect(Number(progressbar.getAttribute('aria-valuenow'))).toBe(100);
    expect(fill.style.width).toBe('100%');
    expect(fixture.nativeElement.querySelector('.startup-status-line').textContent.trim()).toBe(
      'Opening your workspace',
    );
  });

  it('labels the current preparation step and freezes progress when unavailable', async () => {
    const fixture = await createComponent();
    const progressbar = fixture.nativeElement.querySelector('.startup-progress');
    const line = fixture.nativeElement.querySelector('.startup-status-line');

    fixture.componentRef.setInput('startedAt', Date.now());
    fixture.detectChanges();
    vi.advanceTimersByTime(4_000);
    fixture.detectChanges();
    expect(line.textContent.trim()).toMatch(/Validating|Preparing|Starting|Connecting/);

    const frozen = Number(progressbar.getAttribute('aria-valuenow'));
    fixture.componentRef.setInput('phase', 'unavailable');
    fixture.detectChanges();
    expect(line.textContent.trim()).toBe('Waiting for the local service');

    vi.advanceTimersByTime(10_000);
    fixture.detectChanges();
    expect(Number(progressbar.getAttribute('aria-valuenow'))).toBe(frozen);
  });

  it('emits retry and completes the ready transition exactly once', async () => {
    const fixture = await createComponent();
    const retry = vi.fn();
    const completed = vi.fn();
    fixture.componentInstance.retryRequested.subscribe(retry);
    fixture.componentInstance.transitionComplete.subscribe(completed);

    fixture.componentRef.setInput('phase', 'unavailable');
    fixture.detectChanges();
    (fixture.nativeElement.querySelector('.startup-retry') as HTMLButtonElement).click();
    expect(retry).toHaveBeenCalledOnce();

    fixture.componentRef.setInput('phase', 'ready');
    fixture.detectChanges();
    vi.advanceTimersByTime(400);
    expect(completed).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(400);
    expect(completed).toHaveBeenCalledOnce();
  });
});