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

  it('renders XREPORT branding and structural report placeholders without clinical findings', async () => {
    const fixture = await createComponent();
    const element = fixture.nativeElement as HTMLElement;

    expect(element.textContent).toContain('XREPORT');
    expect(element.textContent).toContain('Radiological Reports Generator');
    expect(element.textContent).toContain('FINDINGS');
    expect(element.textContent).toContain('IMPRESSION');
    expect(element.textContent).not.toMatch(/pneumonia|fracture|mass|effusion/i);
    const image = element.querySelector<HTMLImageElement>('.startup-xray-image');
    expect(image?.getAttribute('src')).toBe('startup-radiograph.png');
    expect(image?.getAttribute('alt')).toBe('');
    expect(image?.getAttribute('aria-hidden')).toBe('true');
    expect(element.querySelector('.startup-visual')?.getAttribute('aria-hidden')).toBe('true');
    expect(element.querySelector('.startup-xray-scan')?.getAttribute('aria-hidden')).toBe('true');
    expect(element.querySelector('svg')).toBeNull();
  });

  it('shows the phase copy and exposes retry only when unavailable', async () => {
    const fixture = await createComponent();
    fixture.componentRef.setInput('phase', 'slow');
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('XREPORT is still initializing');
    expect(fixture.nativeElement.querySelector('.startup-retry')).toBeNull();

    fixture.componentRef.setInput('phase', 'unavailable');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.startup-status')?.getAttribute('role')).toBe('alert');
    expect(fixture.nativeElement.querySelector('.startup-retry')).not.toBeNull();
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
