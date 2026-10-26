import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { vi } from 'vitest';
import { AppStateService } from '../services/app-state.service';
import { InferenceApiService } from '../services/inference-api.service';
import { SystemStatusService } from '../services/system-status.service';
import { SystemStatusBarComponent } from './system-status-bar.component';

describe('SystemStatusBarComponent', () => {
  const releaseRuntime = vi.fn(async () => ({ result: { released: true, model_ref: 'huggingface:model' }, error: null }));
  const refresh = vi.fn(async () => undefined);

  beforeEach(async () => {
    releaseRuntime.mockClear();
    refresh.mockClear();
    await TestBed.configureTestingModule({
      imports: [SystemStatusBarComponent],
      providers: [
        { provide: InferenceApiService, useValue: { releaseRuntime } },
        { provide: SystemStatusService, useValue: { backendOnline: signal(true), status: signal(null), refresh } },
      ],
    }).compileComponents();
  });

  function button(fixture: { nativeElement: HTMLElement }): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('.system-status-action');
  }

  it('releases model memory on demand and refreshes the RAM and VRAM readings', async () => {
    const fixture = TestBed.createComponent(SystemStatusBarComponent);
    fixture.detectChanges();

    button(fixture)!.click();
    await fixture.whenStable();

    expect(releaseRuntime).toHaveBeenCalledOnce();
    expect(refresh).toHaveBeenCalledOnce();
  });

  it('disables the release while inference is running', async () => {
    TestBed.inject(AppStateService).updateInference((state) => ({ ...state, isGenerating: true }));
    const fixture = TestBed.createComponent(SystemStatusBarComponent);
    fixture.detectChanges();

    expect(button(fixture)!.disabled).toBe(true);
    await fixture.componentInstance.freeMemory();
    expect(releaseRuntime).not.toHaveBeenCalled();
  });

  it('is not offered in Studio', () => {
    const fixture = TestBed.createComponent(SystemStatusBarComponent);
    fixture.componentRef.setInput('studio', true);
    fixture.detectChanges();

    expect(button(fixture)).toBeNull();
  });
});
