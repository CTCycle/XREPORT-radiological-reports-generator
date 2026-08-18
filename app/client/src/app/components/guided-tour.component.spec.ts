import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { GuidedTourComponent } from './guided-tour.component';
import type { GuidanceDefinition } from '../types/guidance';

@Component({
  standalone: true,
  imports: [GuidedTourComponent],
  template: `
    <button #source>Source</button>
    <div data-guidance-target="first-target">Target</div>
    <app-guided-tour [definition]="definition" [open]="open" (closed)="closedReason = $event" />
  `,
})
class TourHostComponent {
  open = true;
  closedReason = '';
  definition: GuidanceDefinition = {
    id: 'test-tour',
    version: 1,
    route: '/inference',
    steps: [
      { id: 'first', target: '[data-guidance-target="first-target"]', title: 'First step', body: 'Choose the model.' },
      { id: 'second', target: '[data-guidance-target="missing-target"]', title: 'Second step', body: 'Add an image.' },
    ],
  };
}

describe('GuidedTourComponent', () => {
  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({ imports: [TourHostComponent] }).compileComponents();
  });

  it('navigates forward and backward, keeps missing targets usable, and restores focus', async () => {
    const fixture = TestBed.createComponent(TourHostComponent);
    const host = fixture.componentInstance;
    document.body.appendChild(fixture.nativeElement);
    const source = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    source.focus();
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 2));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Step 1 of 2');
    expect(document.body.querySelector('.guided-tour-spotlight')).toBeTruthy();
    const next = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Next')) as HTMLButtonElement;
    next.click();
    fixture.detectChanges();
    await fixture.whenStable();
    await new Promise((resolve) => setTimeout(resolve, 2));
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Step 2 of 2');
    expect(document.body.querySelector('.guided-tour-spotlight')).toBeNull();
    const back = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Back')) as HTMLButtonElement;
    back.click();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Step 1 of 2');

    host.open = false;
    fixture.detectChanges();
    await fixture.whenStable();
    expect(host.closedReason).toBe('');
    expect(document.activeElement).toBe(source);
  });

  it('records skip and finish outcomes', async () => {
    const fixture = TestBed.createComponent(TourHostComponent);
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
    await fixture.whenStable();
    const skip = Array.from(fixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Skip')) as HTMLButtonElement;
    skip.click();
    fixture.detectChanges();
    expect(fixture.componentInstance.closedReason).toBe('skipped');

    const secondFixture = TestBed.createComponent(TourHostComponent);
    document.body.appendChild(secondFixture.nativeElement);
    secondFixture.detectChanges();
    await secondFixture.whenStable();
    const next = Array.from(secondFixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Next')) as HTMLButtonElement;
    next.click();
    secondFixture.detectChanges();
    await secondFixture.whenStable();
    const finish = Array.from(secondFixture.nativeElement.querySelectorAll('button')).find((button) => (button as HTMLElement).textContent?.includes('Finish')) as HTMLButtonElement;
    finish.click();
    expect(secondFixture.componentInstance.closedReason).toBe('completed');
  });
});
