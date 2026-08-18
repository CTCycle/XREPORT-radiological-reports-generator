import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HelpPopoverComponent } from './help-popover.component';

@Component({
  standalone: true,
  imports: [HelpPopoverComponent],
  template: '<app-help-popover title="Model help" body="Model details" />',
})
class HelpHostComponent {}

describe('HelpPopoverComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [HelpHostComponent] }).compileComponents();
  });

  it('opens from the trigger, closes on Escape, and restores focus', async () => {
    const fixture = TestBed.createComponent(HelpHostComponent);
    fixture.detectChanges();
    const trigger = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    trigger.focus();
    trigger.click();
    fixture.detectChanges();
    await fixture.whenStable();

    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(document.body.querySelector('.guidance-popover-panel')).toBeTruthy();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    fixture.detectChanges();
    await fixture.whenStable();

    expect(trigger.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(trigger);
  });
});
