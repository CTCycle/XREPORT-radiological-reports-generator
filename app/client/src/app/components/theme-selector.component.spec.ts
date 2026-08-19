import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ThemeSelectorComponent } from './theme-selector.component';

@Component({
  standalone: true,
  imports: [ThemeSelectorComponent],
  template: '<app-theme-selector />',
})
class ThemeSelectorHostComponent {}

describe('ThemeSelectorComponent', () => {
  beforeEach(async () => {
    localStorage.clear();
    await TestBed.configureTestingModule({ imports: [ThemeSelectorHostComponent] }).compileComponents();
  });

  afterEach(() => localStorage.clear());

  it('renders all preferences with accessible selected-state controls', () => {
    const fixture = TestBed.createComponent(ThemeSelectorHostComponent);
    fixture.detectChanges();
    const buttons = [...fixture.nativeElement.querySelectorAll('button')] as HTMLButtonElement[];

    expect(buttons.map((button) => button.textContent?.trim())).toEqual(['Light', 'Dark', 'System']);
    expect(buttons.map((button) => button.getAttribute('aria-pressed'))).toEqual(['false', 'false', 'true']);
    expect(buttons[0].getAttribute('aria-label')).toBe('Use Light theme');
  });

  it('changes the service preference and root theme when selected', () => {
    const fixture = TestBed.createComponent(ThemeSelectorHostComponent);
    fixture.detectChanges();
    const buttons = [...fixture.nativeElement.querySelectorAll('button')] as HTMLButtonElement[];

    buttons[1].click();
    fixture.detectChanges();

    expect(buttons[1].getAttribute('aria-pressed')).toBe('true');
    expect(buttons[2].getAttribute('aria-pressed')).toBe('false');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('theme-preference')).toBe('dark');
  });
});
