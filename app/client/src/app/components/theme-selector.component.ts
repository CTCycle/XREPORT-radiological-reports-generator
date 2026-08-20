import { Component, inject } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideMonitor, lucideMoon, lucideSun } from '@ng-icons/lucide';
import { ThemeService } from '../services/theme.service';
import type { ThemePreference } from '../types/theme';

interface ThemeOption {
  readonly value: ThemePreference;
  readonly label: string;
  readonly icon: string;
}

@Component({
  selector: 'app-theme-selector',
  imports: [NgIcon],
  providers: [provideIcons({ lucideMonitor, lucideMoon, lucideSun })],
  template: `
    <fieldset class="theme-selector" aria-label="Theme preference">
      <p class="sr-only">Choose the application color theme.</p>
      <div class="theme-selector-options" role="group" aria-label="Theme preference">
        @for (option of options; track option.value) {
          <button
            type="button"
            class="theme-selector-option"
            [class.selected]="theme.preference() === option.value"
            [attr.aria-label]="'Use ' + option.label + ' theme'"
            [attr.aria-pressed]="theme.preference() === option.value"
            [attr.title]="'Use ' + option.label + ' theme'"
            (click)="setPreference(option.value)"
          >
            <ng-icon [name]="option.icon" size="15" aria-hidden="true" />
            <span class="theme-selector-option-label">{{ option.label }}</span>
          </button>
        }
      </div>
    </fieldset>
  `,
  styleUrl: '../styles/ThemeSelector.css',
})
export class ThemeSelectorComponent {
  readonly theme = inject(ThemeService);
  readonly options: readonly ThemeOption[] = [
    { value: 'light', label: 'Light', icon: 'lucideSun' },
    { value: 'dark', label: 'Dark', icon: 'lucideMoon' },
    { value: 'system', label: 'System', icon: 'lucideMonitor' },
  ];

  setPreference(preference: ThemePreference): void {
    this.theme.setPreference(preference);
  }
}
