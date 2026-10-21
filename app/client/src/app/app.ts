import { Component, effect, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { StartupScreenComponent } from './components/startup-screen.component';
import { SettingsPage } from './pages/settings.page';
import { StartupReadinessService } from './services/startup-readiness.service';
import { ThemeService } from './services/theme.service';
import { markStartupPhase } from './services/startup-timing';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, SettingsPage, StartupScreenComponent],
  template: `
    @if (showApplication()) {
      <router-outlet (activate)="onRouteActivated()" />
    } @else if (readiness.phase() === 'recovery') {
      <app-settings-page [databaseOnly]="true" (databaseActivated)="readiness.retry()" />
    } @else {
      <app-startup-screen
        [phase]="readiness.phase()"
        [startedAt]="readiness.startedAt()"
        (retryRequested)="readiness.retry()"
        (transitionComplete)="onStartupTransitionComplete()"
      />
    }
  `,
})
export class App {
  readonly readiness = inject(StartupReadinessService);
  private readonly theme = inject(ThemeService);
  readonly showApplication = signal(false);
  private routeActivated = false;

  constructor() {
    this.readiness.start();
    effect(() => {
      if (this.readiness.phase() === 'ready') void this.theme.loadFromDatabase();
    });
  }

  onStartupTransitionComplete(): void {
    if (this.showApplication()) return;
    this.showApplication.set(true);
    markStartupPhase('startup_transition_completed');
  }

  onRouteActivated(): void {
    if (this.routeActivated) return;
    this.routeActivated = true;
    markStartupPhase('first_route_activated');
  }
}
