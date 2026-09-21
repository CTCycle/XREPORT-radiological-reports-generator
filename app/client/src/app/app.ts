import { Component, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { StartupScreenComponent } from './components/startup-screen.component';
import { StartupReadinessService } from './services/startup-readiness.service';
import { markStartupPhase } from './services/startup-timing';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, StartupScreenComponent],
  template: `
    @if (showApplication()) {
      <router-outlet (activate)="onRouteActivated()" />
    } @else {
      <app-startup-screen
        [phase]="readiness.phase()"
        (retryRequested)="readiness.retry()"
        (transitionComplete)="onStartupTransitionComplete()"
      />
    }
  `,
})
export class App {
  readonly readiness = inject(StartupReadinessService);
  readonly showApplication = signal(false);
  private routeActivated = false;

  constructor() {
    this.readiness.start();
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
