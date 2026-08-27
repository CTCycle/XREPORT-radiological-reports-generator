import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { markStartupPhase } from './services/startup-timing';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  template: '<router-outlet (activate)="onRouteActivated()" />',
})
export class App {
  private routeActivated = false;

  onRouteActivated(): void {
    if (this.routeActivated) return;
    this.routeActivated = true;
    markStartupPhase('first_route_activated');
  }
}
