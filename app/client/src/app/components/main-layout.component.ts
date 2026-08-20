import { Component, DestroyRef, ElementRef, ViewChild, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideBrainCircuit, lucideCircleHelp, lucideFileSearch, lucideFileStack, lucideSettings } from '@ng-icons/lucide';
import { GuidanceService } from '../services/guidance.service';
import type { GuidanceDefinition } from '../types/guidance';
import { GuidedTourComponent } from './guided-tour.component';
import { ThemeSelectorComponent } from './theme-selector.component';
import { TipsAndTricksComponent } from './tips-and-tricks.component';

@Component({
  selector: 'app-main-layout',
  imports: [RouterLink, RouterLinkActive, RouterOutlet, NgIcon, GuidedTourComponent, ThemeSelectorComponent, TipsAndTricksComponent],
  providers: [provideIcons({ lucideBrainCircuit, lucideCircleHelp, lucideFileSearch, lucideFileStack, lucideSettings })],
  template: `
    <div class="main-layout">
      <div class="main-layout-chrome">
        <nav class="app-nav-bar" aria-label="Primary navigation">
          <div class="app-nav-brand">
            <img class="app-nav-logo" src="/favicon.png" alt="XREPORT logo" />
            <div class="app-nav-titles">
              <h1 class="app-nav-title">XREPORT</h1>
              <p class="app-nav-subtitle">Radiological Reports Generator</p>
            </div>
          </div>
          <div class="app-nav-list">
            <a routerLink="/inference" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }" class="app-nav-button app-nav-button-primary" title="Inference" aria-label="Inference">
              <ng-icon name="lucideFileSearch" size="16" /> <span>Inference</span>
            </a>
            <span class="app-nav-separator" aria-hidden="true"></span>
            <span class="app-nav-group-label">Model development</span>
            <a routerLink="/dataset" routerLinkActive="active" class="app-nav-button" title="Dataset" aria-label="Dataset">
              <ng-icon name="lucideFileStack" size="16" /> <span>Dataset</span>
            </a>
            <a routerLink="/training" routerLinkActive="active" class="app-nav-button" title="Training" aria-label="Training">
              <ng-icon name="lucideBrainCircuit" size="16" /> <span>Training</span>
            </a>
          </div>
          <div class="app-nav-footer">
            <button type="button" class="app-nav-button app-nav-help" title="Help and tips" aria-label="Help and tips" aria-haspopup="dialog" [attr.aria-expanded]="tipsOpen()" (click)="tipsOpen.set(true)">
              <ng-icon name="lucideCircleHelp" size="16" /> <span>Help and tips</span>
            </button>
            <button type="button" class="app-nav-button app-nav-settings" title="Settings" aria-label="Settings" disabled>
              <ng-icon name="lucideSettings" size="16" /> <span>Settings</span>
            </button>
            <app-theme-selector />
          </div>
        </nav>
      </div>
      <div #content class="main-layout-content" (scroll)="onContentScroll($event)"><router-outlet /></div>
      @if (tipsOpen()) { <app-tips-and-tricks [open]="true" (closed)="tipsOpen.set(false)" /> }
      @if (activeTour(); as tour) { <app-guided-tour [definition]="tour" [open]="true" (closed)="activeTour.set(null)" /> }
    </div>
  `,
  styleUrl: '../styles/MainLayout.css',
})
export class MainLayoutComponent {
  @ViewChild('content', { static: true }) private readonly content?: ElementRef<HTMLElement>;
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  private readonly guidance = inject(GuidanceService);
  readonly tipsOpen = signal(false);
  readonly activeTour = signal<GuidanceDefinition | null>(null);

  constructor() {
    this.guidance.tourRequests$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((tourId) => this.handleTourRequest(tourId));
    this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      const tour = this.activeTour();
      if (tour && !tour.steps.some((step) => this.router.url.startsWith(step.route ?? tour.route))) this.activeTour.set(null);
      queueMicrotask(() => { if (this.content) this.content.nativeElement.scrollTop = 0; });
    });
  }

  private async handleTourRequest(tourId: string): Promise<void> {
    const tour = this.guidance.getTour(tourId);
    if (!tour) return;
    this.tipsOpen.set(false);
    const firstRoute = tour.steps[0]?.route ?? tour.route;
    if (!this.router.url.startsWith(firstRoute)) await this.router.navigateByUrl(firstRoute);
    setTimeout(() => this.activeTour.set(tour), 0);
  }

  onContentScroll(event: Event): void {
    const element = event.target as HTMLElement;
    // Reset the content scroll container defensively when the browser reports a negative offset.
    if (element.scrollTop < 0) element.scrollTop = 0;
  }
}
