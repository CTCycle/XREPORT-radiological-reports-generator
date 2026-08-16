import { Component, DestroyRef, ElementRef, ViewChild, inject } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideBrainCircuit, lucideFileSearch, lucideFileStack, lucideSettings } from '@ng-icons/lucide';

@Component({
  selector: 'app-main-layout',
  imports: [RouterLink, RouterLinkActive, RouterOutlet, NgIcon],
  providers: [provideIcons({ lucideBrainCircuit, lucideFileSearch, lucideFileStack, lucideSettings })],
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
          <button type="button" class="app-nav-button app-nav-settings" title="Settings" aria-label="Settings" disabled>
            <ng-icon name="lucideSettings" size="16" /> <span>Settings</span>
          </button>
        </nav>
      </div>
      <div #content class="main-layout-content" (scroll)="onContentScroll($event)"><router-outlet /></div>
    </div>
  `,
  styleUrl: '../styles/MainLayout.css',
})
export class MainLayoutComponent {
  @ViewChild('content', { static: true }) private readonly content?: ElementRef<HTMLElement>;
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      queueMicrotask(() => { if (this.content) this.content.nativeElement.scrollTop = 0; });
    });
  }

  onContentScroll(event: Event): void {
    const element = event.target as HTMLElement;
    // Reset the content scroll container defensively when the browser reports a negative offset.
    if (element.scrollTop < 0) element.scrollTop = 0;
  }
}
