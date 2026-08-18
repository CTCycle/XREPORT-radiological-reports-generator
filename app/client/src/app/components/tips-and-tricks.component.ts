import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, inject } from '@angular/core';
import { Router } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideAlertTriangle, lucideBarChart2, lucideBrainCircuit, lucideFileSearch, lucideFileStack, lucideX } from '@ng-icons/lucide';
import { GuidanceService, INFERENCE_TOUR_ID } from '../services/guidance.service';
import { ModalFocusDirective } from './modal-focus.directive';

interface TipCard {
  icon: string;
  title: string;
  body: string;
  route?: string;
  routeLabel?: string;
  tone?: 'warning';
}

@Component({
  standalone: true,
  selector: 'app-tips-and-tricks',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideAlertTriangle, lucideBarChart2, lucideBrainCircuit, lucideFileSearch, lucideFileStack, lucideX })],
  template: `
    @if (open) {
      <div class="guidance-modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="guidance-modal" role="dialog" aria-modal="true" aria-labelledby="tips-and-tricks-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="guidance-modal-header">
            <div>
              <h2 id="tips-and-tricks-title">Tips &amp; Tricks</h2>
              <p class="guidance-modal-subtitle">Short reminders for the workflows you use most.</p>
            </div>
            <button type="button" class="guidance-close-button" aria-label="Close Tips and Tricks" (click)="closed.emit()">
              <ng-icon name="lucideX" size="17" aria-hidden="true" />
            </button>
          </header>

          <div class="guidance-modal-body">
            <div class="guidance-tips-grid">
              @for (tip of tips; track tip.title) {
                <article class="guidance-tip-card" [class.guidance-tip-card-warning]="tip.tone === 'warning'">
                  <div class="guidance-tip-card-header">
                    <ng-icon [name]="tip.icon" size="17" aria-hidden="true" />
                    <h3>{{ tip.title }}</h3>
                  </div>
                  <p>{{ tip.body }}</p>
                  @if (tip.route) {
                    <div class="guidance-tip-card-actions">
                      <button type="button" class="guidance-link-button" (click)="openRoute(tip.route!)">{{ tip.routeLabel || 'Open page' }}</button>
                    </div>
                  }
                </article>
              }
            </div>

            <article class="guidance-tip-card guidance-tip-card-wide">
              <div class="guidance-tip-card-header">
                <ng-icon name="lucideFileSearch" size="17" aria-hidden="true" />
                <h3>Inference walkthrough</h3>
              </div>
              <p>Use the walkthrough for a guided pass through model selection, image setup, generation, and draft review.</p>
              <div class="guidance-tip-card-actions">
                <button type="button" class="guidance-button guidance-button-primary" (click)="startInferenceTour()">Show walkthrough</button>
              </div>
            </article>
          </div>

          <footer class="guidance-modal-footer">
            <button type="button" class="guidance-button" (click)="closed.emit()">Close</button>
          </footer>
        </section>
      </div>
    }
  `,
  styleUrl: '../styles/Guidance.css',
})
export class TipsAndTricksComponent {
  private readonly router = inject(Router);
  private readonly guidance = inject(GuidanceService);

  @Input() open = false;
  @Output() readonly closed = new EventEmitter<void>();

  readonly tips: readonly TipCard[] = [
    {
      icon: 'lucideFileSearch',
      title: 'Read the model contract',
      body: 'Model cards show anatomy, image limits, supported context, output sections, and readiness before you generate.',
      route: '/inference',
      routeLabel: 'Open Inference',
    },
    {
      icon: 'lucideFileStack',
      title: 'Build in order',
      body: 'Load the image folder and report file first. Then select exactly one dataset row before building a processed dataset.',
      route: '/dataset',
      routeLabel: 'Open Dataset',
    },
    {
      icon: 'lucideBrainCircuit',
      title: 'Use checkpoints deliberately',
      body: 'Start new training from a processed dataset; use Resume for continuation and Evaluate Model to inspect a saved checkpoint.',
      route: '/training',
      routeLabel: 'Open Training',
    },
    {
      icon: 'lucideBarChart2',
      title: 'Sample validation while exploring',
      body: 'Use a fraction for quick checks and the full dataset for final validation decisions.',
      route: '/dataset',
      routeLabel: 'Open Dataset',
    },
    {
      icon: 'lucideAlertTriangle',
      title: 'Treat output as a draft',
      body: 'Generated text remains an editable research-use draft. Inspect the returned provenance before copying or exporting it.',
      tone: 'warning',
    },
  ];

  openRoute(route: string): void {
    this.closed.emit();
    void this.router.navigateByUrl(route);
  }

  startInferenceTour(): void {
    this.closed.emit();
    this.guidance.requestTour(INFERENCE_TOUR_ID);
  }
}
