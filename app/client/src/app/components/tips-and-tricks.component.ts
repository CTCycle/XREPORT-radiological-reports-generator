import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, Type, ViewContainerRef, ViewChild, AfterViewInit, inject, signal, InjectionToken } from '@angular/core';
import { Router } from '@angular/router';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideAlertTriangle, lucideBarChart2, lucideBrainCircuit, lucideFileSearch, lucideX } from '@ng-icons/lucide';
import { DATA_TRAINING_TOUR_ID, GuidanceService, INFERENCE_TOUR_ID } from '../services/guidance.service';
import { ModalFocusDirective } from './modal-focus.directive';

interface TipCard {
  icon: string;
  title: string;
  body: string;
  route?: string;
  routeLabel?: string;
  tone?: 'warning';
  scope: 'xreport' | 'studio' | 'both';
}

/** Studio provides the dataset/training journey card; XREPORT never bundles it. */
export interface TipsJourneyComponent {
  routeRequested: { subscribe(next: (route: string) => void): unknown };
  walkthroughRequested: { subscribe(next: () => void): unknown };
}
export const TIPS_JOURNEY = new InjectionToken<Type<TipsJourneyComponent>>('TIPS_JOURNEY');

@Component({
  standalone: true,
  selector: 'app-tips-and-tricks',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideAlertTriangle, lucideBarChart2, lucideBrainCircuit, lucideFileSearch, lucideX })],
  template: `
    @if (open) {
      <div class="guidance-modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="guidance-modal" role="dialog" aria-modal="true" aria-labelledby="tips-and-tricks-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="guidance-modal-header">
            <div>
              <h2 id="tips-and-tricks-title">Tips &amp; Tricks</h2>
              <p class="guidance-modal-subtitle">{{ studio ? 'Build a dataset, then train.' : 'Quick reminders for generating reports.' }}</p>
            </div>
            <button type="button" class="guidance-close-button" aria-label="Close Tips and Tricks" (click)="closed.emit()">
              <ng-icon name="lucideX" size="17" aria-hidden="true" />
            </button>
          </header>

          <div class="guidance-modal-body">
            <ng-container #journeyHost></ng-container>
            <h3 class="guidance-section-heading">Good to know</h3>
            <div class="guidance-tips-grid">
              @for (tip of visibleTips(); track tip.title) {
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

            @if (!studio) {
              <article class="guidance-tip-card guidance-tip-card-wide">
                <div class="guidance-tip-card-header">
                  <ng-icon name="lucideFileSearch" size="17" aria-hidden="true" />
                  <h3>Inference walkthrough</h3>
                </div>
                <p>A guided pass through model selection, image setup, generation and review.</p>
                <div class="guidance-tip-card-actions">
                  <button type="button" class="guidance-button guidance-button-primary" (click)="startInferenceTour()">Show walkthrough</button>
                </div>
              </article>
            }
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
export class TipsAndTricksComponent implements AfterViewInit {
  private readonly router = inject(Router);
  private readonly guidance = inject(GuidanceService);

  private readonly journey = inject(TIPS_JOURNEY, { optional: true });
  private readonly studioFlag = signal(false);
  readonly visibleTips = () =>
    this.tips.filter((tip) => tip.scope === 'both' || tip.scope === (this.studioFlag() ? 'studio' : 'xreport'));

  @Input() open = false;
  @Input() set studio(value: boolean) { this.studioFlag.set(value); }
  get studio(): boolean { return this.studioFlag(); }
  @Output() readonly closed = new EventEmitter<void>();
  @ViewChild('journeyHost', { read: ViewContainerRef }) private journeyHost?: ViewContainerRef;

  ngAfterViewInit(): void {
    if (!this.journey || !this.journeyHost) return;
    const ref = this.journeyHost.createComponent(this.journey);
    ref.instance.routeRequested.subscribe((route) => this.openRoute(route));
    ref.instance.walkthroughRequested.subscribe(() => this.startDataTrainingTour());
  }

  readonly tips: readonly TipCard[] = [
    {
      icon: 'lucideFileSearch',
      title: 'Read the model contract',
      body: 'Model cards show anatomy, image limits, context and readiness before you generate.',
      route: '/inference',
      routeLabel: 'Open Inference',
      scope: 'xreport',
    },
    {
      icon: 'lucideBrainCircuit',
      title: 'Save checkpoints',
      body: 'Enable checkpoint saving to resume later or run Evaluate Model.',
      route: '/training',
      routeLabel: 'Open Training',
      scope: 'studio',
    },
    {
      icon: 'lucideBarChart2',
      title: 'Sample while exploring',
      body: 'Use a fraction for quick checks; use the full dataset for final decisions.',
      route: '/dataset',
      routeLabel: 'Open Dataset',
      scope: 'studio',
    },
    {
      icon: 'lucideAlertTriangle',
      title: 'Treat output as a draft',
      body: 'Generated text is an editable research-use draft. Check its provenance before copying or exporting.',
      tone: 'warning',
      scope: 'both',
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

  startDataTrainingTour(): void {
    this.closed.emit();
    this.guidance.requestTour(DATA_TRAINING_TOUR_ID);
  }
}
