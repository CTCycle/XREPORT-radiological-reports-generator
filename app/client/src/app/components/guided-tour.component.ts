import { AfterViewInit, Component, ElementRef, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideChevronLeft, lucideChevronRight, lucideX } from '@ng-icons/lucide';
import { GuidanceService } from '../services/guidance.service';
import type { GuidanceDefinition, TourPlacement, TourStep } from '../types/guidance';
import { ModalFocusDirective } from './modal-focus.directive';

interface TargetRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

interface PanelPosition {
  top: number;
  left: number;
}

export type TourCloseReason = 'skipped' | 'completed';

@Component({
  standalone: true,
  selector: 'app-guided-tour',
  imports: [NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideChevronLeft, lucideChevronRight, lucideX })],
  template: `
    @if (open) {
      <div class="guided-tour-backdrop" role="presentation" (click)="skip()">
        @if (targetRect(); as rect) {
          <div
            class="guided-tour-spotlight"
            aria-hidden="true"
            [style.top.px]="rect.top - 6"
            [style.left.px]="rect.left - 6"
            [style.width.px]="rect.width + 12"
            [style.height.px]="rect.height + 12"
          ></div>
        }

        <section
          class="guided-tour-dialog"
          role="dialog"
          aria-modal="true"
          [attr.aria-labelledby]="titleId"
          [attr.aria-describedby]="bodyId"
          [style.top.px]="panelPosition().top"
          [style.left.px]="panelPosition().left"
          appModalFocus
          (modalEscape)="skip()"
          (click)="$event.stopPropagation()"
        >
          <header class="guidance-tour-header">
            <div>
              <div class="guided-tour-progress" aria-live="polite">Step {{ currentStepIndex() + 1 }} of {{ definition.steps.length }}</div>
              <h2 [id]="titleId">{{ currentStep().title }}</h2>
            </div>
            <button type="button" class="guidance-close-button" aria-label="Close walkthrough" (click)="skip()">
              <ng-icon name="lucideX" size="17" aria-hidden="true" />
            </button>
          </header>
          <p class="guided-tour-body" [id]="bodyId">{{ currentStep().body }}</p>
          <footer class="guidance-tour-footer">
            <div class="guided-tour-navigation">
              <button type="button" class="guidance-button" [disabled]="currentStepIndex() === 0" (click)="back()">
                <ng-icon name="lucideChevronLeft" size="15" aria-hidden="true" />Back
              </button>
              <button type="button" class="guidance-button guidance-button-primary" (click)="next()">
                {{ currentStepIndex() === definition.steps.length - 1 ? 'Finish' : 'Next' }}
                @if (currentStepIndex() < definition.steps.length - 1) { <ng-icon name="lucideChevronRight" size="15" aria-hidden="true" /> }
              </button>
            </div>
          </footer>
        </section>
      </div>
    }
  `,
  styleUrl: '../styles/Guidance.css',
})
export class GuidedTourComponent implements AfterViewInit, OnChanges, OnDestroy {
  private static nextId = 0;
  private readonly host = inject(ElementRef<HTMLElement>);
  private readonly guidance = inject(GuidanceService);
  private layoutTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly viewportHandler = () => {
    if (this.open) this.measureTarget();
  };

  @Input({ required: true }) definition!: GuidanceDefinition;
  @Input() open = false;
  @Output() readonly closed = new EventEmitter<TourCloseReason>();

  readonly currentStepIndex = signal(0);
  readonly targetRect = signal<TargetRect | null>(null);
  readonly panelPosition = signal<PanelPosition>({ top: 80, left: 80 });
  readonly titleId = `guided-tour-${GuidedTourComponent.nextId++}-title`;
  readonly bodyId = `${this.titleId}-body`;

  ngAfterViewInit(): void {
    window.addEventListener('resize', this.viewportHandler, { passive: true });
    window.addEventListener('scroll', this.viewportHandler, { passive: true, capture: true });
    if (this.open) this.scheduleLayout();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['open']?.currentValue === true || changes['definition']) {
      this.currentStepIndex.set(0);
      this.scheduleLayout();
    }
  }

  ngOnDestroy(): void {
    window.removeEventListener('resize', this.viewportHandler);
    window.removeEventListener('scroll', this.viewportHandler, true);
    if (this.layoutTimer) clearTimeout(this.layoutTimer);
  }

  currentStep(): TourStep {
    return this.definition.steps[this.currentStepIndex()] ?? this.definition.steps[0];
  }

  back(): void {
    if (this.currentStepIndex() === 0) return;
    this.currentStepIndex.update((index) => index - 1);
    this.scheduleLayout();
  }

  next(): void {
    if (this.currentStepIndex() >= this.definition.steps.length - 1) {
      this.guidance.complete(this.definition.id, this.definition.version);
      this.closed.emit('completed');
      return;
    }
    this.currentStepIndex.update((index) => index + 1);
    this.scheduleLayout();
  }

  skip(): void {
    this.guidance.skip(this.definition.id, this.definition.version);
    this.closed.emit('skipped');
  }

  private scheduleLayout(): void {
    if (!this.open) return;
    if (this.layoutTimer) clearTimeout(this.layoutTimer);
    queueMicrotask(() => {
      if (!this.open) return;
      this.scrollTargetIntoView();
      this.measureTarget();
      this.layoutTimer = setTimeout(() => this.measureTarget(), 140);
    });
  }

  private targetElement(): HTMLElement | null {
    const selector = this.currentStep()?.target;
    if (!selector) return null;
    const documentRef = (this.host.nativeElement as HTMLElement).ownerDocument;
    return documentRef.querySelector(selector) as HTMLElement | null;
  }

  private scrollTargetIntoView(): void {
    const target = this.targetElement();
    if (!target || typeof target.scrollIntoView !== 'function') return;
    const rect = target.getBoundingClientRect();
    const margin = 88;
    if (rect.top < margin || rect.bottom > window.innerHeight - margin) {
      target.scrollIntoView({ block: 'center', behavior: this.prefersReducedMotion() ? 'auto' : 'smooth' });
    }
  }

  private measureTarget(): void {
    if (!this.open) return;
    const target = this.targetElement();
    if (!target) {
      this.targetRect.set(null);
      this.panelPosition.set({
        top: Math.max(16, (window.innerHeight - 220) / 2),
        left: Math.max(16, (window.innerWidth - 360) / 2),
      });
      return;
    }

    const rect = target.getBoundingClientRect();
    const targetRect = { top: rect.top, left: rect.left, width: rect.width, height: rect.height };
    this.targetRect.set(targetRect);
    this.panelPosition.set(this.positionPanel(targetRect, this.currentStep().placement ?? 'bottom'));
  }

  private positionPanel(rect: TargetRect, placement: TourPlacement): PanelPosition {
    const viewportWidth = Math.max(window.innerWidth, 320);
    const viewportHeight = Math.max(window.innerHeight, 320);
    const panelWidth = Math.min(360, viewportWidth - 32);
    const panelHeight = 240;
    const gap = 16;
    let top = rect.top + rect.height + gap;
    let left = rect.left;

    if (placement === 'top') top = rect.top - panelHeight - gap;
    if (placement === 'right') {
      top = rect.top;
      left = rect.left + rect.width + gap;
    }
    if (placement === 'left') {
      top = rect.top;
      left = rect.left - panelWidth - gap;
    }

    if (top + panelHeight > viewportHeight - 16) top = rect.top - panelHeight - gap;
    if (top < 16) top = Math.min(viewportHeight - panelHeight - 16, rect.top + rect.height + gap);
    if (left + panelWidth > viewportWidth - 16) left = viewportWidth - panelWidth - 16;
    if (left < 16) left = 16;

    return { top: Math.max(16, top), left: Math.max(16, left) };
  }

  private prefersReducedMotion(): boolean {
    return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
}
