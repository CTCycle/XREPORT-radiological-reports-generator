import { CdkConnectedOverlay, CdkOverlayOrigin, ConnectedPosition } from '@angular/cdk/overlay';
import { Component, ElementRef, EventEmitter, HostListener, Input, Output, ViewChild, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideCircleHelp, lucideX } from '@ng-icons/lucide';

@Component({
  standalone: true,
  selector: 'app-help-popover',
  imports: [CdkConnectedOverlay, CdkOverlayOrigin, NgIcon],
  providers: [provideIcons({ lucideCircleHelp, lucideX })],
  template: `
    <button
      #origin="cdkOverlayOrigin"
      type="button"
      class="guidance-popover-trigger"
      cdkOverlayOrigin
      [attr.aria-label]="label"
      [attr.aria-expanded]="open()"
      [attr.aria-controls]="panelId"
      (click)="toggle()"
    >
      <ng-icon name="lucideCircleHelp" size="15" aria-hidden="true" />
    </button>

    <ng-template
      cdkConnectedOverlay
      [cdkConnectedOverlayOrigin]="origin"
      [cdkConnectedOverlayOpen]="open()"
      [cdkConnectedOverlayPositions]="positions"
      [cdkConnectedOverlayPush]="true"
      [cdkConnectedOverlayViewportMargin]="12"
      [cdkConnectedOverlayHasBackdrop]="true"
      cdkConnectedOverlayBackdropClass="guidance-popover-backdrop"
      (backdropClick)="close()"
      (detach)="close()"
    >
      <section class="guidance-popover-panel" [id]="panelId" role="dialog" [attr.aria-labelledby]="titleId">
        <div class="guidance-popover-header">
          <h2 [id]="titleId">{{ title }}</h2>
          <button #closeButton type="button" class="guidance-close-button" aria-label="Close help" (click)="close()">
            <ng-icon name="lucideX" size="16" aria-hidden="true" />
          </button>
        </div>
        <p>{{ body }}</p>
      </section>
    </ng-template>
  `,
  styleUrl: '../styles/Guidance.css',
})
export class HelpPopoverComponent {
  private static nextId = 0;
  private readonly host = inject(ElementRef<HTMLElement>);
  private returnFocus: HTMLElement | null = null;

  @Input() label = 'More information';
  @Input() title = 'More information';
  @Input() body = '';
  @Output() readonly closed = new EventEmitter<void>();
  @ViewChild('closeButton') private closeButton?: ElementRef<HTMLButtonElement>;

  readonly open = signal(false);
  readonly panelId = `guidance-popover-${HelpPopoverComponent.nextId++}`;
  readonly titleId = `${this.panelId}-title`;
  readonly positions: ConnectedPosition[] = [
    { originX: 'end', originY: 'bottom', overlayX: 'end', overlayY: 'top', offsetY: 8 },
    { originX: 'end', originY: 'top', overlayX: 'end', overlayY: 'bottom', offsetY: -8 },
    { originX: 'start', originY: 'bottom', overlayX: 'start', overlayY: 'top', offsetY: 8 },
    { originX: 'start', originY: 'top', overlayX: 'start', overlayY: 'bottom', offsetY: -8 },
  ];

  toggle(): void {
    if (this.open()) {
      this.close();
      return;
    }

    this.returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    this.open.set(true);
    queueMicrotask(() => this.closeButton?.nativeElement.focus());
  }

  close(): void {
    if (!this.open()) return;
    this.open.set(false);
    this.closed.emit();
    queueMicrotask(() => this.returnFocus?.focus());
  }

  @HostListener('document:keydown.escape', ['$event'])
  onEscape(event: Event): void {
    if (!this.open()) return;
    event.preventDefault();
    this.close();
  }

  @HostListener('document:keydown.tab', ['$event'])
  onTab(event: Event): void {
    if (!this.open()) return;
    const panel = this.host.nativeElement.ownerDocument.getElementById(this.panelId);
    if (!panel || panel.contains((event as KeyboardEvent).target as Node)) return;
    event.preventDefault();
    this.closeButton?.nativeElement.focus();
  }
}
