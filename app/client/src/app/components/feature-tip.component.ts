import { AfterViewInit, Component, EventEmitter, Input, Output, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideInfo, lucideX } from '@ng-icons/lucide';
import { GuidanceService } from '../services/guidance.service';

@Component({
  standalone: true,
  selector: 'app-feature-tip',
  imports: [NgIcon],
  providers: [provideIcons({ lucideInfo, lucideX })],
  template: `
    @if (visible()) {
      <aside class="guidance-feature-tip" [attr.data-guidance-id]="guidanceId" aria-label="Getting started tip">
        <span class="guidance-feature-tip-icon" aria-hidden="true"><ng-icon name="lucideInfo" size="15" /></span>
        <div class="guidance-feature-tip-content">
          <strong>{{ title }}</strong>
          <p>{{ message }}</p>
          @if (showAction) {
            <div class="guidance-feature-tip-actions">
              <button type="button" class="guidance-button guidance-button-primary" (click)="invokeAction()">{{ actionLabel }}</button>
            </div>
          }
        </div>
        <button type="button" class="guidance-dismiss-button" aria-label="Dismiss tip" (click)="dismiss()">
          <ng-icon name="lucideX" size="16" aria-hidden="true" />
        </button>
      </aside>
    }
  `,
  styleUrl: '../styles/Guidance.css',
})
export class FeatureTipComponent implements AfterViewInit {
  private readonly guidance = inject(GuidanceService);

  @Input({ required: true }) guidanceId!: string;
  @Input() version = 1;
  @Input() title = 'Helpful tip';
  @Input() message = '';
  @Input() actionLabel = 'Show me';
  @Input() showAction = false;
  @Output() readonly action = new EventEmitter<void>();
  @Output() readonly dismissed = new EventEmitter<void>();
  private readonly hidden = signal(false);

  visible(): boolean {
    return !this.hidden() && Boolean(this.guidanceId) && this.guidance.shouldShow(this.guidanceId, this.version);
  }

  ngAfterViewInit(): void {
    if (this.visible()) this.guidance.markSeen(this.guidanceId, this.version);
  }

  dismiss(): void {
    this.hidden.set(true);
    this.guidance.dismiss(this.guidanceId, this.version);
    this.dismissed.emit();
  }

  invokeAction(): void {
    this.hidden.set(true);
    this.action.emit();
  }
}
