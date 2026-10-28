import { Component, EventEmitter, Input, Output } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideLoaderCircle, lucideX } from '@ng-icons/lucide';
import { ModalFocusDirective } from '@xreport/shared/app/components/modal-focus.directive';
import { ValidationStatisticsComponent } from './validation-statistics.component';

type ReportStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null | undefined;

@Component({
  standalone: true,
  selector: 'app-validation-report-modal',
  imports: [NgIcon, ModalFocusDirective, ValidationStatisticsComponent],
  providers: [provideIcons({ lucideLoaderCircle, lucideX })],
  template: `
    @if (open) {
      <div class="modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="report-modal" role="dialog" aria-modal="true" aria-labelledby="validation-report-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="report-header">
            <div><h3 id="validation-report-title">Validation Report</h3><p class="report-subtitle">Dataset: <strong>{{ datasetName || 'Unknown' }}</strong></p></div>
            <button type="button" class="report-close" aria-label="Close validation report" (click)="closed.emit()"><ng-icon name="lucideX"/></button>
          </header>
          <div class="report-body">
            @if (loading) {
              <div class="loading-container" role="status"><ng-icon name="lucideLoaderCircle" class="spin"/><span class="loading-text">{{ status === 'pending' ? 'Queued' : 'Running' }} validation... {{ progress ?? 0 }}%</span><div class="progress-bar"><span class="progress-fill" [style.width.%]="progress ?? 0"></span></div></div>
            } @else if (error) {
              <p class="idle-message error" role="alert">{{ error }}</p>
            } @else {
              <app-validation-statistics [datasetName]="datasetName" [refreshToken]="refreshToken"/>
            }
          </div>
        </section>
      </div>
    }
  `,
  styleUrls: ['../styles/ValidationReportModal.css', '../styles/ValidationDashboard.css'],
})
export class ValidationReportModalComponent {
  @Input() open = false;
  @Input() datasetName: string | null = null;
  @Input() loading = false;
  @Input() error: string | null = null;
  @Input() progress: number | null = null;
  @Input() status: ReportStatus = null;
  /** Changes whenever a validation run finishes so the statistics reload. */
  @Input() refreshToken = 0;
  @Output() readonly closed = new EventEmitter<void>();
}
