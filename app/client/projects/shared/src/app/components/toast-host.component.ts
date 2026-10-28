import { Component, inject } from '@angular/core';
import { NotificationService } from '../services/notification.service';

@Component({
  selector: 'app-toast-host',
  template: `
    <div class="toast-host" role="status" aria-live="polite">
      @for (toast of notifications.toasts(); track toast.id) {
        <div class="toast" [class]="'toast toast-' + toast.type">{{ toast.message }}</div>
      }
    </div>
  `,
  styleUrl: '../styles/ToastHost.css',
})
export class ToastHostComponent {
  readonly notifications = inject(NotificationService);
}
