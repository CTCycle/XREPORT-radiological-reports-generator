import { DatePipe } from '@angular/common';
import { Component, ElementRef, HostListener, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideBell, lucideCheck, lucideCircleAlert, lucideCircleCheck, lucideInfo, lucideTriangleAlert, lucideX } from '@ng-icons/lucide';
import { NotificationService } from '../services/notification.service';
import type { NotificationType } from '../types/notification';

const TYPE_ICONS: Record<NotificationType, string> = {
  success: 'lucideCircleCheck',
  error: 'lucideCircleAlert',
  warning: 'lucideTriangleAlert',
  info: 'lucideInfo',
};

@Component({
  selector: 'app-notification-center',
  imports: [NgIcon, DatePipe],
  providers: [provideIcons({ lucideBell, lucideCheck, lucideCircleAlert, lucideCircleCheck, lucideInfo, lucideTriangleAlert, lucideX })],
  template: `
    <button type="button" class="notification-trigger" title="Notifications" aria-haspopup="dialog" [attr.aria-expanded]="open()" [attr.aria-label]="'Notifications' + (service.unreadCount() ? ', ' + service.unreadCount() + ' unread' : '')" (click)="open.set(!open())">
      <ng-icon name="lucideBell" size="16" /> <span>Notifications</span>
      @if (service.unreadCount() > 0) { <span class="notification-badge">{{ service.unreadCount() > 99 ? '99+' : service.unreadCount() }}</span> }
    </button>
    @if (open()) {
      <section class="notification-panel" role="dialog" aria-label="Notifications">
        <header class="notification-panel-header">
          <h2>Notifications</h2>
          <div class="notification-panel-actions">
            <button type="button" class="notification-link" [disabled]="service.unreadCount() === 0" (click)="service.markAllRead()">Mark all read</button>
            <button type="button" class="notification-link" [disabled]="service.notifications().length === 0" (click)="service.clear()">Clear all</button>
          </div>
        </header>
        @if (service.notifications().length === 0) {
          <p class="notification-empty">No notifications.</p>
        } @else {
          <ul class="notification-list">
            @for (item of service.notifications(); track item.id) {
              <li class="notification-item" [class.unread]="!item.read">
                <ng-icon [class]="'notification-icon notification-' + item.type" [name]="icon(item.type)" size="15" />
                <div class="notification-body">
                  <span class="notification-message">{{ item.message }}</span>
                  <span class="notification-meta">{{ item.type }} · {{ item.createdAt | date: 'MMM d, HH:mm' }}</span>
                </div>
                <div class="notification-item-actions">
                  @if (!item.read) {
                    <button type="button" class="notification-icon-button" title="Mark as read" aria-label="Mark as read" (click)="service.markRead(item.id)"><ng-icon name="lucideCheck" size="14" /></button>
                  }
                  <button type="button" class="notification-icon-button" title="Dismiss" aria-label="Dismiss" (click)="service.dismiss(item.id)"><ng-icon name="lucideX" size="14" /></button>
                </div>
              </li>
            }
          </ul>
        }
      </section>
    }
  `,
  styleUrl: '../styles/NotificationCenter.css',
})
export class NotificationCenterComponent {
  readonly service = inject(NotificationService);
  readonly open = signal(false);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  icon(type: NotificationType): string {
    return TYPE_ICONS[type];
  }

  @HostListener('document:mousedown', ['$event'])
  onDocumentMouseDown(event: MouseEvent): void {
    if (this.open() && !this.host.nativeElement.contains(event.target as Node)) this.open.set(false);
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.open.set(false);
  }
}
