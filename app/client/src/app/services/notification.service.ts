import { Injectable, computed, inject, signal } from '@angular/core';
import type { AppNotification, NotifyOptions, ToastItem } from '../types/notification';
import { StorageService } from './storage.service';

const STORAGE_KEY = 'xreport.notifications.v1';
const MAX_HISTORY = 50;
const TOAST_DURATION_MS = 3500;
const NOTIFICATION_TYPES = new Set(['success', 'error', 'warning', 'info']);

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private readonly storage = inject(StorageService);
  private sequence = 0;

  readonly notifications = signal<readonly AppNotification[]>(this.load());
  readonly toasts = signal<readonly ToastItem[]>([]);
  readonly unreadCount = computed(() => this.notifications().filter((item) => !item.read).length);

  notify(options: NotifyOptions): void {
    const id = `${Date.now().toString(36)}-${(this.sequence++).toString(36)}`;
    if (options.history !== false) {
      const entry: AppNotification = { id, type: options.type, message: options.message, createdAt: new Date().toISOString(), read: false };
      this.update([entry, ...this.notifications()].slice(0, MAX_HISTORY));
    }
    if (options.toast !== false) {
      this.toasts.update((items) => [...items, { id, type: options.type, message: options.message }]);
      setTimeout(() => this.dismissToast(id), TOAST_DURATION_MS);
    }
  }

  dismissToast(id: string): void {
    this.toasts.update((items) => items.filter((item) => item.id !== id));
  }

  markRead(id: string): void {
    this.update(this.notifications().map((item) => (item.id === id ? { ...item, read: true } : item)));
  }

  markAllRead(): void {
    this.update(this.notifications().map((item) => ({ ...item, read: true })));
  }

  dismiss(id: string): void {
    this.update(this.notifications().filter((item) => item.id !== id));
  }

  clear(): void {
    this.update([]);
  }

  private update(items: readonly AppNotification[]): void {
    this.notifications.set(items);
    this.storage.writeString(STORAGE_KEY, JSON.stringify(items));
  }

  private load(): AppNotification[] {
    try {
      const parsed: unknown = JSON.parse(this.storage.readString(STORAGE_KEY) ?? '[]');
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((item): item is AppNotification =>
          !!item && typeof item.id === 'string' && typeof item.message === 'string' && typeof item.createdAt === 'string'
          && typeof item.read === 'boolean' && NOTIFICATION_TYPES.has(item.type))
        .slice(0, MAX_HISTORY);
    } catch {
      return [];
    }
  }
}
