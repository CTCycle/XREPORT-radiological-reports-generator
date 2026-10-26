export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface AppNotification {
  readonly id: string;
  readonly type: NotificationType;
  readonly message: string;
  readonly createdAt: string;
  readonly read: boolean;
}

export interface NotifyOptions {
  readonly type: NotificationType;
  readonly message: string;
  /** Show a transient toast. Defaults to true. */
  readonly toast?: boolean;
  /** Keep the notification in the history list. Defaults to true. */
  readonly history?: boolean;
}

export interface ToastItem {
  readonly id: string;
  readonly type: NotificationType;
  readonly message: string;
}
