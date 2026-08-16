import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class StorageService {
  readRecord<T>(key: string): Record<string, T> {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return {};
      const parsed: unknown = JSON.parse(raw);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, T> : {};
    } catch { return {}; }
  }

  writeRecord<T>(key: string, value: Record<string, T>): void {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode/quota */ }
  }
}
