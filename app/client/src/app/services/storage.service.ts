import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class StorageService {
  readString(key: string): string | null {
    try { return localStorage.getItem(key); } catch { return null; }
  }

  writeString(key: string, value: string): void {
    try { localStorage.setItem(key, value); } catch { /* private mode/quota */ }
  }

  readRecord<T>(key: string): Record<string, T> {
    try {
      const raw = this.readString(key);
      if (!raw) return {};
      const parsed: unknown = JSON.parse(raw);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, T> : {};
    } catch { return {}; }
  }

  writeRecord<T>(key: string, value: Record<string, T>): void {
    this.writeString(key, JSON.stringify(value));
  }
}
