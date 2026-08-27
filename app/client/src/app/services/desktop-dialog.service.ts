import { Injectable } from '@angular/core';
import { isTauri } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';

@Injectable({ providedIn: 'root' })
export class DesktopDialogService {
  isTauriSurface(): boolean {
    return isTauri();
  }

  async openImageFolder(): Promise<string | null> {
    if (!this.isTauriSurface()) return null;

    const selection = await open({
      directory: true,
      multiple: false,
      title: 'Select image folder',
    });
    return Array.isArray(selection) ? selection[0] ?? null : selection;
  }
}
