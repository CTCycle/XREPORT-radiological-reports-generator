import { Component, computed, inject } from '@angular/core';
import type { MemoryUsage } from '../types/systemStatus';
import { SystemStatusService } from '../services/system-status.service';

@Component({
  selector: 'app-system-status-bar',
  template: `
    <section class="system-status" aria-label="System status">
      <span class="system-status-item" [title]="backendTitle()">
        <span class="system-status-led" [class.on]="backend() === true" [class.off]="backend() === false" aria-hidden="true"></span>
        Backend <span class="visually-hidden">{{ backendTitle() }}</span>
      </span>
      <span class="system-status-item" [title]="cudaTitle()">
        <span class="system-status-led" [class.on]="cuda() === true" [class.off]="cuda() === false" aria-hidden="true"></span>
        CUDA <span class="visually-hidden">{{ cudaTitle() }}</span>
      </span>
      <span class="system-status-metric"><span>RAM</span><span>{{ ram() }}</span></span>
      <span class="system-status-metric"><span>VRAM</span><span>{{ vram() }}</span></span>
    </section>
  `,
  styleUrl: '../styles/SystemStatusBar.css',
})
export class SystemStatusBarComponent {
  private readonly service = inject(SystemStatusService);

  readonly backend = this.service.backendOnline;
  readonly cuda = computed(() => {
    const status = this.service.status();
    return status ? status.cuda.available : this.service.backendOnline() === false ? false : null;
  });
  readonly backendTitle = computed(() => {
    const online = this.backend();
    return online === null ? 'Backend: checking' : online ? 'Backend: connected' : 'Backend: disconnected';
  });
  readonly cudaTitle = computed(() => {
    const status = this.service.status();
    if (!status) return 'CUDA: unknown';
    return status.cuda.available ? `CUDA: available${status.cuda.device_name ? ` (${status.cuda.device_name})` : ''}` : 'CUDA: unavailable';
  });
  readonly ram = computed(() => this.format(this.service.status()?.ram));
  readonly vram = computed(() => this.format(this.service.status()?.vram));

  private format(usage: MemoryUsage | null | undefined): string {
    return usage ? `${usage.used_gb.toFixed(1)} / ${usage.total_gb.toFixed(1)} GB` : '—';
  }
}
